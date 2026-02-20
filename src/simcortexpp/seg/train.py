import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import hydra

from simcortexpp.seg.data.dataloader import SegDataset
from simcortexpp.seg.models.unet import Unet


# -------------------------
# Metrics / losses
# -------------------------
def _state_dict(model: nn.Module) -> dict:
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def dice_score(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    exclude_bg: bool = True,
    eps: float = 1e-6,
) -> float:
    with torch.no_grad():
        pred = logits.argmax(1)  # [B,D,H,W]
        pred_1h = F.one_hot(pred, num_classes).permute(0, 4, 1, 2, 3).float()
        y_1h = F.one_hot(y, num_classes).permute(0, 4, 1, 2, 3).float()
        if exclude_bg and num_classes > 1:
            pred_1h = pred_1h[:, 1:]
            y_1h = y_1h[:, 1:]
        pred_f = pred_1h.flatten(2)
        y_f = y_1h.flatten(2)
        inter = (pred_f * y_f).sum(-1)
        union = pred_f.sum(-1) + y_f.sum(-1)
        return ((2 * inter + eps) / (union + eps)).mean().item()


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return (logits.argmax(1) == y).float().mean().item()


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, exclude_bg: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.exclude_bg = exclude_bg
        self.eps = eps

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, 1)  # [B,C,D,H,W]
        y_1h = F.one_hot(y, self.num_classes).permute(0, 4, 1, 2, 3).float()
        if self.exclude_bg and self.num_classes > 1:
            p = p[:, 1:]
            y_1h = y_1h[:, 1:]
        p_f = p.flatten(2)
        y_f = y_1h.flatten(2)
        inter = (p_f * y_f).sum(-1)
        union = p_f.sum(-1) + y_f.sum(-1)
        dice = (2 * inter + self.eps) / (union + self.eps)
        return (1.0 - dice).mean()


# -------------------------
# DDP
# -------------------------
def setup_ddp(cfg) -> Tuple[int, int, bool, int]:
    use_ddp = bool(getattr(cfg.trainer, "use_ddp", False))
    if use_ddp and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world, True, local_rank
    return 0, 1, False, 0


def is_main(rank: int) -> bool:
    return rank == 0


def setup_logging(log_dir: str, rank: int):
    os.makedirs(log_dir, exist_ok=True)
    kwargs = dict(
        level=logging.INFO if is_main(rank) else logging.WARNING,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        force=True,
    )
    if is_main(rank):
        logging.basicConfig(filename=os.path.join(log_dir, "train_seg.log"), **kwargs)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger("").addHandler(console)
    else:
        logging.basicConfig(**kwargs)


def set_seed(seed: int, deterministic: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------
# Multi-dataset builder
# -------------------------
def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _get_roots_map(ds_cfg) -> Optional[Dict[str, str]]:
    for key in ("roots", "dataset_roots", "deriv_roots"):
        val = getattr(ds_cfg, key, None)

        if val is not None and hasattr(val, "items"):
            return {str(k): str(v) for k, v in val.items()}
            
    return None


def _cache_per_dataset_csvs(
    split_csv: str,
    cache_dir: Path,
    roots: Dict[str, str],
    rank: int,
    is_ddp: bool,
) -> Dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    if is_main(rank):
        df = pd.read_csv(split_csv)
        req = {"subject", "split", "dataset"}
        if not req.issubset(set(df.columns)):
            raise ValueError(
                f"Multi-dataset split_file must contain columns {sorted(req)}. Got: {list(df.columns)}"
            )

        for ds_name in roots.keys():
            out = cache_dir / f"split_{ds_name}.csv"
            df_ds = df[df["dataset"].astype(str).str.strip() == ds_name][["subject", "split"]]
            if df_ds.empty:
                logging.warning(f"No rows for dataset='{ds_name}' in {split_csv}")
                continue
            df_ds.to_csv(out, index=False)

    if is_ddp:
        dist.barrier()

    out_map: Dict[str, str] = {}
    for ds_name in roots.keys():
        p = cache_dir / f"split_{ds_name}.csv"
        if p.exists():
            out_map[ds_name] = str(p)
    if not out_map:
        raise RuntimeError(f"No cached per-dataset split files found in {cache_dir}")
    return out_map


def build_dataset(cfg, split: str, rank: int, is_ddp: bool):
    ds_cfg = cfg.dataset
    pad_mult = int(getattr(ds_cfg, "pad_mult", 16))
    session_label = str(getattr(ds_cfg, "session_label", "01"))
    space = str(getattr(ds_cfg, "space", "MNI152"))
    augment = bool(getattr(ds_cfg, "augment", False)) and split == str(getattr(ds_cfg, "train_split", "train"))

    roots_map = _get_roots_map(ds_cfg)
    paths = _as_list(getattr(ds_cfg, "path", None))
    split_files = _as_list(getattr(ds_cfg, "split_file", None))

    # Mode A: combined split CSV + roots map
    if roots_map is not None:
        if len(split_files) != 1:
            raise ValueError(
                "When cfg.dataset.roots is provided, cfg.dataset.split_file must be a single combined CSV."
            )
        cache_dir = Path(cfg.outputs.log_dir) / "split_cache"
        per_ds_csv = _cache_per_dataset_csvs(str(split_files[0]), cache_dir, roots_map, rank, is_ddp)

        dsets = []
        for ds_name, root in roots_map.items():
            if ds_name not in per_ds_csv:
                continue
            dsets.append(
                SegDataset(
                    deriv_root=root,
                    split_csv=per_ds_csv[ds_name],
                    split=split,
                    session_label=session_label,
                    space=space,
                    pad_mult=pad_mult,
                    augment=augment,
                )
            )
        if not dsets:
            raise RuntimeError("No datasets constructed. Check dataset names in split_file vs cfg.dataset.roots keys.")
        return dsets[0] if len(dsets) == 1 else ConcatDataset(dsets)

    # Mode B: list of datasets (one split file per path)
    if len(paths) > 1:
        if len(split_files) != len(paths):
            raise ValueError(
                "For multi-dataset list mode, provide one split_file per dataset path (same length)."
            )
        dsets = [
            SegDataset(
                deriv_root=str(root),
                split_csv=str(csv),
                split=split,
                session_label=session_label,
                space=space,
                pad_mult=pad_mult,
                augment=augment,
            )
            for root, csv in zip(paths, split_files)
        ]
        return ConcatDataset(dsets)

    # Mode C: single dataset
    if len(paths) != 1 or len(split_files) != 1:
        raise ValueError("Single-dataset mode requires one dataset.path and one dataset.split_file.")
    return SegDataset(
        deriv_root=str(paths[0]),
        split_csv=str(split_files[0]),
        split=split,
        session_label=session_label,
        space=space,
        pad_mult=pad_mult,
        augment=augment,
    )


# -------------------------
# Train entry
# -------------------------
@hydra.main(version_base="1.3", config_path="pkg://simcortexpp.configs.seg", config_name="train")
def main(cfg):
    rank, world, is_ddp, local_rank = setup_ddp(cfg)
    setup_logging(cfg.outputs.log_dir, rank)

    if is_main(rank):
        logging.info("=== Segmentation config ===")
        logging.info("\n" + OmegaConf.to_yaml(cfg))
        if bool(getattr(cfg.trainer, "use_ddp", False)) and not is_ddp:
            logging.warning(
                "use_ddp=true but torchrun env vars not found -> running single-process. Use torchrun for DDP."
            )

    seed = int(getattr(cfg.trainer, "seed", 0))
    deterministic = bool(getattr(cfg.trainer, "deterministic", False))
    if seed:
        set_seed(seed, deterministic)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if is_ddp else str(cfg.trainer.device))
    else:
        device = torch.device("cpu")

    os.makedirs(cfg.outputs.ckpt_dir, exist_ok=True)

    train_split = str(getattr(cfg.dataset, "train_split", "train"))
    val_split = str(getattr(cfg.dataset, "val_split", "val"))

    train_ds = build_dataset(cfg, train_split, rank, is_ddp)
    val_ds = build_dataset(cfg, val_split, rank, is_ddp)

    if is_main(rank):
        logging.info(f"Train samples={len(train_ds)} | Val samples={len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False) if is_ddp else None

    pin_memory = torch.cuda.is_available()
    nw = int(cfg.trainer.num_workers)

    train_dl = DataLoader(
        train_ds,
        batch_size=int(cfg.trainer.batch_size),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=(nw > 0),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(cfg.trainer.batch_size),
        shuffle=False,
        sampler=val_sampler,
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=(nw > 0),
    )

    num_classes = int(cfg.model.out_channels)
    model = Unet(c_in=int(cfg.model.in_channels), c_out=num_classes).to(device)

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    elif torch.cuda.device_count() > 1 and bool(getattr(cfg.trainer, "data_parallel", False)):
        model = nn.DataParallel(model)

    loss_ce = nn.CrossEntropyLoss()
    loss_dice = DiceLoss(num_classes=num_classes, exclude_bg=True)
    dice_w = float(getattr(cfg.trainer, "dice_weight", 1.0))
    opt = optim.Adam(model.parameters(), lr=float(cfg.trainer.learning_rate))

    writer = SummaryWriter(cfg.outputs.log_dir) if is_main(rank) else None

    best_dice = -1.0
    best_epoch = -1
    save_interval = int(getattr(cfg.trainer, "save_interval", 0))
    val_every = int(getattr(cfg.trainer, "validation_interval", 1))

    for epoch in range(1, int(cfg.trainer.num_epochs) + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        totals = torch.zeros(4, device=device)  # loss, dice, acc, n
        pbar = tqdm(train_dl, desc=f"[train] ep{epoch}", leave=False, disable=not is_main(rank))
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            b = x.size(0)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_ce(logits, y) + dice_w * loss_dice(logits, y)
            loss.backward()
            opt.step()

            d = dice_score(logits, y, num_classes)
            a = accuracy(logits, y)

            totals[0] += loss.item() * b
            totals[1] += d * b
            totals[2] += a * b
            totals[3] += b

            if is_main(rank):
                pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.3f}", acc=f"{a:.3f}")

        if is_ddp:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        loss_m = totals[0].item() / max(totals[3].item(), 1.0)
        dice_m = totals[1].item() / max(totals[3].item(), 1.0)
        acc_m = totals[2].item() / max(totals[3].item(), 1.0)

        if is_main(rank) and writer is not None:
            logging.info(f"Epoch {epoch:03d} TRAIN | loss={loss_m:.4f} dice={dice_m:.4f} acc={acc_m:.4f}")
            writer.add_scalar("train/loss", loss_m, epoch)
            writer.add_scalar("train/dice", dice_m, epoch)
            writer.add_scalar("train/acc", acc_m, epoch)

        if save_interval and (epoch % save_interval == 0) and is_main(rank):
            ckpt = Path(cfg.outputs.ckpt_dir) / f"seg_epoch_{epoch:03d}.pt"
            torch.save(_state_dict(model), ckpt)
            logging.info(f"Saved checkpoint: {ckpt}")

        if epoch % val_every != 0:
            continue

        model.eval()
        vtot = torch.zeros(4, device=device)
        vpbar = tqdm(val_dl, desc=f"[val] ep{epoch}", leave=False, disable=not is_main(rank))
        with torch.no_grad():
            for x, y in vpbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                b = x.size(0)
                logits = model(x)
                vloss = loss_ce(logits, y) + dice_w * loss_dice(logits, y)
                d = dice_score(logits, y, num_classes)
                a = accuracy(logits, y)
                vtot[0] += vloss.item() * b
                vtot[1] += d * b
                vtot[2] += a * b
                vtot[3] += b

        if is_ddp:
            dist.all_reduce(vtot, op=dist.ReduceOp.SUM)

        vloss_m = vtot[0].item() / max(vtot[3].item(), 1.0)
        vdice_m = vtot[1].item() / max(vtot[3].item(), 1.0)
        vacc_m = vtot[2].item() / max(vtot[3].item(), 1.0)

        if is_main(rank):
            logging.info(f"Epoch {epoch:03d} VAL   | loss={vloss_m:.4f} dice={vdice_m:.4f} acc={vacc_m:.4f}")
            if writer is not None:
                writer.add_scalar("val/loss", vloss_m, epoch)
                writer.add_scalar("val/dice", vdice_m, epoch)
                writer.add_scalar("val/acc", vacc_m, epoch)

            if vdice_m > best_dice:
                best_dice = vdice_m
                best_epoch = epoch
                best_path = Path(cfg.outputs.ckpt_dir) / "seg_best_dice.pt"
                torch.save(_state_dict(model), best_path)
                logging.info(f"🌟 Best updated: epoch={best_epoch} dice={best_dice:.4f} -> {best_path}")

    if writer is not None:
        writer.close()
    if is_ddp:
        dist.destroy_process_group()

    if is_main(rank):
        logging.info(f"Done. Best epoch={best_epoch}, best val dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
