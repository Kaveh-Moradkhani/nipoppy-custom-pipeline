import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.metrics import compute_surface_dice
from omegaconf import OmegaConf

from simcortexpp.seg.data.dataloader import EvalSegDataset


def setup_logger(log_dir: str, filename: str = "seg_eval.log"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / filename
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        force=True,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)


def dice_np(gt: np.ndarray, pred: np.ndarray, num_classes: int, exclude_background: bool = True, eps: float = 1e-6) -> float:
    dices: List[float] = []
    start_cls = 1 if exclude_background else 0
    for c in range(start_cls, num_classes):
        gt_c = (gt == c)
        pred_c = (pred == c)
        inter = np.logical_and(gt_c, pred_c).sum()
        union = gt_c.sum() + pred_c.sum()
        if union == 0:
            continue
        dices.append((2.0 * inter + eps) / (union + eps))
    return float(np.mean(dices)) if dices else 0.0


def accuracy_np(gt: np.ndarray, pred: np.ndarray) -> float:
    return float((gt == pred).sum() / gt.size)


def nsd_monai(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    tolerance_vox: float = 1.0,
    include_background: bool = False,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    gt_t = torch.from_numpy(gt).long().unsqueeze(0)
    pred_t = torch.from_numpy(pred).long().unsqueeze(0)

    gt_1h = F.one_hot(gt_t, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    pred_1h = F.one_hot(pred_t, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    n_thr = num_classes if include_background else (num_classes - 1)
    class_thresholds = [float(tolerance_vox)] * n_thr

    nsd_per_class = compute_surface_dice(
        y_pred=pred_1h,
        y=gt_1h,
        class_thresholds=class_thresholds,
        include_background=include_background,
        distance_metric="euclidean",
        spacing=spacing,
        use_subvoxels=False,
    )[0]

    vals = nsd_per_class[~torch.isnan(nsd_per_class)]
    return float(vals.mean().item()) if vals.numel() else 0.0


def _get_map(cfg_node, keys: Tuple[str, ...]) -> Optional[Dict[str, str]]:
    for k in keys:
        v = getattr(cfg_node, k, None)
        if v is not None and hasattr(v, "items"):
            return {str(kk): str(vv) for kk, vv in v.items()}
    return None


def _cache_per_dataset_csvs(split_csv: str, cache_dir: Path, roots: Dict[str, str]) -> Dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv)
    req = {"subject", "split", "dataset"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"split_file must contain columns {sorted(req)}. Got: {list(df.columns)}")

    out_map: Dict[str, str] = {}
    for ds_name in roots.keys():
        out = cache_dir / f"split_{ds_name}.csv"
        df_ds = df[df["dataset"].astype(str).str.strip() == ds_name][["subject", "split"]]
        if df_ds.empty:
            logging.warning(f"No rows for dataset='{ds_name}' in {split_csv}")
            continue
        df_ds.to_csv(out, index=False)
        out_map[ds_name] = str(out)

    if not out_map:
        raise RuntimeError(f"No cached per-dataset split files found in {cache_dir}")
    return out_map


def build_eval_datasets(cfg):
    ds_cfg = cfg.dataset
    split_csv = str(ds_cfg.split_file)
    split_name = str(ds_cfg.split_name)
    session_label = str(getattr(ds_cfg, "session_label", "01"))
    space = str(getattr(ds_cfg, "space", "MNI152"))

    roots_map = _get_map(ds_cfg, ("roots", "dataset_roots", "deriv_roots"))
    pred_roots_map = _get_map(cfg.outputs, ("pred_roots", "out_roots", "pred_out_roots"))

    if roots_map is not None:
        if pred_roots_map is None:
            raise ValueError("For multi-dataset eval, set outputs.pred_roots (map) to BIDS derivatives roots for predictions.")
        missing = [k for k in roots_map.keys() if k not in pred_roots_map]
        if missing:
            raise ValueError(f"outputs.pred_roots missing keys: {missing}")

        cache_dir = Path(cfg.outputs.log_dir) / "split_cache"
        per_ds_csv = _cache_per_dataset_csvs(split_csv, cache_dir, roots_map)

        items = []
        for ds_name, deriv_root in roots_map.items():
            if ds_name not in per_ds_csv:
                continue
            pred_root = pred_roots_map[ds_name]
            ds = EvalSegDataset(
                deriv_root=str(deriv_root),
                split_csv=str(per_ds_csv[ds_name]),
                pred_root=str(pred_root),
                split_name=split_name,
                session_label=session_label,
                space=space,
            )
            items.append((ds_name, ds))
        if not items:
            raise RuntimeError("No datasets constructed for eval. Check dataset names in split_file vs cfg.dataset.roots keys.")
        return items

    if getattr(ds_cfg, "path", None) is None:
        raise ValueError("Single-dataset eval requires dataset.path if dataset.roots is not provided.")

    if getattr(cfg.outputs, "pred_root", None) is None:
        raise ValueError("Single-dataset eval requires outputs.pred_root if outputs.pred_roots is not provided.")

    ds = EvalSegDataset(
        deriv_root=str(ds_cfg.path),
        split_csv=str(ds_cfg.split_file),
        pred_root=str(cfg.outputs.pred_root),
        split_name=split_name,
        session_label=session_label,
        space=space,
    )
    return [("SINGLE", ds)]


@hydra.main(version_base="1.3", config_path="pkg://simcortexpp.configs.seg", config_name="eval")
def main(cfg):
    setup_logger(cfg.outputs.log_dir, "seg_eval.log")
    logging.info("=== Segmentation Eval config ===")
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    num_classes = int(cfg.evaluation.num_classes)
    exclude_bg = bool(cfg.evaluation.exclude_background)
    nsd_tol = float(getattr(cfg.evaluation, "nsd_tolerance_vox", 1.0))
    spacing = tuple(getattr(cfg.evaluation, "spacing", (1.0, 1.0, 1.0)))

    datasets = build_eval_datasets(cfg)

    records = []
    n_total = 0
    n_failed = 0

    for ds_name, ds in datasets:
        logging.info(f"[{ds_name}] Evaluating {len(ds)} subjects on split={cfg.dataset.split_name}")
        for i in range(len(ds)):
            try:
                gt9, pred_arr, sub, ses = ds[i]
                d = dice_np(gt9, pred_arr, num_classes=num_classes, exclude_background=exclude_bg)
                acc = accuracy_np(gt9, pred_arr)
                nsd = nsd_monai(
                    gt9,
                    pred_arr,
                    num_classes=num_classes,
                    tolerance_vox=nsd_tol,
                    include_background=False,
                    spacing=spacing,
                )
                records.append(
                    {
                        "subject": sub,
                        "session": ses,
                        "dataset": ds_name,
                        "dice": d,
                        "accuracy": acc,
                        "nsd": nsd,
                    }
                )
                logging.info(f"[{ds_name}] {sub} {ses}: Dice={d:.4f}, Acc={acc:.4f}, NSD={nsd:.4f}")
                n_total += 1
            except Exception as e:
                logging.warning(f"[{ds_name}] Failed: index={i} err={repr(e)}")
                n_failed += 1

    if not records:
        logging.warning("No subjects evaluated.")
        return

    df = pd.DataFrame(records)

    overall = df[["dice", "accuracy", "nsd"]].agg(["mean", "std"])
    by_dataset = df.groupby("dataset")[["dice", "accuracy", "nsd"]].agg(["count", "mean", "std"])
    by_dataset.columns = [f"{m}_{s}" for (m, s) in by_dataset.columns]  # flatten MultiIndex
    by_dataset = by_dataset.reset_index()

    logging.info(
        f"OVERALL mean±std | Dice={overall.loc['mean','dice']:.4f}±{overall.loc['std','dice']:.4f} | "
        f"Acc={overall.loc['mean','accuracy']:.4f}±{overall.loc['std','accuracy']:.4f} | "
        f"NSD={overall.loc['mean','nsd']:.4f}±{overall.loc['std','nsd']:.4f}"
    )
    logging.info(f"Done. Evaluated={n_total} Failed={n_failed}")

    out_csv = Path(cfg.outputs.eval_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved per-subject metrics to {out_csv}")

    out_xlsx = getattr(cfg.outputs, "eval_xlsx", None)
    if out_xlsx:
        out_xlsx = Path(str(out_xlsx))
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="per_subject", index=False)
            by_dataset.to_excel(w, sheet_name="summary_by_dataset", index=False)
            overall.reset_index().rename(columns={"index": "stat"}).to_excel(w, sheet_name="summary_overall", index=False)
        logging.info(f"Saved Excel report to {out_xlsx}")


if __name__ == "__main__":
    main()
