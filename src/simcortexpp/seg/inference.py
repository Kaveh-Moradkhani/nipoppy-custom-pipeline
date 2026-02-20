#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from simcortexpp.seg.models.unet import Unet
from simcortexpp.seg.data.dataloader import PredictSegDataset


# -------------------------
# Logging
# -------------------------
def setup_logger(log_dir: str, filename: str = "inference.log") -> None:
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


# -------------------------
# Checkpoint helpers
# -------------------------
def _strip_module_prefix(state_dict: dict) -> dict:
    """Load checkpoints saved with DataParallel/DDP that may contain 'module.' prefixes."""
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _load_checkpoint_strict(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    logging.info(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()


# -------------------------
# BIDS helpers
# -------------------------
def _norm_ses(s: Any) -> str:
    """Normalize session label to 'ses-XX'."""
    s = str(s)
    return s if s.startswith("ses-") else f"ses-{s}"


def _get_pkg_version(pkg_name: str) -> str:
    try:
        import importlib.metadata as importlib_metadata
        return importlib_metadata.version(pkg_name)
    except Exception:
        return "0.0.0"


def _write_dataset_description(deriv_root: Path, name: str, version: str) -> None:
    """Create dataset_description.json if missing (BIDS derivatives requirement)."""
    deriv_root.mkdir(parents=True, exist_ok=True)
    p = deriv_root / "dataset_description.json"
    if p.exists():
        return
    desc = {
        "Name": name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "SimCortexPP",
                "Version": version,
                "Description": "3D U-Net 9-class segmentation inference in MNI space",
            }
        ],
        "GeneratedOn": str(date.today()),
    }
    p.write_text(json.dumps(desc, indent=2))


# -------------------------
# Multi-dataset support
# -------------------------
def _get_roots_map(ds_cfg) -> Optional[Dict[str, str]]:
    val = getattr(ds_cfg, "roots", None)
    if val is not None and hasattr(val, "items"):
        return {str(k): str(v) for k, v in val.items()}
    return None


def _cache_per_dataset_csvs(split_csv: str, cache_dir: Path, roots: Dict[str, str]) -> Dict[str, str]:
    """
    Given a combined CSV with columns: subject, split, dataset,
    write per-dataset CSVs with columns: subject, split.
    """
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
        raise RuntimeError(f"No per-dataset split files created in: {cache_dir}")

    return out_map


class _TagDataset(Dataset):
    """Attach a dataset name to each sample so we can route outputs per dataset."""

    def __init__(self, base: Dataset, ds_name: str):
        self.base = base
        self.ds_name = ds_name

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        vol, sub, ses, affine, orig_shape = self.base[idx]
        return vol, sub, ses, affine, orig_shape, self.ds_name


def _resolve_out_root(cfg, ds_name: Optional[str]) -> Path:
    """
    Multi-dataset mode: outputs.out_roots is a mapping {DATASET: PATH}
    Single-dataset mode: outputs.out_root is a string path
    """
    if ds_name is not None:
        if not hasattr(cfg.outputs, "out_roots"):
            raise ValueError("Multi-dataset inference requires outputs.out_roots mapping.")
        out_roots = cfg.outputs.out_roots
        if ds_name not in out_roots:
            raise KeyError(f"outputs.out_roots missing key '{ds_name}'. Keys: {list(out_roots.keys())}")
        return Path(str(out_roots[ds_name]))
    if not hasattr(cfg.outputs, "out_root"):
        raise ValueError("Single-dataset inference requires outputs.out_root.")
    return Path(str(cfg.outputs.out_root))


# -------------------------
# Main
# -------------------------
@hydra.main(version_base="1.3", config_path="pkg://simcortexpp.configs.seg", config_name="inference")
def main(cfg) -> None:
    setup_logger(cfg.outputs.log_dir, "inference.log")
    logging.info("=== Inference config ===")
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    device = torch.device(str(cfg.trainer.device)) if torch.cuda.is_available() else torch.device("cpu")
    pin_memory = torch.cuda.is_available()

    roots_map = _get_roots_map(cfg.dataset)

    # Build dataset(s)
    if roots_map is not None:
        cache_dir = Path(cfg.outputs.log_dir) / "split_cache"
        per_ds_csv = _cache_per_dataset_csvs(str(cfg.dataset.split_file), cache_dir, roots_map)

        dsets = []
        for ds_name, root in roots_map.items():
            if ds_name not in per_ds_csv:
                continue
            base = PredictSegDataset(
                deriv_root=str(root),
                split_csv=str(per_ds_csv[ds_name]),
                split_name=str(cfg.dataset.split_name),
                session_label=str(cfg.dataset.session_label),
                space=str(cfg.dataset.space),
                pad_mult=int(cfg.dataset.pad_mult),
            )
            dsets.append(_TagDataset(base, ds_name))

        if not dsets:
            raise RuntimeError("No datasets constructed. Check dataset.roots keys and split_file dataset values.")

        ds: Dataset = dsets[0] if len(dsets) == 1 else ConcatDataset(dsets)
    else:
        ds = PredictSegDataset(
            deriv_root=str(cfg.dataset.path),
            split_csv=str(cfg.dataset.split_file),
            split_name=str(cfg.dataset.split_name),
            session_label=str(cfg.dataset.session_label),
            space=str(cfg.dataset.space),
            pad_mult=int(cfg.dataset.pad_mult),
        )

    dl = DataLoader(
        ds,
        batch_size=int(cfg.trainer.batch_size),
        shuffle=False,
        num_workers=int(cfg.trainer.num_workers),
        pin_memory=pin_memory,
    )

    # Prepare outputs (BIDS derivatives)
    version = _get_pkg_version("simcortexpp")
    if roots_map is not None:
        for ds_name in roots_map.keys():
            out_root = _resolve_out_root(cfg, ds_name)
            _write_dataset_description(out_root, name=f"SimCortexPP Segmentation ({ds_name})", version=version)
    else:
        out_root = _resolve_out_root(cfg, None)
        _write_dataset_description(out_root, name="SimCortexPP Segmentation", version=version)

    # Model
    model = Unet(c_in=int(cfg.model.in_channels), c_out=int(cfg.model.out_channels))
    _load_checkpoint_strict(model, str(cfg.model.ckpt_path), device)

    writer = SummaryWriter(str(cfg.outputs.log_dir))

    processed = 0
    with torch.no_grad():
        pbar = tqdm(dl, desc="Inferring", total=len(dl))
        for step, batch in enumerate(pbar):
            if roots_map is not None:
                vol, sub, ses, affine, orig_shape, ds_name = batch
            else:
                vol, sub, ses, affine, orig_shape = batch
                ds_name = [None] * int(vol.shape[0])

            vol = vol.to(device, non_blocking=True)  # [B,1,D',H',W']

            shapes = orig_shape.cpu().numpy() if isinstance(orig_shape, torch.Tensor) else np.array(orig_shape)
            affines = affine.cpu().numpy() if isinstance(affine, torch.Tensor) else np.array(affine)

            logits = model(vol)                       # [B,C,D',H',W']
            pred = logits.argmax(dim=1).cpu().numpy() # [B,D',H',W']

            for b in range(pred.shape[0]):
                sid = str(sub[b])
                ses_b = _norm_ses(ses[b])

                D, H, W = shapes[b].tolist()
                pred_b = pred[b, :D, :H, :W].astype(np.int16)

                out_root = _resolve_out_root(cfg, ds_name[b])
                out_dir = out_root / sid / ses_b / "anat"
                out_dir.mkdir(parents=True, exist_ok=True)

                stem = f"{sid}_{ses_b}"
                out_path = out_dir / f"{stem}_space-{cfg.dataset.space}_desc-seg9_dseg.nii.gz"

                out_img = nib.Nifti1Image(pred_b, affines[b])
                nib.save(out_img, str(out_path))

                logging.info(f"[{ds_name[b] if ds_name[b] is not None else 'SINGLE'}] Saved: {out_path}")
                processed += 1

            writer.add_scalar("inference/processed_subjects", processed, step)

    writer.close()
    logging.info("Inference finished.")


if __name__ == "__main__":
    main()
