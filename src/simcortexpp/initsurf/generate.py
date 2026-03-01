import os
import json
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from tqdm import tqdm
from nibabel.affines import apply_affine

import hydra
from omegaconf import DictConfig, OmegaConf

import trimesh
from trimesh.collision import CollisionManager

from skimage.filters import gaussian
from skimage.measure import marching_cubes
from skimage.measure import label as compute_cc
from scipy.ndimage import distance_transform_edt as edt
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.special import expit

from simcortexpp.initsurf.paths import (
    t1_mni_path, seg9_dseg_path, out_anat_dir, out_surf_dir
)

# Topology corrector
from simcortexpp.utils.tca import topology

log = logging.getLogger("scpp.initsurf")

def save_nifti(data: np.ndarray, affine: np.ndarray, out_path: str, dtype=np.float32):
    img = nib.Nifti1Image(np.asarray(data, dtype=dtype), affine)
    nib.save(img, out_path)

def write_dataset_description(root: str, name: str = "scpp-initsurf", version: str = "0.1"):
    path = os.path.join(root, "dataset_description.json")
    if os.path.exists(path):
        return
    dd = {
        "Name": name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "SimCortexPP", "Version": version}],
    }
    os.makedirs(root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dd, f, indent=2)

def separate_hemispheres(seg_mask: np.ndarray, gap_size: int = 2) -> np.ndarray:
    lh_wm_mask = (seg_mask == 1) | (seg_mask == 7)
    rh_wm_mask = (seg_mask == 2) | (seg_mask == 8)
    struct = generate_binary_structure(3, 2)
    dilated_left = binary_dilation(lh_wm_mask, structure=struct, iterations=gap_size)
    dilated_right = binary_dilation(rh_wm_mask, structure=struct, iterations=gap_size)
    collision_zone = dilated_left & dilated_right
    new_seg = seg_mask.copy()
    new_seg[collision_zone] = 0
    return new_seg

def build_wm_masks_from_labels(seg_npy: np.ndarray):
    lh = (seg_npy == 1) | (seg_npy == 7)
    rh = (seg_npy == 2) | (seg_npy == 8)
    return lh.astype(np.uint8), rh.astype(np.uint8)

def compute_sdf(binary_seg: np.ndarray, sigma: float = 0.5, keep_largest: bool = True) -> np.ndarray:
    binary_seg = (binary_seg > 0).astype(np.uint8)
    cc, nc = compute_cc(binary_seg, connectivity=2, return_num=True)
    if nc == 0:
        raise ValueError("No connected components found")
    if keep_largest:
        volumes = np.bincount(cc.ravel())[1:]
        cc_id = 1 + int(np.argmax(volumes))
        seg = (cc == cc_id).astype(np.uint8)
    else:
        seg = (cc > 0).astype(np.uint8)
    sdf = (-edt(seg) + edt(1 - seg)).astype(np.float32)
    sdf = gaussian(sdf, sigma=sigma, preserve_range=True).astype(np.float32)
    return sdf

def sdf_to_probability(sdf: np.ndarray, beta: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    prob = expit(-beta * sdf)
    return np.clip(prob, eps, 1.0 - eps).astype(np.float32)

def laplacian_smooth(verts, faces, lambd=1.0):
    v = verts[0]
    f = faces[0]
    with torch.no_grad():
        V = v.shape[0]
        edge = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0).T
        L = torch.sparse_coo_tensor(edge, torch.ones_like(edge[0]).float(), (V, V))
        norm_w = 1.0 / torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
        v_bar = L.mm(v) * norm_w
    return ((1 - lambd) * v + lambd * v_bar).unsqueeze(0)

def meshes_collide(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> bool:
    cm = CollisionManager()
    cm.add_object("a", mesh_a)
    cm.add_object("b", mesh_b)
    return bool(cm.in_collision_internal())

def pial_vs_wm_collide(pial_mesh: trimesh.Trimesh, wm_mesh: trimesh.Trimesh) -> bool:
    cm = CollisionManager()
    cm.add_object("wm", wm_mesh)
    cm.add_object("pial", pial_mesh)
    return bool(cm.in_collision_internal())

topo_correct = topology()

def prepare_topo_sdf(sdf: np.ndarray, topo_threshold: float) -> np.ndarray:
    sdf = np.asarray(sdf, dtype=np.float32)
    sdf_topo = topo_correct.apply(sdf, threshold=np.float32(topo_threshold))
    return np.asarray(sdf_topo, dtype=np.float32)

def mesh_from_topo_sdf(sdf_topo: np.ndarray, level: float, brain_affine: np.ndarray, n_smooth: int) -> trimesh.Trimesh:
    v_mc, f_mc, _, _ = marching_cubes(-sdf_topo, level=-float(level), method="lorensen")
    v_mc = torch.tensor(v_mc.copy(), dtype=torch.float32).unsqueeze(0)
    f_mc = torch.tensor(f_mc.copy(), dtype=torch.long).unsqueeze(0)
    for _ in range(n_smooth):
        v_mc = laplacian_smooth(v_mc, f_mc, lambd=1.0)
    v_np = v_mc[0].cpu().numpy()
    f_np = f_mc[0].cpu().numpy()
    verts_world = apply_affine(brain_affine, v_np)
    return trimesh.Trimesh(vertices=verts_world, faces=f_np, process=False)

def free_collision_wm_from_topo(
    lh_wm_topo, rh_wm_topo, brain_affine,
    start_level, step, min_level, max_iters, n_smooth
):
    lvl = float(start_level)
    last_l = last_r = None

    for _ in range(1, max_iters + 1):
        mesh_l = mesh_from_topo_sdf(lh_wm_topo, level=lvl, brain_affine=brain_affine, n_smooth=n_smooth)
        mesh_r = mesh_from_topo_sdf(rh_wm_topo, level=lvl, brain_affine=brain_affine, n_smooth=n_smooth)

        cm = CollisionManager()
        cm.add_object("lh", mesh_l)
        cm.add_object("rh", mesh_r)

        if not cm.in_collision_internal():
            log.info(f"[WM] Collision-free at lvl={lvl:.3f}")
            return mesh_l, mesh_r, lvl

        last_l, last_r = mesh_l, mesh_r
        lvl += step
        if lvl < min_level:
            break

    log.warning(f"[WM] FAILED collision-free. returning last at lvl={lvl:.3f}")
    return (last_l if last_l is not None else mesh_l), (last_r if last_r is not None else mesh_r), lvl

def find_pial_level_no_collision_with_wm_topo(pial_topo, wm_mesh, brain_affine, start_level, step, max_level, max_iters, n_smooth):
    lvl = float(start_level)
    last_mesh = None
    for _ in range(1, max_iters + 1):
        pial_mesh = mesh_from_topo_sdf(pial_topo, level=lvl, brain_affine=brain_affine, n_smooth=n_smooth)
        last_mesh = pial_mesh
        if not pial_vs_wm_collide(pial_mesh, wm_mesh):
            return pial_mesh, lvl
        lvl += step
        if lvl > max_level:
            break
    log.warning(f"[Pial-vs-WM] FAILED, returning last at level={lvl:.3f}")
    return last_mesh, lvl

def free_collision_pial_joint_topo(
    lh_pial_topo, rh_pial_topo, wm_l_mesh, wm_r_mesh, brain_affine,
    start_level, step, max_level, max_iters,
    shrink_step, shrink_max_iters, n_smooth
):
    pial_l, lvl_l = find_pial_level_no_collision_with_wm_topo(
        lh_pial_topo, wm_l_mesh, brain_affine, start_level, step, max_level, max_iters, n_smooth
    )
    pial_r, lvl_r = find_pial_level_no_collision_with_wm_topo(
        rh_pial_topo, wm_r_mesh, brain_affine, start_level, step, max_level, max_iters, n_smooth
    )

    for it in range(1, shrink_max_iters + 1):
        if not meshes_collide(pial_l, pial_r):
            log.info(f"[Pial-LR] collision-free with levels L={lvl_l:.3f}, R={lvl_r:.3f}")
            return pial_l, pial_r, lvl_l, lvl_r

        def try_shrink_left(curr_lvl):
            new_lvl = curr_lvl - shrink_step
            new_mesh = mesh_from_topo_sdf(lh_pial_topo, level=new_lvl, brain_affine=brain_affine, n_smooth=n_smooth)
            if pial_vs_wm_collide(new_mesh, wm_l_mesh):
                return None, curr_lvl
            return new_mesh, new_lvl

        def try_shrink_right(curr_lvl):
            new_lvl = curr_lvl - shrink_step
            new_mesh = mesh_from_topo_sdf(rh_pial_topo, level=new_lvl, brain_affine=brain_affine, n_smooth=n_smooth)
            if pial_vs_wm_collide(new_mesh, wm_r_mesh):
                return None, curr_lvl
            return new_mesh, new_lvl

        tried = False
        if lvl_l >= lvl_r:
            nm, nl = try_shrink_left(lvl_l)
            if nm is not None:
                pial_l, lvl_l = nm, nl
                tried = True
            else:
                nm, nl = try_shrink_right(lvl_r)
                if nm is not None:
                    pial_r, lvl_r = nm, nl
                    tried = True
        else:
            nm, nl = try_shrink_right(lvl_r)
            if nm is not None:
                pial_r, lvl_r = nm, nl
                tried = True
            else:
                nm, nl = try_shrink_left(lvl_l)
                if nm is not None:
                    pial_l, lvl_l = nm, nl
                    tried = True

        log.info(f"[Pial-LR] collision -> shrink iter={it} | L={lvl_l:.3f}, R={lvl_r:.3f}")
        if not tried:
            log.warning("[Pial-LR] infeasible to resolve collisions without breaking pial-vs-WM constraints")
            break

    return pial_l, pial_r, lvl_l, lvl_r

@hydra.main(config_path="pkg://simcortexpp.configs.initsurf", config_name="generate", version_base=None)
def main(cfg: DictConfig):
    print("=== InitSurf generate config ===")
    print(OmegaConf.to_yaml(cfg))

    df = pd.read_csv(cfg.dataset.split_file)

    split_name = str(cfg.dataset.split_name)
    if split_name != "all":
        df = df[df["split"].astype(str) == split_name].copy()

    subjects = df["subject"].astype(str).tolist()
    log.info(f"InitSurf: {len(subjects)} subjects (split={split_name})")

    ses = str(cfg.dataset.session_label)
    space = str(cfg.dataset.space)

    n_smooth = int(cfg.params.n_smooth)
    topo_thr = float(cfg.params.topo_threshold)

    for ds_key, out_root in cfg.outputs.out_roots.items():
        write_dataset_description(str(out_root), name="scpp-initsurf", version="0.1")

    for subject_id in tqdm(subjects):
        ds_key = str(df.loc[df["subject"].astype(str) == subject_id, "dataset"].iloc[0])

        preproc_root = str(cfg.dataset.roots[ds_key])
        seg_root = str(cfg.dataset.seg_roots[ds_key])
        out_root = str(cfg.outputs.out_roots[ds_key])

        brain_path = t1_mni_path(preproc_root, subject_id, ses=ses, space=space)
        pred_path = seg9_dseg_path(seg_root, subject_id, ses=ses, space=space)

        if not os.path.exists(brain_path):
            log.warning(f"[{subject_id}] Missing preproc T1 -> skip: {brain_path}")
            continue
        if not os.path.exists(pred_path):
            log.warning(f"[{subject_id}] Missing seg9 dseg -> skip: {pred_path}")
            continue

        brain = nib.load(brain_path)
        affine = brain.affine

        seg_pred = nib.load(pred_path).get_fdata(dtype=np.float32).astype(np.uint8)

        anat_dir = out_anat_dir(out_root, subject_id, ses=ses)
        surf_dir = out_surf_dir(out_root, subject_id, ses=ses)
        os.makedirs(anat_dir, exist_ok=True)
        os.makedirs(surf_dir, exist_ok=True)

        save_nifti(seg_pred, affine, os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-seg9_dseg_used.nii.gz"), dtype=np.uint8)

        seg_clean = separate_hemispheres(seg_pred, gap_size=int(cfg.params.gap_size))
        save_nifti(seg_clean, affine, os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-seg9_dseg_cleaned.nii.gz"), dtype=np.uint8)

        lh_mask, rh_mask = build_wm_masks_from_labels(seg_clean)

        try:
            lh_wm_sdf_raw = compute_sdf(lh_mask, sigma=float(cfg.params.sdf_sigma), keep_largest=True)
            rh_wm_sdf_raw = compute_sdf(rh_mask, sigma=float(cfg.params.sdf_sigma), keep_largest=True)
        except ValueError as e:
            log.error(f"[{subject_id}] SDF Error: {e}")
            continue

        lh_wm_topo = prepare_topo_sdf(lh_wm_sdf_raw, topo_threshold=topo_thr)
        rh_wm_topo = prepare_topo_sdf(rh_wm_sdf_raw, topo_threshold=topo_thr)

        mesh_l_wm, mesh_r_wm, wm_iso = free_collision_wm_from_topo(
            lh_wm_topo, rh_wm_topo, affine,
            start_level=float(cfg.params.wm_start_level),
            step=float(cfg.params.wm_step),
            min_level=float(cfg.params.wm_min_level),
            max_iters=int(cfg.params.wm_max_iters),
            n_smooth=n_smooth,
        )

        shift = -float(wm_iso)
        lh_wm_sdf = (lh_wm_sdf_raw + np.float32(shift)).astype(np.float32)
        rh_wm_sdf = (rh_wm_sdf_raw + np.float32(shift)).astype(np.float32)

        wm_thick = float(cfg.params.wm_thickness)
        lh_pial_base = (lh_wm_sdf - np.float32(wm_thick)).astype(np.float32)
        rh_pial_base = (rh_wm_sdf - np.float32(wm_thick)).astype(np.float32)

        lh_pial_topo = prepare_topo_sdf(lh_pial_base, topo_threshold=topo_thr)
        rh_pial_topo = prepare_topo_sdf(rh_pial_base, topo_threshold=topo_thr)

        mesh_l_pial, mesh_r_pial, pial_iso_l, pial_iso_r = free_collision_pial_joint_topo(
            lh_pial_topo, rh_pial_topo, mesh_l_wm, mesh_r_wm, affine,
            start_level=float(cfg.params.pial_start_level),
            step=float(cfg.params.pial_step),
            max_level=float(cfg.params.pial_max_level),
            max_iters=int(cfg.params.pial_max_iters),
            shrink_step=float(cfg.params.pial_shrink_step),
            shrink_max_iters=int(cfg.params.pial_shrink_max_iters),
            n_smooth=n_smooth,
        )

        lh_pial_sdf = (lh_pial_base - np.float32(pial_iso_l)).astype(np.float32)
        rh_pial_sdf = (rh_pial_base - np.float32(pial_iso_r)).astype(np.float32)

        mesh_l_wm.export(os.path.join(surf_dir, f"{subject_id}_ses-{ses}_space-{space}_hemi-L_white.surf.ply"))
        mesh_r_wm.export(os.path.join(surf_dir, f"{subject_id}_ses-{ses}_space-{space}_hemi-R_white.surf.ply"))
        mesh_l_pial.export(os.path.join(surf_dir, f"{subject_id}_ses-{ses}_space-{space}_hemi-L_pial.surf.ply"))
        mesh_r_pial.export(os.path.join(surf_dir, f"{subject_id}_ses-{ses}_space-{space}_hemi-R_pial.surf.ply"))

        save_nifti(lh_wm_sdf, affine, os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-lh_white_sdf.nii.gz"))
        save_nifti(rh_wm_sdf, affine, os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-rh_white_sdf.nii.gz"))
        save_nifti(lh_pial_sdf, affine, os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-lh_pial_sdf.nii.gz"))
        save_nifti(rh_pial_sdf, affine, os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-rh_pial_sdf.nii.gz"))

        lh_ribbon = (lh_pial_sdf <= 0) & (~(lh_wm_sdf <= 0))
        rh_ribbon = (rh_pial_sdf <= 0) & (~(rh_wm_sdf <= 0))
        ribbon_mask = (lh_ribbon | rh_ribbon).astype(np.uint8)

        ribbon_sdf = compute_sdf(ribbon_mask, sigma=float(cfg.params.sdf_sigma), keep_largest=False)
        ribbon_prob = sdf_to_probability(ribbon_sdf, beta=1.0)

        save_nifti(ribbon_sdf, affine, os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-ribbon_sdf.nii.gz"))
        save_nifti(ribbon_prob, affine, os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-ribbon_prob.nii.gz"))

        log.info(f"[{ds_key}][{subject_id}] OK | wm_iso={wm_iso:.3f} pialL={pial_iso_l:.3f} pialR={pial_iso_r:.3f}")

    log.info("InitSurf generation finished.")

if __name__ == "__main__":
    main()