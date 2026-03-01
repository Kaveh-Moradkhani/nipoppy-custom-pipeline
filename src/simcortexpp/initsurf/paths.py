import os

def t1_mni_path(preproc_root: str, subject: str, ses: str = "01", space: str = "MNI152") -> str:
    return os.path.join(
        preproc_root, subject, f"ses-{ses}", "anat",
        f"{subject}_ses-{ses}_space-{space}_desc-preproc_T1w.nii.gz"
    )

def seg9_dseg_path(seg_root: str, subject: str, ses: str = "01", space: str = "MNI152") -> str:
    return os.path.join(
        seg_root, subject, f"ses-{ses}", "anat",
        f"{subject}_ses-{ses}_space-{space}_desc-seg9_dseg.nii.gz"
    )

def out_anat_dir(out_root: str, subject: str, ses: str = "01") -> str:
    return os.path.join(out_root, subject, f"ses-{ses}", "anat")

def out_surf_dir(out_root: str, subject: str, ses: str = "01") -> str:
    return os.path.join(out_root, subject, f"ses-{ses}", "surfaces")
