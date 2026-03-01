from __future__ import annotations

import sys
import subprocess
import typer

from simcortexpp.preproc.fs_to_mni import app as fs_to_mni_app
from simcortexpp.initsurf.generate import main as initsurf_generate


app = typer.Typer(help="SimCortexPP (SCPP) CLI")

# Preprocessing
app.add_typer(fs_to_mni_app, name="fs-to-mni", help="FreeSurfer -> MNI preprocessing")

# Segmentation group
seg_app = typer.Typer(help="Segmentation (3D U-Net) stage")
app.add_typer(seg_app, name="seg")

initsurf_app = typer.Typer(help="Initial surface generation (from seg predictions)")


@seg_app.command("train")
def seg_train(
    overrides: list[str] = typer.Argument(None, help="Hydra overrides, e.g. dataset.path=... trainer.use_ddp=true"),
    torchrun: bool = typer.Option(False, "--torchrun", help="Launch training with torchrun for DDP"),
    nproc_per_node: int = typer.Option(1, "--nproc-per-node", help="GPUs per node for torchrun"),
):
    """
    Run segmentation training.
    - Single GPU: scpp seg train ...
    - Multi GPU:  scpp seg train --torchrun --nproc-per-node 2 trainer.use_ddp=true ...
    """
    if torchrun:
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            "-m",
            "simcortexpp.seg.train",
        ]
    else:
        cmd = [sys.executable, "-m", "simcortexpp.seg.train"]

    cmd += (overrides or [])
    raise typer.Exit(subprocess.call(cmd))


@seg_app.command("infer")
def seg_infer(overrides: list[str] = typer.Argument(None)):
    """Run segmentation inference (Hydra)."""
    cmd = [sys.executable, "-m", "simcortexpp.seg.inference"] + (overrides or [])
    raise typer.Exit(subprocess.call(cmd))


@seg_app.command("eval")
def seg_eval(overrides: list[str] = typer.Argument(None)):
    """Run segmentation evaluation (Hydra)."""
    cmd = [sys.executable, "-m", "simcortexpp.seg.eval"] + (overrides or [])
    raise typer.Exit(subprocess.call(cmd))

@initsurf_app.command("generate")
def generate(overrides: list[str] = typer.Argument(None)):
    # overrides comes like: ["dataset.split_name=all", ...]
    if overrides is None:
        overrides = []
    # Hydra entrypoint expects sys.argv style; easiest: temporarily patch sys.argv
    import sys
    sys.argv = [sys.argv[0]] + overrides
    initsurf_generate()

app.add_typer(initsurf_app, name="initsurf")



if __name__ == "__main__":
    app()
