"""Bookkeeping shared by the run scripts: seeds, code provenance and run folders."""
from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from src.training.experiments import Transfer
from src.training.train_transfer import ROOT
from src.training.utils import SEED
from .data import checkpoint_dir

# The five paired seeds of the project. SEED (52) is the seed of the supplied source-only checkpoints.
SEEDS = (SEED, 100, 445, 545, 881)
# Defaults of run_transfer, identical in all supplied config.json files.
BASELINE = {"epochs": 30, "batch_size": 32, "lr": 3e-4, "weight_decay": 1e-4}
SOURCE_ONLY_ROOT = ROOT / "results" / "source_only"
# Starting point of an adaptation run. per_seed: the source-only model of the run's own seed (stage 2 so far).
# shared: the supplied seed-52 source-only model for every seed, so the seed only steers the adaptation
# (batch order, dropout, new heads such as the DANN discriminator); results go to SHARED_INIT_ROOT.
INITS = ("per_seed", "shared")
SHARED_INIT_ROOT = ROOT / "results" / "shared_init"
SOURCE_ONLY_SPLITS = ("source_train", "source_val", "source_test", "target_train", "target_val", "target_test")
NONDETERMINISM = ("Trained on Apple MPS. PyTorch does not guarantee bitwise reproducible results on MPS; "
                  "a repeated run with the same seed can differ slightly.")


def clean_commit() -> str:
    """Return HEAD after checking that uda/ and all Python files match it.

    Stops on staged, unstaged and untracked changes, so that the commit written into the
    config.json of a run really describes the code of that run.
    """
    # core.trustctime=false: iCloud changes the ctime of files it offloads, which would make git rehash them.
    status = subprocess.check_output(["git", "-c", "core.trustctime=false", "status", "--porcelain", "--", "uda/", "*.py"],
                                     cwd=ROOT, text=True)
    if status:
        raise RuntimeError(f"Code differs from the last commit, refusing to start a run:\n{status}")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def is_complete(run_dir: Path, splits: tuple[str, ...]) -> bool:
    """True if the folder holds everything a finished run writes. A run writes config.json last."""
    required = [run_dir / "model_last.pt", run_dir / "train_log.csv",
                *(run_dir / "predictions" / f"{split}.npz" for split in splits)]
    if not all(path.is_file() for path in required):
        return False
    try:
        json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return True


def source_only_dir(transfer: Transfer, seed: int) -> Path:
    """Folder of the source-only run of a seed: the supplied one for SEED, otherwise our own."""
    if seed == SEED:
        return checkpoint_dir(transfer)
    return SOURCE_ONLY_ROOT / f"seed_{seed}" / transfer.family / transfer.name


def source_only_checkpoint(transfer: Transfer, seed: int) -> Path:
    """Checkpoint all runs of this seed start from. Refuses unfinished or foreign source-only runs."""
    folder = source_only_dir(transfer, seed)
    if seed != SEED:
        if not is_complete(folder, SOURCE_ONLY_SPLITS):
            raise FileNotFoundError(f"No finished source-only run for seed {seed}: {folder}")
        recorded = json.loads((folder / "config.json").read_text(encoding="utf-8"))["seed"]
        if recorded != seed:
            raise ValueError(f"{folder} was trained with seed {recorded}, expected {seed}")
    return folder / "model_last.pt"


def init_seed(seed: int, init: str) -> int:
    """Seed of the source-only model the runs of `seed` start from, see INITS."""
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}, got {init!r}")
    return SEED if init == "shared" else seed


def init_checkpoint(transfer: Transfer, seed: int, init: str) -> Path:
    """Checkpoint an adaptation run starts from, see INITS."""
    return source_only_checkpoint(transfer, init_seed(seed, init))


def runs_root(init: str, seed: int) -> Path:
    """Folder that holds the adaptation runs of a seed.

    With the shared start the seed-52 runs are the stage-2 runs in results/ (they started from the supplied
    checkpoint already); the runs of the other seeds are in SHARED_INIT_ROOT.
    """
    if init_seed(seed, init) == seed:  # per_seed, or seed 52 with the shared start
        return ROOT / "results"
    return SHARED_INIT_ROOT


def staging_dir(run_dir: Path) -> Path:
    """Fresh, empty folder next to run_dir. A run writes all its files there; publish_run moves it into place."""
    work = run_dir.with_name(run_dir.name + ".tmp")
    if work.exists():
        shutil.rmtree(work)  # left over from an interrupted run
    work.mkdir(parents=True)
    return work


def publish_run(work: Path, run_dir: Path) -> None:
    """Move a finished run into place with one rename, so run_dir is either complete or absent."""
    if run_dir.exists():
        shutil.rmtree(run_dir)  # unfinished run of the old writing scheme, or --overwrite
    os.replace(work, run_dir)


def environment(device: torch.device) -> dict:
    return {"device": str(device), "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "nondeterminism": NONDETERMINISM}


def write_train_log(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
