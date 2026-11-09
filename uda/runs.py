"""Bookkeeping shared by the run scripts: seeds, code provenance and run folders."""
from __future__ import annotations

import csv
import json
import platform
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
SOURCE_ONLY_SPLITS = ("source_train", "source_val", "source_test", "target_train", "target_val", "target_test")
NONDETERMINISM = ("Trained on Apple MPS. PyTorch does not guarantee bitwise reproducible results on MPS; "
                  "a repeated run with the same seed can differ slightly.")


def clean_commit() -> str:
    """Return HEAD after checking that uda/ and all Python files match it.

    Stops on staged, unstaged and untracked changes, so that the commit written into the
    config.json of a run really describes the code of that run.
    """
    status = subprocess.check_output(["git", "status", "--porcelain", "--", "uda/", "*.py"], cwd=ROOT, text=True)
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


def environment(device: torch.device) -> dict:
    return {"device": str(device), "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "nondeterminism": NONDETERMINISM}


def write_train_log(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
