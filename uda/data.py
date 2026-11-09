"""Data and checkpoint access on top of the supervisor's IO in src/training (read only)."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import torch

from src.training.experiments import TRANSFERS, Transfer, resolve_group
from src.training.model import SelfHARCNN
from src.training.train_transfer import ROOT, _arrays, _load

# Supplied assets, supervisor source and environment: never written by our code.
PROTECTED_DIRS = ("models", "data", "src", ".git", ".venv")


def get_transfer(name: str) -> Transfer:
    for transfer in TRANSFERS:
        if transfer.name == name:
            return transfer
    raise ValueError(f"Unknown transfer: {name}")


def load_labeled(family: str, name: str) -> tuple[dict, dict, dict]:
    """Features, labels and metadata of one group, per split (train/val/test).

    Uses the supervisor's _load and _arrays, so the window order is the same as in
    run_transfer and in the supplied predictions.
    """
    features, labels, meta = _load(family, name)
    return _arrays(features), _arrays(labels), meta


def load_unlabeled(family: str, name: str) -> dict:
    """Features of one group per split, without labels.

    Same folder logic as src/training/train_transfer._load and the same concatenation
    (_arrays), but only features.pkl is opened, never evaluation_labels.pkl.
    """
    dataset, group = resolve_group(name)
    folder = ROOT / "data" / "prepared" / family / dataset
    if group:
        folder /= group
    with (folder / "features.pkl").open("rb") as handle:
        return _arrays(pickle.load(handle))


def checkpoint_dir(transfer: Transfer) -> Path:
    """Folder of the supplied source-only checkpoint of a transfer."""
    return ROOT / "models" / transfer.family / transfer.name


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_label_map(transfer: Transfer) -> dict:
    return json.loads((checkpoint_dir(transfer) / "config.json").read_text())["label_map"]


def load_source_only_model(transfer: Transfer) -> SelfHARCNN:
    """Build SelfHARCNN and load the supplied source-only weights (the file is only read)."""
    model = SelfHARCNN((150, 18), len(load_label_map(transfer)))
    state = torch.load(checkpoint_dir(transfer) / "model_last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def check_output_dir(path: Path) -> Path:
    """Resolve an output folder and refuse folders inside the supplied assets or source."""
    path = path.resolve()
    if any(path.is_relative_to(ROOT / folder) for folder in PROTECTED_DIRS):
        raise ValueError(f"Refusing to write into supplied assets or source folders: {path}")
    return path
