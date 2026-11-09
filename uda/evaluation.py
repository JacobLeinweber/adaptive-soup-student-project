"""Common evaluation for the baseline and all UDA runs.

This is the only module of our code that reads target labels. It reads them from the
supervisor's prepared files after training and uses them for scoring only.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.training.experiments import Transfer
from .data import checkpoint_dir, load_label_map, load_labeled

# The four splits every method is scored on.
SPLITS = ("source_test", "target_train", "target_val", "target_test")
METRICS = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")


def predict(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """Predicted class per window. Needs no labels.

    Same procedure as src/training/metrics.evaluate, which produced the baseline predictions:
    eval mode (BatchNorm running statistics, no dropout), no_grad, batches of 512, argmax.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            predictions.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """Accuracy, macro-F1 and weighted F1 over the fixed class list, balanced accuracy."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    labels = list(range(num_classes))
    return {
        "n_windows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def true_labels(transfer: Transfer) -> dict[str, np.ndarray]:
    """True labels of the four splits, in the window order of the prepared files."""
    label_map = load_label_map(transfer)
    labels = {}
    for domain, name in (("source", transfer.source), ("target", transfer.target)):
        _, by_split, meta = load_labeled(transfer.family, name)
        if meta["label_map"] != label_map:
            raise ValueError(f"Label map of {name} does not match the checkpoint of {transfer.name}")
        for split in ("train", "val", "test"):
            key = f"{domain}_{split}"
            if key in SPLITS:
                labels[key] = np.asarray(by_split[split], dtype=np.int64)
    return labels


def score_baseline(transfer: Transfer, labels: dict[str, np.ndarray]) -> dict[str, dict]:
    """Metrics of the supplied source-only model, from its stored predictions (no new run)."""
    num_classes = len(load_label_map(transfer))
    results = {}
    for split in SPLITS:
        with np.load(checkpoint_dir(transfer) / "predictions" / f"{split}.npz") as stored:
            if not np.array_equal(stored["y_true"], labels[split]):
                raise ValueError(f"Stored labels are not in data order: {transfer.name}/{split}")
            results[split] = compute_metrics(labels[split], stored["y_pred"], num_classes)
    return results


def score_run(run_dir, transfer: Transfer, labels: dict[str, np.ndarray]) -> dict[str, dict]:
    """Metrics of one of our runs, from the label-free predictions it saved."""
    num_classes = len(load_label_map(transfer))
    results = {}
    for split in SPLITS:
        with np.load(run_dir / "predictions" / f"{split}.npz") as stored:
            results[split] = compute_metrics(labels[split], stored["y_pred"], num_classes)
    return results
