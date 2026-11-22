"""Hyperparameter search on top of the Stage-2 training loops: combinations, training call and criteria.

Nothing here changes how a method trains. train_run calls the Stage-2 loops (uda.training.train_continued,
uda.dann.train_dann, uda.dsan.train_dsan) with the learning rate and the two loss weights of one combination;
epochs, batch size, weight decay, seed, start model and forward passes stay as in Stage 2.

The combinations themselves are fixed in the search configuration (configs/hp_search_cross_dataset.json,
written once by make_hp_search_config.py), so every machine runs exactly the same values.

Criteria of a finished model, all from eval-mode outputs:
    SRC     cross-entropy and metrics on source val (labelled source data only)
    DEV     deep embedded validation as in uda.criteria.dev_risk, plus its raw quantities
    FST     few-shot target: a fixed set of 5 labelled target-val windows per class (uses target labels)
    ORACLE  all of target val with its labels (uses target labels; reference only)
Target train and target test are scored for the report only and are kept apart from the criteria.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .criteria import DEV_LEARNING_RATES, dev_value, dev_weights, per_window_ce
from .dann import train_dann
from .dsan import train_dsan
from .evaluation import compute_metrics
from .training import train_continued

METHODS = ("deep_coral", "dann", "dsan")
# Where a combination comes from: 0 = Stage-2 setting, 1 = AdaTime hparams.py, from 2 on = random draws.
ORIGINS = ("stage2", "adatime_hparams", "random")
# Alignment loss of each method, i.e. what alignment_weight multiplies.
ALIGNMENT_LOSS = {"deep_coral": "CORAL", "dann": "domain loss (discriminator, gradient reversal)", "dsan": "LMMD"}
# Splits whose predictions every run saves; none of them needs labels.
OUTPUT_SPLITS = ("source_val", "source_test", "target_train", "target_val", "target_test")


@dataclass(frozen=True)
class Combination:
    id: int
    origin: str
    learning_rate: float
    source_ce_weight: float
    alignment_weight: float


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_digest(config: dict) -> str:
    """SHA-256 of the configuration content, independent of formatting and line endings."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def combinations(config: dict, method: str) -> list[Combination]:
    return [Combination(int(row["id"]), row["origin"], float(row["learning_rate"]), float(row["source_ce_weight"]),
                        float(row["alignment_weight"])) for row in config["combinations"][method]]


def config_problems(config: dict) -> list[str]:
    """Structural check of a search configuration; returns a list of problems (empty if usable)."""
    required = ("name", "family", "transfers", "methods", "seed", "fixed", "combinations", "inputs", "few_shot", "dev")
    missing = [key for key in required if key not in config]
    if missing:
        return [f"configuration: missing keys {missing}"]
    problems = []
    fixed = config["fixed"]
    for key in ("epochs", "batch_size", "weight_decay"):
        if key not in fixed:
            problems.append(f"configuration: fixed.{key} is missing")
    unknown = [m for m in config["methods"] if m not in METHODS]
    if unknown:
        problems.append(f"configuration: unknown methods {unknown}")
    counts = set()
    for method in config["methods"]:
        rows = config["combinations"].get(method, [])
        if not rows:
            problems.append(f"configuration: no combinations for {method}")
        counts.add(len(rows))
        if [row.get("id") for row in rows] != list(range(len(rows))):
            problems.append(f"configuration: combination ids of {method} are not 0, 1, 2, ...")
        for row in rows:
            values = [row.get(key) for key in ("learning_rate", "source_ce_weight", "alignment_weight")]
            if row.get("origin") not in ORIGINS or not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values):
                problems.append(f"configuration: invalid combination {method} {row}")
    if len(counts) > 1:
        problems.append(f"configuration: the methods have different numbers of combinations {sorted(counts)}")
    return problems


def train_run(method: str, combination: Combination, model, x_source: np.ndarray, y_source: np.ndarray,
              x_target: np.ndarray, *, epochs: int, batch_size: int, weight_decay: float, seed: int, device, log):
    """Continue training `model` in place with one combination; return (epoch log, DANN discriminator or None).

    Only target windows are passed in; there is no parameter for target labels.
    """
    common = dict(source_ce_weight=combination.source_ce_weight, epochs=epochs, batch_size=batch_size,
                  lr=combination.learning_rate, weight_decay=weight_decay, seed=seed, device=device, log=log)
    if method == "deep_coral":
        return train_continued(model, x_source, y_source, x_target, coral_weight=combination.alignment_weight, **common), None
    if method == "dann":
        return train_dann(model, x_source, y_source, x_target, domain_weight=combination.alignment_weight, **common)
    if method == "dsan":
        return train_dsan(model, x_source, y_source, x_target, lmmd_weight=combination.alignment_weight, **common), None
    raise ValueError(f"Unknown method: {method}")


def few_shot_indices(labels: np.ndarray, num_classes: int, per_class: int, seed: int) -> tuple[np.ndarray, list[int]]:
    """Up to `per_class` windows of every class, drawn without replacement with numpy default_rng(seed).

    Returns the indices (per class in ascending order, classes in order) and the number taken per class.
    Like AdaTime's few_shot_data_generator, a class with fewer windows contributes all of them.
    """
    rng = np.random.default_rng(seed)
    chosen, taken = [], []
    for label in range(num_classes):
        candidates = np.flatnonzero(np.asarray(labels) == label)
        k = min(per_class, len(candidates))
        chosen.append(np.sort(rng.choice(candidates, size=k, replace=False)) if k else np.empty(0, dtype=np.int64))
        taken.append(int(k))
    return np.concatenate(chosen).astype(np.int64), taken


def split_scores(prefix: str, logits: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    """Mean cross-entropy over the windows and the metrics of uda.evaluation.compute_metrics, keys prefixed."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = {"ce": float(per_window_ce(logits, labels).mean()),
              **compute_metrics(labels, np.asarray(logits).argmax(1), num_classes)}
    return {f"{prefix}_{key}": value for key, value in scores.items()}


def dev_with_raw(source_features: np.ndarray, target_features: np.ndarray, validation_features: np.ndarray,
                 validation_errors: np.ndarray, seed: int) -> tuple[dict, np.ndarray]:
    """DEV exactly as uda.criteria.dev_risk, and the importance weight of every source-val window.

    With the returned weights and the per-window errors DEV can be recomputed later (for example with
    clipped or normalised weights) without training the domain classifier again.
    """
    weights, details = dev_weights(source_features, target_features, validation_features, seed)
    accuracies = {f"dev_domain_accuracy_lr{lr:g}": acc
                  for lr, acc in zip(DEV_LEARNING_RATES, details.pop("dev_domain_accuracies"))}
    if not np.isfinite(weights).all():
        result = {"dev_risk": math.nan, "dev_eta": math.nan, **details, **accuracies,
                  "dev_weight_mean": math.nan, "dev_weight_max": math.nan, "dev_weights_finite": False}
    else:
        risk, eta = dev_value(weights, validation_errors)
        result = {"dev_risk": risk, "dev_eta": eta, **details, **accuracies, "dev_weight_mean": float(weights.mean()),
                  "dev_weight_max": float(weights.max()), "dev_weights_finite": True}
    return result, weights
