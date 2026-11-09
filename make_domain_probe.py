"""Domain probe, a diagnostic only: can a new linear classifier tell source from target on frozen features?

For every run the saved model produces its 96-dim features in eval mode for the source-train and
target-train windows. Equally many windows per domain go into a logistic regression with 5-fold
cross-validation. 50 % means the probe cannot separate the domains, 100 % means it separates them
perfectly. Only the domain of a window is used, no activity labels, and the result selects nothing.

Tables in results/tables/:
    <prefix>_domain_probe_runs.csv      transfer x seed x variant, mean accuracy over the 5 folds
    <prefix>_domain_probe_summary.csv   mean and sd over the five seeds, per transfer and per group

Examples:
    python make_domain_probe.py
    python make_domain_probe.py --prefix T3 --variants baseline deep_coral dsan dsan_lambda0_target_pass
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from make_dann_tables import VARIANTS, scopes
from make_results_tables import write_csv
from make_seed_tables import TABLES, run_dir
from src.training.experiments import TRANSFERS
from src.training.model import SelfHARCNN
from uda.data import check_output_dir, load_source_only_model, load_unlabeled
from uda.features import extract_features
from uda.runs import SEEDS


def frozen_features(model: SelfHARCNN, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """96-dim features in eval mode (BatchNorm running statistics, no dropout), no gradients."""
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            parts.append(extract_features(model, batch).cpu().numpy())
    return np.concatenate(parts)


def probe_accuracy(source: np.ndarray, target: np.ndarray, seed: int) -> tuple[float, int]:
    """Mean accuracy over 5 folds and the number of windows per domain.

    The larger domain is subsampled without replacement. The scaler is fitted inside each fold,
    so the test fold of a split does not influence it.
    """
    rng = np.random.default_rng(seed)
    n = min(len(source), len(target))
    x = np.concatenate([source[rng.choice(len(source), n, replace=False)],
                        target[rng.choice(len(target), n, replace=False)]])
    y = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(probe, x, y, cv=folds, scoring="accuracy").mean()), n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--prefix", default="T2", help="prefix of the table names")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    tables = check_output_dir(TABLES)

    accuracy, rows = {}, []
    for transfer in TRANSFERS:
        source = load_unlabeled(transfer.family, transfer.source)["train"]
        target = load_unlabeled(transfer.family, transfer.target)["train"]
        for seed in SEEDS:
            for variant in args.variants:
                model = load_source_only_model(transfer, run_dir(variant, transfer, seed) / "model_last.pt").to(device)
                value, n = probe_accuracy(frozen_features(model, source, device), frozen_features(model, target, device), seed)
                accuracy[transfer.name, seed, variant] = value
                rows.append([transfer.family, transfer.name, seed, variant, n, f"{value:.6f}"])
        print(f"done: {transfer.name}", flush=True)

    summary = []
    for kind, family, name, members in scopes():
        for variant in args.variants:
            per_seed = [np.mean([accuracy[member, seed, variant] for member in members]) for seed in SEEDS]
            summary.append([kind, family, name, variant, len(members), len(SEEDS),
                            f"{np.mean(per_seed):.6f}", f"{np.std(per_seed, ddof=1):.6f}"])
    write_csv(tables / f"{args.prefix}_domain_probe_runs.csv",
              ["family", "transfer", "seed", "variant", "windows_per_domain", "probe_accuracy"], rows)
    write_csv(tables / f"{args.prefix}_domain_probe_summary.csv",
              ["scope", "family", "name", "variant", "n_transfers", "n_seeds", "probe_accuracy_mean", "probe_accuracy_sd"], summary)


if __name__ == "__main__":
    main()
