"""Check whether DEV (deep embedded validation, ported from AdaTime) gives usable values on our splits.

For the seed-52 runs with the AdaTime weights (Deep CORAL, DANN, DSAN and their controls, 12 transfers),
DEV is computed three times with different seeds of its domain classifier (uda.criteria.dev_risk). No
target labels are read: the domain classifier sees source-train and target-train windows, the errors
are the cross-entropies on source validation.

DEV estimates the target risk only if the importance weights are sound. Two checks follow from that:
the mean weight over the source validation windows should be close to 1 (the weights are a density
ratio), and a risk estimate of the cross-entropy cannot be negative. The third check is whether the
choice between a method and its own control stays the same when only the seed of the domain
classifier changes.

Tables in results/tables/:
    T7_dev_check.csv            one row per run: DEV for three classifier seeds and the weight diagnostics
    T7_dev_check_summary.csv    the three checks over all runs

Example: python make_dev_check.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from make_seed_tables import TABLES
from make_sensitivity_tables import ADATIME_RUNS, all_runs
from src.training.experiments import TRANSFERS
from src.training.train_transfer import ROOT
from src.training.utils import SEED
from uda.criteria import dev_risk, features_and_logits, per_window_ce
from uda.data import check_output_dir, load_labeled, load_source_only_model, load_unlabeled

DEV_SEEDS = (SEED, 100, 445)


def check_rows(device: torch.device) -> pd.DataFrame:
    runs = [run for run in all_runs() if run.convention == "adatime"]
    rows = []
    for transfer in TRANSFERS:
        source_x, source_y, _ = load_labeled(transfer.family, transfer.source)
        target_x = load_unlabeled(transfer.family, transfer.target)  # windows only, no labels
        for run in runs:
            model = load_source_only_model(transfer, run.folder(transfer) / "model_last.pt").to(device)
            source_features, _ = features_and_logits(model, source_x["train"], device)
            target_features, _ = features_and_logits(model, target_x["train"], device)
            val_features, val_logits = features_and_logits(model, source_x["val"], device)
            errors = per_window_ce(val_logits, source_y["val"])
            row = {"family": transfer.family, "transfer": transfer.name, "variant": run.variant, "method": run.method,
                   "is_control": run.factor == 0, "n_source_train": len(source_x["train"]),
                   "n_target_train": len(target_x["train"]), "n_source_val": len(source_y["val"]),
                   "src_val_ce": float(errors.mean())}
            for seed in DEV_SEEDS:
                dev = dev_risk(source_features, target_features, val_features, errors, seed)
                row.update({f"{key}_seed{seed}": value for key, value in dev.items() if key != "dev_domain_accuracies"})
            rows.append(row)
        print(f"done: {transfer.name}", flush=True)
    return pd.DataFrame(rows)


def summary(check: pd.DataFrame) -> pd.DataFrame:
    risks = check[[f"dev_risk_seed{s}" for s in DEV_SEEDS]]
    weights = check[f"dev_weight_mean_seed{SEED}"]
    flips = 0
    for method, (run, control, _, _) in ADATIME_RUNS.items():
        a = check[check["variant"] == run].set_index("transfer")
        b = check[check["variant"] == control].set_index("transfer")
        lower = pd.DataFrame({s: a[f"dev_risk_seed{s}"] < b[f"dev_risk_seed{s}"] for s in DEV_SEEDS})
        flips += int((lower.nunique(axis=1) > 1).sum())
    rows = [
        ("runs", len(check)),
        (f"domain classifier hold-out accuracy >= 0.95 (seed {SEED})", int((check[f"dev_domain_accuracy_seed{SEED}"] >= 0.95).sum())),
        (f"median domain classifier hold-out accuracy (seed {SEED})", float(check[f"dev_domain_accuracy_seed{SEED}"].median())),
        (f"mean importance weight on source val, median over runs (seed {SEED})", float(weights.median())),
        (f"mean importance weight, min over runs (seed {SEED})", float(weights.min())),
        (f"mean importance weight, max over runs (seed {SEED})", float(weights.max())),
        (f"mean importance weight outside [0.5, 2] (seed {SEED})", int(((weights < 0.5) | (weights > 2)).sum())),
        (f"largest single importance weight over all runs (seed {SEED})", float(check[f"dev_weight_max_seed{SEED}"].max())),
        ("non-finite DEV values (all seeds)", int((~np.isfinite(risks.to_numpy())).sum())),
        ("runs with a negative DEV for at least one seed", int((risks < 0).any(axis=1).sum())),
        ("runs whose DEV changes sign between the seeds", int(((risks < 0).any(axis=1) & (risks > 0).any(axis=1)).sum())),
        ("runs where max/min of |DEV| over the seeds exceeds 2",
         int((risks.abs().max(axis=1) > 2 * risks.abs().min(axis=1)).sum())),
        ("method vs own control pairs", len(TRANSFERS) * len(ADATIME_RUNS)),
        ("pairs whose DEV preference (method or control) changes with the seed", flips),
    ]
    return pd.DataFrame(rows, columns=["check", "value"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    tables = check_output_dir(TABLES)
    check = check_rows(torch.device(args.device))
    check.to_csv(tables / "T7_dev_check.csv", index=False, float_format="%.6f")
    result = summary(check)
    result.to_csv(tables / "T7_dev_check_summary.csv", index=False)
    print(result.to_string(index=False))
    for name in ("T7_dev_check.csv", "T7_dev_check_summary.csv"):
        print("Saved:", (tables / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
