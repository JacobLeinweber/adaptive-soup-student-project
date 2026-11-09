"""Score all seeds and variants and write the result tables (CSV) into results/tables/.

The baseline of a seed is the source-only model of that seed: for seed 52 the supplied predictions
in models/, otherwise results/source_only/. This script reads the target labels, only after all
training has finished and only for scoring.

Tables:
    T5_runs_all_splits.csv            transfer x seed x variant x split, all metrics
    T5_summary_target_test.csv        target test, mean and sd over the five seeds, per transfer and per group
    T5_per_class_f1_target_test.csv   target test, F1 per class of every run
    T5_confusion_target_test.csv      target test, confusion matrix of every run (one row per cell)
    T5_costs_per_run.csv              training time of every run
    T5_costs_per_variant.csv          training time summed over each variant

Example: python make_seed_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from make_results_tables import write_csv
from src.training.experiments import TRANSFERS, Transfer
from src.training.train_transfer import ROOT
from src.training.utils import SEED
from uda.data import check_output_dir, load_label_map
from uda.evaluation import METRICS, SPLITS, per_class_f1_and_confusion, score_baseline, score_run, true_labels
from uda.runs import SEEDS, source_only_dir

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
VARIANTS = ("baseline", "source_only_continued", "lambda0_target_pass", "deep_coral")
SUMMARY_METRICS = ("weighted_f1", "macro_f1", "accuracy")
FAMILIES = ("cross_dataset", "cross_gender", "cross_age")


def run_dir(variant: str, transfer: Transfer, seed: int) -> Path:
    return source_only_dir(transfer, seed) if variant == "baseline" else RESULTS / variant / f"seed_{seed}" / transfer.name


def summary_rows(scores: dict) -> list[list]:
    """Mean and sample sd over the seeds. A group is first averaged over its transfers within each seed."""
    scopes = [("transfer", t.family, t.name, [t.name]) for t in TRANSFERS]
    scopes += [("group", f, f, [t.name for t in TRANSFERS if t.family == f]) for f in FAMILIES]
    scopes += [("group", "all", "all", [t.name for t in TRANSFERS])]
    rows = []
    for kind, family, name, members in scopes:
        for variant in VARIANTS:
            row = [kind, family, name, variant, len(members), len(SEEDS)]
            for metric in SUMMARY_METRICS:
                per_seed = [np.mean([scores[member, seed, variant]["target_test"][metric] for member in members])
                            for seed in SEEDS]
                row += [f"{np.mean(per_seed):.6f}", f"{np.std(per_seed, ddof=1):.6f}"]
            rows.append(row)
    return rows


def main() -> None:
    tables = check_output_dir(TABLES)
    tables.mkdir(parents=True, exist_ok=True)
    missing = [str(run_dir(v, t, s)) for t in TRANSFERS for s in SEEDS for v in VARIANTS
               if not (run_dir(v, t, s) / "config.json").is_file()]
    if missing:
        raise SystemExit("Missing runs, no tables written:\n" + "\n".join(missing))

    scores, all_splits, per_class, confusion, costs = {}, [], [], [], []
    for transfer in TRANSFERS:
        labels = true_labels(transfer)
        label_map = load_label_map(transfer)
        names = sorted(label_map, key=label_map.get)
        for seed in SEEDS:
            for variant in VARIANTS:
                directory = run_dir(variant, transfer, seed)
                supplied = variant == "baseline" and seed == SEED
                result = score_baseline(transfer, labels) if supplied else score_run(directory, transfer, labels)
                scores[transfer.name, seed, variant] = result
                config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
                head = [transfer.family, transfer.name, seed, variant]
                all_splits += [[*head, config["device"], split, result[split]["n_windows"],
                                *(f"{result[split][m]:.6f}" for m in METRICS)] for split in SPLITS]
                with np.load(directory / "predictions" / "target_test.npz") as stored:
                    f1, matrix = per_class_f1_and_confusion(labels["target_test"], stored["y_pred"], len(names))
                for i, name in enumerate(names):
                    per_class.append([*head, i, name, int(matrix[i].sum()), f"{f1[i]:.6f}"])
                    confusion += [[*head, i, name, j, other, int(matrix[i, j])] for j, other in enumerate(names)]
                costs.append([*head, config["device"], config.get("train_seconds", "")])  # supplied runs have none

    key = ["family", "transfer", "seed", "variant"]
    write_csv(tables / "T5_runs_all_splits.csv", [*key, "device", "split", "n_windows", *METRICS], all_splits)
    write_csv(tables / "T5_summary_target_test.csv",
              ["scope", "family", "name", "variant", "n_transfers", "n_seeds",
               *(f"{m}_{stat}" for m in SUMMARY_METRICS for stat in ("mean", "sd"))], summary_rows(scores))
    write_csv(tables / "T5_per_class_f1_target_test.csv", [*key, "class_index", "class_name", "support", "f1"], per_class)
    write_csv(tables / "T5_confusion_target_test.csv",
              [*key, "true_index", "true_name", "predicted_index", "predicted_name", "count"], confusion)
    write_csv(tables / "T5_costs_per_run.csv", [*key, "device", "train_seconds"], costs)
    per_variant = []
    for variant in VARIANTS:
        seconds = [row[-1] for row in costs if row[3] == variant and row[-1] != ""]
        per_variant.append([variant, len(seconds), f"{sum(seconds):.1f}", f"{sum(seconds) / 60:.1f}"])
    write_csv(tables / "T5_costs_per_variant.csv", ["variant", "runs_with_duration", "total_seconds", "total_minutes"], per_variant)


if __name__ == "__main__":
    main()
