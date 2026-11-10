"""Score all three methods and their controls next to the baseline; write the tables (CSV) into results/tables/.

The baseline of a seed is the source-only model of that seed: for seed 52 the supplied predictions
in models/, otherwise results/source_only/. This script reads the target labels, only after all
training has finished and only for scoring.

Tables:
    T3_runs_all_splits.csv                    transfer x seed x variant x split, all metrics
    T3_summary_target_test.csv                target test, mean and sd over the five seeds, per transfer and per group
    T3_method_differences_target_test.csv     target test, each method minus its own control and minus the baseline
    T3_per_class_f1_target_test.csv           target test, F1 per class of every run
    T3_confusion_target_test.csv              target test, confusion matrix of every run (one row per cell)
    T3_costs_per_run.csv                      training time of every run
    T3_costs_per_variant.csv                  training time summed over each variant
    T3_commits_per_variant.csv                the code commit of every variant

Example: python make_dsan_tables.py
"""
from __future__ import annotations

import json

import numpy as np

from make_dann_tables import per_seed, scopes
from make_results_tables import write_csv
from make_seed_tables import SUMMARY_METRICS, TABLES, run_dir
from src.training.experiments import TRANSFERS
from src.training.utils import SEED
from uda.data import check_output_dir, load_label_map
from uda.evaluation import METRICS, SPLITS, per_class_f1_and_confusion, score_baseline, score_run, true_labels
from uda.runs import SEEDS

# Every method with its own control: weight of the alignment loss 0, target windows still pass through the model.
CONTROLS = {"deep_coral": "lambda0_target_pass", "dann": "dann_lambda0_target_pass", "dsan": "dsan_lambda0_target_pass"}
VARIANTS = ("baseline", *(variant for pair in CONTROLS.items() for variant in pair))


def summary_rows(scores: dict) -> list[list]:
    """Mean and sample sd over the seeds."""
    rows = []
    for kind, family, name, members in scopes():
        for variant in VARIANTS:
            row = [kind, family, name, variant, len(members), len(SEEDS)]
            for metric in SUMMARY_METRICS:
                values = per_seed(scores, members, variant, metric)
                row += [f"{values.mean():.6f}", f"{values.std(ddof=1):.6f}"]
            rows.append(row)
    return rows


def difference_rows(scores: dict) -> list[list]:
    """Each method minus its own control and minus the baseline, paired by seed.

    Mean and sample sd of the difference over the seeds, and in how many seeds the method is higher.
    """
    rows = []
    for kind, family, name, members in scopes():
        for method, control in CONTROLS.items():
            for reference, reference_variant in (("own_control", control), ("baseline", "baseline")):
                row = [kind, family, name, method, reference, reference_variant, len(members), len(SEEDS)]
                for metric in SUMMARY_METRICS:
                    difference = (per_seed(scores, members, method, metric)
                                  - per_seed(scores, members, reference_variant, metric))
                    row += [f"{difference.mean():.6f}", f"{difference.std(ddof=1):.6f}", int((difference > 0).sum())]
                rows.append(row)
    return rows


def commit_rows() -> list[list]:
    """Stop unless all runs of a variant carry one code commit; return one row per variant.

    The supplied seed-52 baselines were trained outside this repository and carry no commit.
    """
    rows = []
    for variant in VARIANTS:
        commits = [json.loads((run_dir(variant, t, s) / "config.json").read_text(encoding="utf-8")).get("repository_commit")
                   for t in TRANSFERS for s in SEEDS]
        found = sorted({commit for commit in commits if commit})
        if len(found) != 1:
            raise SystemExit(f"{variant}: expected one code commit, found {found}")
        rows.append([variant, found[0], len(commits) - commits.count(None), commits.count(None)])
        print(f"{variant}: commit {found[0]}, {rows[-1][2]} runs, {rows[-1][3]} supplied runs without commit")
    return rows


def main() -> None:
    tables = check_output_dir(TABLES)
    tables.mkdir(parents=True, exist_ok=True)
    missing = [str(run_dir(v, t, s)) for t in TRANSFERS for s in SEEDS for v in VARIANTS
               if not (run_dir(v, t, s) / "config.json").is_file()]
    if missing:
        raise SystemExit("Missing runs, no tables written:\n" + "\n".join(missing))
    commits = commit_rows()

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
    write_csv(tables / "T3_runs_all_splits.csv", [*key, "device", "split", "n_windows", *METRICS], all_splits)
    write_csv(tables / "T3_summary_target_test.csv",
              ["scope", "family", "name", "variant", "n_transfers", "n_seeds",
               *(f"{m}_{stat}" for m in SUMMARY_METRICS for stat in ("mean", "sd"))], summary_rows(scores))
    write_csv(tables / "T3_method_differences_target_test.csv",
              ["scope", "family", "name", "method", "reference", "reference_variant", "n_transfers", "n_seeds",
               *(f"{m}_{stat}" for m in SUMMARY_METRICS for stat in ("diff_mean", "diff_sd", "seeds_method_higher"))],
              difference_rows(scores))
    write_csv(tables / "T3_per_class_f1_target_test.csv", [*key, "class_index", "class_name", "support", "f1"], per_class)
    write_csv(tables / "T3_confusion_target_test.csv",
              [*key, "true_index", "true_name", "predicted_index", "predicted_name", "count"], confusion)
    write_csv(tables / "T3_costs_per_run.csv", [*key, "device", "train_seconds"], costs)
    per_variant = []
    for variant in VARIANTS:
        seconds = [row[-1] for row in costs if row[3] == variant and row[-1] != ""]
        per_variant.append([variant, len(seconds), f"{sum(seconds):.1f}", f"{sum(seconds) / 60:.1f}"])
    write_csv(tables / "T3_costs_per_variant.csv", ["variant", "runs_with_duration", "total_seconds", "total_minutes"], per_variant)
    write_csv(tables / "T3_commits_per_variant.csv", ["variant", "repository_commit", "runs_with_commit", "supplied_runs_without_commit"], commits)


if __name__ == "__main__":
    main()
