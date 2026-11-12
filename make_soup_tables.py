"""Score the uniform soups, the output ensembles and their ingredients; write the T8 tables (CSV) into results/tables/.

The script works in three steps, in this order:
1. Criterion, without target labels. SRC on source validation (cross-entropy and macro-F1, uda.criteria.source_risk)
   for every ingredient, soup and ensemble, from the source-val outputs that run_soup.py saved.
2. Selection, without target labels. Per transfer, seed and soup, the ingredient with the lowest SRC cross-entropy
   and the one with the highest SRC macro-F1 ("best branch by SRC").
3. Scoring. Only now are target labels read. Every output is scored on source val and the four splits, and the
   derived comparisons are formed: mean of the three branches, the branches chosen in step 2, and as a reference
   that uses target labels, the branch with the highest weighted F1 on target test (oracle).

The baseline of a seed is the source-only model of that seed: for seed 52 the supplied predictions in models/,
otherwise results/source_only/. Differences are paired by transfer and seed; a group value is first averaged over
the transfers of the group within each seed, then mean and sample SD are taken over the five seeds.

Tables (names chosen so that make_figures.py does not merge them into the stage-2 tables):
    T8_criteria.csv                      SRC on source val of every ingredient, soup and ensemble
    T8_selection.csv                     branch chosen by each SRC variant and by the oracle, per transfer, seed and soup
    T8_metrics_all_splits.csv            output x split (source val, source test, target train, val, test), all metrics
    T8_summary_target_test.csv           mean and SD over the seeds per transfer and group, also derived variants
    T8_differences_target_test.csv      the paired comparisons 1-6 of the task (and two for context)
    T8_per_class_f1_target_test.csv      F1 per class of every soup and ensemble on target test
    T8_confusion_matrices_target_test.csv  confusion matrix of every soup and ensemble on target test (one row per cell)
    T8_costs.csv                         seconds for loading, averaging, BatchNorm re-estimation, prediction per output
    T8_costs_summary.csv                 the same summed per variant, next to the training time of the ingredients
    T8_commits.csv                       code commit of every kind of output, and of the ingredient runs

Example: python make_soup_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from make_dann_tables import scopes
from make_results_tables import write_csv
from make_seed_tables import SUMMARY_METRICS, TABLES, run_dir
from run_soup import RESULTS, ensemble_dir, soup_dir, source_val_dir
from src.training.experiments import TRANSFERS, Transfer
from src.training.utils import SEED
from uda.criteria import source_risk
from uda.data import check_output_dir, load_label_map, load_labeled
from uda.evaluation import METRICS, SPLITS, compute_metrics, per_class_f1_and_confusion, score_baseline, score_run, true_labels
from uda.runs import SEEDS
from uda.soup import BN_VARIANTS, ENSEMBLES, SOUPS

ALL_SPLITS = ("source_val", *SPLITS)
GROUPS = {ENSEMBLES[soup]: soup for soup in SOUPS}  # "methods" -> "soup_methods", "controls" -> "soup_controls"
CRITERIA = {"src_ce": ("ce", True), "src_macro_f1": ("macro_f1", False)}  # column of source_risk, lower is better
SOUP_VARIANTS = [f"{soup}_{bn}" for soup in SOUPS for bn in BN_VARIANTS]
ENSEMBLE_VARIANTS = [f"ensemble_{group}" for group in GROUPS]
BRANCHES = [branch for soup in SOUPS for branch in SOUPS[soup]]
OUTPUTS = ["baseline", *BRANCHES, *SOUP_VARIANTS, *ENSEMBLE_VARIANTS]
DERIVED = [f"{kind}_{group}" for group in GROUPS for kind in ("branch_mean", "best_src_ce", "best_src_macro_f1", "oracle")]


def output_dir(variant: str, transfer: Transfer, seed: int) -> Path:
    if variant in SOUP_VARIANTS:
        soup, bn = variant.rsplit("_bn_", 1)
        return soup_dir(RESULTS, soup, f"bn_{bn}", seed, transfer)
    if variant in ENSEMBLE_VARIANTS:
        return ensemble_dir(RESULTS, GROUPS[variant.removeprefix("ensemble_")], seed, transfer)
    return run_dir(variant, transfer, seed)


def prediction_file(variant: str, transfer: Transfer, seed: int, split: str) -> Path:
    """The branches saved no source-val predictions; run_soup.py computed them from their checkpoints (D5)."""
    if variant in BRANCHES and split == "source_val":
        return source_val_dir(RESULTS, variant, seed, transfer) / "source_val.npz"
    return output_dir(variant, transfer, seed) / "predictions" / f"{split}.npz"


def config(folder: Path) -> dict:
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))


def check_outputs() -> list[list]:
    """Stop unless every output exists, has the start checkpoint of its seed and one code commit per kind."""
    missing = [str(prediction_file(v, t, s, "source_val")) for t in TRANSFERS for s in SEEDS for v in [*BRANCHES, *SOUP_VARIANTS, *ENSEMBLE_VARIANTS]
               if not prediction_file(v, t, s, "source_val").is_file()]
    if missing:
        raise SystemExit("Missing outputs, no tables written:\n" + "\n".join(missing[:20]))
    kinds = {"soup": SOUP_VARIANTS, "ensemble": ENSEMBLE_VARIANTS}
    rows = []
    for kind, variants in kinds.items():
        configs = [config(output_dir(v, t, s)) for v in variants for t in TRANSFERS for s in SEEDS]
        commits = sorted({c["repository_commit"] for c in configs})
        if len(commits) != 1:
            raise SystemExit(f"{kind}: expected one code commit, found {commits}")
        for c in configs:
            if c["target_labels_used"] or len({i["init_checkpoint_sha256"] for i in c["ingredients"]} | {c["init_checkpoint_sha256"]}) != 1:
                raise SystemExit(f"{kind}: start checkpoints differ or target labels flagged in {c['transfer']} seed {c['seed']}")
        rows.append([kind, commits[0], len(configs)])
    source_val = [config(source_val_dir(RESULTS, b, s, t)) for b in BRANCHES for t in TRANSFERS for s in SEEDS]
    commits = sorted({c["repository_commit"] for c in source_val})
    if len(commits) != 1:
        raise SystemExit(f"source-val predictions: expected one code commit, found {commits}")
    rows.append(["source_val_predictions", commits[0], len(source_val)])
    for branch in BRANCHES:
        found = sorted({config(run_dir(branch, t, s))["repository_commit"] for t in TRANSFERS for s in SEEDS})
        if len(found) != 1:
            raise SystemExit(f"{branch}: expected one code commit, found {found}")
        rows.append([f"ingredient {branch}", found[0], len(TRANSFERS) * len(SEEDS)])
    for row in rows:
        print(f"{row[0]}: commit {row[1]}, {row[2]} outputs")
    return rows


def source_val_outputs(variant: str, transfer: Transfer, seed: int) -> np.ndarray:
    """Logits (soups, branches) or log-probabilities (ensembles) on source val; both give SRC through a softmax."""
    with np.load(prediction_file(variant, transfer, seed, "source_val")) as stored:
        if "logits" in stored:
            return stored["logits"].astype(np.float64)
        return np.log(np.clip(stored["prob"].astype(np.float64), 1e-12, None))


def criteria_and_selection() -> tuple[list[list], dict]:
    """Steps 1 and 2: reads source windows' labels only, nothing of the target."""
    criteria, chosen = [], {}
    for transfer in TRANSFERS:
        num_classes = len(load_label_map(transfer))
        _, source_y, _ = load_labeled(transfer.family, transfer.source)
        for seed in SEEDS:
            src = {}
            for variant in [*BRANCHES, *SOUP_VARIANTS, *ENSEMBLE_VARIANTS]:
                src[variant] = source_risk(source_val_outputs(variant, transfer, seed), source_y["val"], num_classes)
                criteria.append([transfer.family, transfer.name, seed, variant, len(source_y["val"]),
                                 *(f"{src[variant][key]:.6f}" for key in ("ce", "macro_f1", "accuracy", "weighted_f1"))])
            for group, soup in GROUPS.items():
                for criterion, (key, lower) in CRITERIA.items():
                    values = [src[branch][key] for branch in SOUPS[soup]]
                    pick = SOUPS[soup][int(np.argmin(values) if lower else np.argmax(values))]  # ties: first branch
                    chosen[transfer.name, seed, f"best_{criterion}_{group}"] = pick
    return criteria, chosen


def score(variant: str, transfer: Transfer, seed: int, labels: dict, num_classes: int) -> dict:
    """Metrics of one output from its label-free predictions; the baseline on the four splits only."""
    if variant == "baseline":
        return score_baseline(transfer, labels) if seed == SEED else score_run(run_dir(variant, transfer, seed), transfer, labels)
    result = {}
    for split in ALL_SPLITS:
        with np.load(prediction_file(variant, transfer, seed, split)) as stored:
            result[split] = compute_metrics(labels[split], stored["y_pred"], num_classes)
    return result


def derived_scores(scores: dict, chosen: dict, transfer: Transfer, seed: int) -> None:
    """Mean of the three branches, the branches chosen without target labels and the oracle branch."""
    for group, soup in GROUPS.items():
        branches = SOUPS[soup]
        scores[transfer.name, seed, f"branch_mean_{group}"] = {
            split: {m: float(np.mean([scores[transfer.name, seed, b][split][m] for b in branches])) for m in METRICS}
            for split in ALL_SPLITS}
        for criterion in CRITERIA:
            key = f"best_{criterion}_{group}"
            scores[transfer.name, seed, key] = scores[transfer.name, seed, chosen[transfer.name, seed, key]]
        target = [scores[transfer.name, seed, b]["target_test"]["weighted_f1"] for b in branches]
        chosen[transfer.name, seed, f"oracle_{group}"] = branches[int(np.argmax(target))]  # uses target labels
        scores[transfer.name, seed, f"oracle_{group}"] = scores[transfer.name, seed, chosen[transfer.name, seed, f"oracle_{group}"]]


def per_seed(scores: dict, members: list[str], variant: str, metric: str, split: str = "target_test") -> np.ndarray:
    return np.array([np.mean([scores[t, s, variant][split][metric] for t in members]) for s in SEEDS])


def comparisons() -> list[tuple[str, str, str, bool]]:
    """(id, a, b, uses target labels to choose b): the comparisons of the task, numbered as there."""
    rows = []
    for group, soup in GROUPS.items():
        for bn in BN_VARIANTS:
            s = f"{soup}_{bn}"
            rows += [("1_soup_minus_branch_mean", s, f"branch_mean_{group}", False),
                     ("2a_soup_minus_best_src_ce", s, f"best_src_ce_{group}", False),
                     ("2b_soup_minus_best_src_macro_f1", s, f"best_src_macro_f1_{group}", False),
                     ("2c_soup_minus_oracle_best_branch", s, f"oracle_{group}", True),
                     ("3_soup_minus_baseline", s, "baseline", False),
                     ("5_soup_minus_ensemble", s, f"ensemble_{group}", False)]
        rows.append(("6_bn_target_minus_bn_avg", f"{soup}_bn_target", f"{soup}_bn_avg", False))
        rows += [("context_ensemble_minus_branch_mean", f"ensemble_{group}", f"branch_mean_{group}", False),
                 ("context_ensemble_minus_baseline", f"ensemble_{group}", "baseline", False)]
    for bn in BN_VARIANTS:
        rows.append(("4_soup_methods_minus_soup_controls", f"soup_methods_{bn}", f"soup_controls_{bn}", False))
    rows.append(("context_branch_mean_methods_minus_controls", "branch_mean_methods", "branch_mean_controls", False))
    return rows


def summary_rows(scores: dict) -> list[list]:
    rows = []
    for kind, family, name, members in scopes():
        for variant in [*OUTPUTS, *DERIVED]:
            row = [kind, family, name, variant, len(members), len(SEEDS)]
            for metric in SUMMARY_METRICS:
                values = per_seed(scores, members, variant, metric)
                row += [f"{values.mean():.6f}", f"{values.std(ddof=1):.6f}"]
            values = per_seed(scores, members, variant, "weighted_f1", "source_test")
            rows.append(row + [f"{values.mean():.6f}", f"{values.std(ddof=1):.6f}"])
    return rows


def difference_rows(scores: dict) -> list[list]:
    rows = []
    for kind, family, name, members in scopes():
        for comparison, a, b, oracle in comparisons():
            row = [kind, family, name, comparison, a, b, oracle, len(members), len(SEEDS)]
            for metric in SUMMARY_METRICS:
                difference = per_seed(scores, members, a, metric) - per_seed(scores, members, b, metric)
                row += [f"{difference.mean():.6f}", f"{difference.std(ddof=1):.6f}", int((difference > 0).sum())]
            pairs = np.array([scores[t, s, a]["target_test"]["weighted_f1"] - scores[t, s, b]["target_test"]["weighted_f1"]
                              for t in members for s in SEEDS])
            rows.append(row + [int((pairs > 0).sum()), int((pairs < 0).sum()), len(pairs)])
    return rows


def main() -> None:
    tables = check_output_dir(TABLES)
    commits = check_outputs()

    # Steps 1 and 2: criterion and selection without target labels.
    criteria, chosen = criteria_and_selection()

    # Step 3: scoring; target labels are read from here on.
    scores, all_splits, per_class, confusion = {}, [], [], []
    for transfer in TRANSFERS:
        labels = true_labels(transfer)
        _, source_y, _ = load_labeled(transfer.family, transfer.source)
        labels["source_val"] = np.asarray(source_y["val"], dtype=np.int64)
        label_map = load_label_map(transfer)
        names = sorted(label_map, key=label_map.get)
        for seed in SEEDS:
            for variant in OUTPUTS:
                result = score(variant, transfer, seed, labels, len(names))
                scores[transfer.name, seed, variant] = result
                head = [transfer.family, transfer.name, seed, variant]
                all_splits += [[*head, split, result[split]["n_windows"], *(f"{result[split][m]:.6f}" for m in METRICS)]
                               for split in result]
                if variant in SOUP_VARIANTS or variant in ENSEMBLE_VARIANTS:
                    with np.load(prediction_file(variant, transfer, seed, "target_test")) as stored:
                        f1, matrix = per_class_f1_and_confusion(labels["target_test"], stored["y_pred"], len(names))
                    for i, cls in enumerate(names):
                        per_class.append([*head, i, cls, int(matrix[i].sum()), f"{f1[i]:.6f}"])
                        confusion += [[*head, i, cls, j, other, int(matrix[i, j])] for j, other in enumerate(names)]
            derived_scores(scores, chosen, transfer, seed)

    selection = []
    for transfer in TRANSFERS:
        for seed in SEEDS:
            for group in GROUPS:
                for kind, oracle in (("best_src_ce", False), ("best_src_macro_f1", False), ("oracle", True)):
                    key = f"{kind}_{group}"
                    pick = chosen[transfer.name, seed, key]
                    selection.append([transfer.family, transfer.name, seed, group, kind, oracle, pick,
                                      f"{scores[transfer.name, seed, pick]['target_test']['weighted_f1']:.6f}"])

    costs, totals = [], {}
    for transfer in TRANSFERS:
        for seed in SEEDS:
            for variant in [*SOUP_VARIANTS, *ENSEMBLE_VARIANTS]:
                c = config(output_dir(variant, transfer, seed))
                seconds = c["seconds"]
                train = sum(i["train_seconds"] for i in c["ingredients"])
                own = sum(v for k, v in seconds.items() if k != "load_ingredients_and_logits")
                costs.append([transfer.family, transfer.name, seed, variant, c["device"],
                              seconds.get("load_ingredients_and_logits", ""), seconds.get("average", ""),
                              seconds.get("bn_recompute", ""), seconds.get("predict", ""), seconds.get("combine", ""),
                              f"{own:.3f}", f"{train:.1f}"])
                total = totals.setdefault(variant, [0, 0.0, 0.0, 0.0, 0.0])
                total[0] += 1
                total[1] += seconds.get("average", 0.0)
                total[2] += seconds.get("bn_recompute", 0.0)
                total[3] += own
                total[4] += train
    summary_costs = [[v, n, f"{a:.2f}", f"{b:.1f}", f"{o:.1f}", f"{o / n:.2f}", f"{t / 60:.1f}", f"{o / t * 100:.3f}"]
                     for v, (n, a, b, o, t) in totals.items()]

    key = ["family", "transfer", "seed", "variant"]
    write_csv(tables / "T8_criteria.csv", [*key, "n_source_val", "src_val_ce", "src_val_macro_f1", "src_val_accuracy",
                                           "src_val_weighted_f1"], criteria)
    write_csv(tables / "T8_selection.csv", ["family", "transfer", "seed", "group", "rule", "uses_target_labels",
                                            "chosen_branch", "target_test_weighted_f1"], selection)
    write_csv(tables / "T8_metrics_all_splits.csv", [*key, "split", "n_windows", *METRICS], all_splits)
    write_csv(tables / "T8_summary_target_test.csv",
              ["scope", "family", "name", "variant", "n_transfers", "n_seeds",
               *(f"{m}_{stat}" for m in SUMMARY_METRICS for stat in ("mean", "sd")),
               "source_test_weighted_f1_mean", "source_test_weighted_f1_sd"], summary_rows(scores))
    write_csv(tables / "T8_differences_target_test.csv",
              ["scope", "family", "name", "comparison", "a", "b", "b_chosen_with_target_labels", "n_transfers", "n_seeds",
               *(f"{m}_{stat}" for m in SUMMARY_METRICS for stat in ("diff_mean", "diff_sd", "seeds_a_higher")),
               "weighted_f1_pairs_a_higher", "weighted_f1_pairs_a_lower", "n_pairs"], difference_rows(scores))
    write_csv(tables / "T8_per_class_f1_target_test.csv", [*key, "class_index", "class_name", "support", "f1"], per_class)
    write_csv(tables / "T8_confusion_matrices_target_test.csv",
              [*key, "true_index", "true_name", "predicted_index", "predicted_name", "count"], confusion)
    write_csv(tables / "T8_costs.csv", [*key, "device", "seconds_load_ingredients_and_logits", "seconds_average",
                                        "seconds_bn_recompute", "seconds_predict", "seconds_combine",
                                        "seconds_own_steps", "train_seconds_of_ingredients"], costs)
    write_csv(tables / "T8_costs_summary.csv", ["variant", "outputs", "total_seconds_average", "total_seconds_bn_recompute",
                                                "total_seconds_own_steps", "mean_seconds_own_steps",
                                                "total_minutes_training_of_ingredients", "own_steps_percent_of_training"],
              summary_costs)
    write_csv(tables / "T8_commits.csv", ["outputs", "repository_commit", "count"], commits)


if __name__ == "__main__":
    main()
