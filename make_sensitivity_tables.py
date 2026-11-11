"""Label-free criteria, selection and scores for the alignment-weight sensitivity; tables into results/tables/.

Runs (seed 52 only): the paper-convention runs of run_paper_convention.py (Deep CORAL, DANN and DSAN with
source weight 1 and five multiples of the base lambda, and the shared control), and the runs with the
AdaTime weights (Deep CORAL, DANN, DSAN and their three controls).

The script works in three steps, in this order:
1. Criteria, without target labels. For every run the saved model gives its logits in eval mode for the
   source validation windows. SRC is the cross-entropy (and the usual metrics) on source validation
   (uda.criteria.source_risk). DEV is not used: see make_dev_check.py for why.
2. Selection, without target labels. Per method, the factor with the lowest mean SRC cross-entropy and the
   one with the highest mean SRC macro-F1 over the transfers, and the same per transfer.
3. Scoring. Only now are target labels read: all runs are scored on five splits, and the target values
   of the factors chosen in step 2 are looked up. Nothing is selected in this step.

Tables:
    T7_criteria.csv              one row per run: SRC on source validation
    T7_metrics_all_splits.csv    run x split (source val, source test, target train, val, test), all metrics
    T7_sensitivity_summary.csv   variant: mean and sd over the transfers of criteria and metrics
    T7_selection.csv             factor chosen by each criterion, over the mean and per transfer, with its scores
    T7_adatime_vs_paper.csv      paper convention x1 minus AdaTime weights (and method minus control), paired by transfer
    T7_commits.csv               the code commit of every group of runs

Example: python make_sensitivity_tables.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_dann
import run_deep_coral
import run_dsan
from make_results_tables import write_csv
from make_seed_tables import TABLES
from run_paper_convention import CONTROL, FACTORS, METHODS, RESULTS as PAPER_RESULTS, SOURCE_WEIGHT, base_lambda, run_folder
from src.training.experiments import TRANSFERS, Transfer
from src.training.train_transfer import ROOT
from src.training.utils import SEED
from uda.criteria import features_and_logits, source_risk
from uda.data import check_output_dir, load_label_map, load_labeled, load_source_only_model
from uda.evaluation import METRICS, SPLITS, compute_metrics, true_labels

ALL_SPLITS = ("source_val", *SPLITS)
# Runs with the AdaTime weights: (method run, its own control) and the weights of the method.
ADATIME_RUNS = {
    "deep_coral": ("deep_coral", "lambda0_target_pass", run_deep_coral.CORAL_WEIGHT, run_deep_coral.SOURCE_CE_WEIGHT),
    "dann": ("dann", "dann_lambda0_target_pass", run_dann.DOMAIN_WEIGHT, run_dann.SOURCE_CE_WEIGHT),
    "dsan": ("dsan", "dsan_lambda0_target_pass", run_dsan.LMMD_WEIGHT, run_dsan.SOURCE_CE_WEIGHT),
}
# Criteria for the selection: column, and whether lower is better.
CRITERIA = {"src_ce": ("src_val_ce", True), "src_macro_f1": ("src_val_macro_f1", False)}
SUMMARY_COLUMNS = ("src_val_ce", "src_val_macro_f1")
SUMMARY_SCORES = [("target_test", m) for m in ("weighted_f1", "macro_f1", "accuracy")] + \
                 [("source_test", "weighted_f1"), ("source_val", "weighted_f1"), ("target_train", "weighted_f1")]


@dataclass(frozen=True)
class Run:
    convention: str        # "paper" or "adatime"
    variant: str           # unique name, e.g. paper_dann_x0.5, paper_control, dann, dann_lambda0_target_pass
    method: str            # deep_coral, dann, dsan, or "control" for the shared paper control
    factor: float          # multiple of the base lambda: lambda / source weight / base lambda (0 for controls)
    alignment_weight: float
    source_weight: float

    def folder(self, transfer: Transfer) -> Path:
        if self.convention == "paper":
            return run_folder(PAPER_RESULTS, self.method, None if self.method == CONTROL else self.factor, SEED, transfer)
        return ROOT / "results" / self.variant / f"seed_{SEED}" / transfer.name


def all_runs() -> list[Run]:
    runs = [Run("paper", "paper_control", CONTROL, 0.0, 0.0, SOURCE_WEIGHT)]
    for method in METHODS:
        runs += [Run("paper", f"paper_{method}_x{f:g}", method, f, f * base_lambda(method), SOURCE_WEIGHT) for f in FACTORS]
    for method, (run, control, weight, source_weight) in ADATIME_RUNS.items():
        runs += [Run("adatime", control, method, 0.0, 0.0, source_weight), Run("adatime", run, method, 1.0, weight, source_weight)]
    return runs


def check_runs(runs: list[Run]) -> list[list]:
    """Stop unless every run is finished and every variant carries one code commit; one row per variant."""
    missing = [str(r.folder(t)) for r in runs for t in TRANSFERS if not (r.folder(t) / "config.json").is_file()]
    if missing:
        raise SystemExit("Missing runs, no tables written:\n" + "\n".join(missing))
    rows = []
    for run in runs:
        configs = [json.loads((run.folder(t) / "config.json").read_text(encoding="utf-8")) for t in TRANSFERS]
        commits = sorted({c.get("repository_commit") for c in configs})
        if len(commits) != 1 or commits[0] is None:
            raise SystemExit(f"{run.variant}: expected one code commit, found {commits}")
        for config in configs:
            if config["seed"] != SEED or not np.isclose(config.get("source_ce_weight"), run.source_weight):
                raise SystemExit(f"{run.variant}: seed or source weight in config.json does not match")
            weight = config.get("alignment_weight", config.get("coral_weight", config.get("domain_weight", config.get("lmmd_weight"))))
            if not np.isclose(weight, run.alignment_weight):
                raise SystemExit(f"{run.variant}: alignment weight {weight} in config.json, expected {run.alignment_weight}")
        rows.append([run.convention, run.variant, commits[0], len(configs)])
    paper = {row[2] for row in rows if row[0] == "paper"}
    if len(paper) != 1:
        raise SystemExit(f"The paper-convention runs carry more than one code commit: {sorted(paper)}")
    return rows


def criteria_rows(runs: list[Run], device: torch.device) -> tuple[pd.DataFrame, dict]:
    """Step 1: SRC of every run. Reads source windows and labels only, nothing of the target."""
    rows, source_val_predictions = [], {}
    for transfer in TRANSFERS:
        num_classes = len(load_label_map(transfer))
        source_x, source_y, _ = load_labeled(transfer.family, transfer.source)
        for run in runs:
            folder = run.folder(transfer)
            model = load_source_only_model(transfer, folder / "model_last.pt").to(device)
            _, val_logits = features_and_logits(model, source_x["val"], device)
            src = source_risk(val_logits, source_y["val"], num_classes)
            prediction = val_logits.argmax(1)
            source_val_predictions[transfer.name, run.variant] = prediction
            saved = folder / "predictions" / "source_val.npz"
            agreement = np.nan
            if saved.is_file():  # only the paper-convention runs saved source validation predictions
                with np.load(saved) as stored:
                    agreement = float((stored["y_pred"] == prediction).mean())
            row = {"family": transfer.family, "transfer": transfer.name, "convention": run.convention,
                   "variant": run.variant, "method": run.method, "factor": run.factor,
                   "alignment_weight": run.alignment_weight, "source_weight": run.source_weight,
                   "n_source_val": len(source_y["val"]),
                   "src_val_ce": src["ce"], "src_val_accuracy": src["accuracy"], "src_val_macro_f1": src["macro_f1"],
                   "src_val_weighted_f1": src["weighted_f1"], "source_val_prediction_agreement": agreement}
            rows.append(row)
        print(f"criteria done: {transfer.name}", flush=True)
    return pd.DataFrame(rows), source_val_predictions


def best(values: pd.Series, lower: bool):
    """Index of the best value; ties go to the first (smallest factor)."""
    return values.idxmin() if lower else values.idxmax()


def selection(criteria: pd.DataFrame) -> pd.DataFrame:
    """Step 2: the factor each criterion picks, from the criteria table alone (no target column exists here)."""
    rows = []
    for method in METHODS:
        for candidates in ("factors", "factors_and_control"):
            pool = criteria[(criteria["convention"] == "paper") & (criteria["method"] == method)]
            if candidates == "factors_and_control":
                pool = pd.concat([pool, criteria[criteria["variant"] == "paper_control"]])
            for criterion, (column, lower) in CRITERIA.items():
                mean = pool.groupby("factor")[column].mean()
                factor = best(mean, lower)
                rows.append({"method": method, "candidates": candidates, "criterion": criterion, "scope": "mean_over_transfers",
                             "transfer": "", "chosen_factor": factor, "criterion_value": mean[factor]})
                for transfer, part in pool.groupby("transfer", sort=False):
                    values = part.set_index("factor")[column].sort_index()
                    factor = best(values, lower)
                    rows.append({"method": method, "candidates": candidates, "criterion": criterion, "scope": "per_transfer",
                                 "transfer": transfer, "chosen_factor": factor, "criterion_value": values[factor]})
    return pd.DataFrame(rows)


def variant_of(method: str, factor: float) -> str:
    return "paper_control" if factor == 0 else f"paper_{method}_x{factor:g}"


def score_rows(runs: list[Run], source_val_predictions: dict) -> pd.DataFrame:
    """Step 3: metrics of every run on the five splits. This is where target labels are read."""
    rows = []
    for transfer in TRANSFERS:
        labels = true_labels(transfer)
        _, source_y, _ = load_labeled(transfer.family, transfer.source)
        labels["source_val"] = np.asarray(source_y["val"], dtype=np.int64)
        num_classes = len(load_label_map(transfer))
        for run in runs:
            for split in ALL_SPLITS:
                saved = run.folder(transfer) / "predictions" / f"{split}.npz"
                if split == "source_val" and not saved.is_file():
                    prediction = source_val_predictions[transfer.name, run.variant]  # AdaTime runs: predicted again in step 1
                else:
                    with np.load(saved) as stored:
                        prediction = stored["y_pred"]
                result = compute_metrics(labels[split], prediction, num_classes)
                rows.append({"family": transfer.family, "transfer": transfer.name, "convention": run.convention,
                             "variant": run.variant, "method": run.method, "factor": run.factor,
                             "alignment_weight": run.alignment_weight, "source_weight": run.source_weight,
                             "split": split, "n_windows": result["n_windows"], **{m: result[m] for m in METRICS}})
    return pd.DataFrame(rows)


def check_against_earlier_tables(scores: pd.DataFrame) -> int:
    """The AdaTime runs were scored before (T3); their four splits must give the same numbers."""
    earlier = pd.read_csv(TABLES / "T3_runs_all_splits.csv")
    earlier = earlier[earlier["seed"] == SEED].set_index(["transfer", "variant", "split"])
    ours = scores[scores["convention"] == "adatime"].set_index(["transfer", "variant", "split"])
    ours = ours[ours.index.get_level_values("split") != "source_val"]
    joined = ours[list(METRICS)].merge(earlier[list(METRICS)], left_index=True, right_index=True, how="left",
                                       suffixes=("", "_t3"))
    for metric in METRICS:
        if not np.allclose(joined[metric], joined[f"{metric}_t3"], atol=5e-7, rtol=0):
            raise SystemExit(f"AdaTime runs: {metric} differs from T3_runs_all_splits.csv")
    return len(joined)


def wide_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per run, one column per split and metric, e.g. target_test_weighted_f1."""
    table = scores.pivot_table(index=["transfer", "variant"], columns="split", values=list(METRICS))
    table.columns = [f"{split}_{metric}" for metric, split in table.columns]
    return table


def summary_rows(criteria: pd.DataFrame, wide: pd.DataFrame) -> list[list]:
    """Mean and sample sd over the transfers of every variant."""
    joined = criteria.set_index(["transfer", "variant"]).join(wide)
    columns = [*SUMMARY_COLUMNS, *(f"{split}_{metric}" for split, metric in SUMMARY_SCORES)]
    rows = []
    for variant, part in joined.groupby("variant", sort=False):
        first = part.iloc[0]
        row = [first["convention"], variant, first["method"], first["factor"], first["alignment_weight"],
               first["source_weight"], len(part)]
        for column in columns:
            row += [f"{part[column].mean():.6f}", f"{part[column].std(ddof=1):.6f}"]
        rows.append(row)
    header = ["convention", "variant", "method", "factor", "alignment_weight", "source_weight", "n_transfers",
              *(f"{column}_{stat}" for column in columns for stat in ("mean", "sd"))]
    return [header, *rows]


def selection_with_scores(selected: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Target and source scores of the chosen factors (evaluation only; the choice is already fixed)."""
    out = []
    target = wide["target_test_weighted_f1"]
    for _, row in selected.iterrows():
        row = row.to_dict()
        if row["scope"] == "per_transfer":
            row["target_test_weighted_f1"] = target[row["transfer"], variant_of(row["method"], row["chosen_factor"])]
        else:
            row["target_test_weighted_f1"] = target.xs(variant_of(row["method"], row["chosen_factor"]), level="variant").mean()
        out.append(row)
    table = pd.DataFrame(out)
    # One more row per method, criterion and candidate set: the mean over the transfers when every transfer
    # uses its own chosen factor.
    per_transfer = table[table["scope"] == "per_transfer"]
    extra = per_transfer.groupby(["method", "candidates", "criterion"], sort=False).agg(
        target_test_weighted_f1=("target_test_weighted_f1", "mean"), n=("transfer", "size")).reset_index()
    extra["scope"], extra["transfer"] = "per_transfer_choice_mean", ""
    for (method, candidates, criterion), part in per_transfer.groupby(["method", "candidates", "criterion"], sort=False):
        counts = part["chosen_factor"].value_counts().sort_index()
        extra.loc[(extra["method"] == method) & (extra["candidates"] == candidates) & (extra["criterion"] == criterion),
                  "chosen_factor_counts"] = "; ".join(f"x{f:g}: {n}" for f, n in counts.items())
    return pd.concat([table, extra.drop(columns="n")], ignore_index=True)


def difference_rows(wide: pd.DataFrame) -> list[list]:
    """Paired by transfer: a - b, mean and sample sd over the transfers, and the number of transfers with a > b."""
    pairs = []
    for method, (run, control, _, _) in ADATIME_RUNS.items():
        pairs.append(("paper_x1_minus_adatime", method, f"paper_{method}_x1", run))
    for method, (_, control, _, _) in ADATIME_RUNS.items():
        pairs.append(("paper_control_minus_adatime_control", method, "paper_control", control))
    for method, (run, control, _, _) in ADATIME_RUNS.items():
        pairs.append(("method_minus_control_paper_x1", method, f"paper_{method}_x1", "paper_control"))
        pairs.append(("method_minus_control_adatime", method, run, control))
    rows = []
    for comparison, method, a, b in pairs:
        for column in wide.columns:
            domain, part, metric = column.split("_", 2)  # e.g. target_test_weighted_f1
            split = f"{domain}_{part}"
            difference = (wide[column].xs(a, level="variant") - wide[column].xs(b, level="variant")) * 100
            rows.append([comparison, method, a, b, split, metric, len(difference), f"{difference.mean():.6f}",
                         f"{difference.std(ddof=1):.6f}", int((difference > 0).sum())])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    tables = check_output_dir(TABLES)
    runs = all_runs()
    commits = check_runs(runs)

    criteria, source_val_predictions = criteria_rows(runs, device)  # step 1, no target labels
    selected = selection(criteria)                                    # step 2, no target labels
    scores = score_rows(runs, source_val_predictions)                 # step 3, target labels for scoring only
    print(f"AdaTime runs agree with T3_runs_all_splits.csv: {check_against_earlier_tables(scores)} rows")
    wide = wide_scores(scores)

    criteria.to_csv(tables / "T7_criteria.csv", index=False, float_format="%.6f")
    scores.to_csv(tables / "T7_metrics_all_splits.csv", index=False, float_format="%.6f")
    header, *rows = summary_rows(criteria, wide)
    write_csv(tables / "T7_sensitivity_summary.csv", header, rows)
    selection_with_scores(selected, wide).to_csv(tables / "T7_selection.csv", index=False, float_format="%.6f")
    write_csv(tables / "T7_adatime_vs_paper.csv",
              ["comparison", "method", "variant_a", "variant_b", "split", "metric", "n_transfers",
               "diff_mean_points", "diff_sd_points", "transfers_a_higher"], difference_rows(wide))
    write_csv(tables / "T7_commits.csv", ["convention", "variant", "repository_commit", "runs"], commits)
    for name in ("T7_criteria.csv", "T7_metrics_all_splits.csv", "T7_selection.csv"):
        print("Saved:", (tables / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
