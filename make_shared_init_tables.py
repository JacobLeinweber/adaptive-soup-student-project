"""Stage 2 and stage 3 with the shared starting point (T11) next to the per-seed start; writes results/tables/T11_*.csv.

Built only from the tables of both starts, written by make_dsan_tables.py and make_soup_tables.py with --init per_seed
(results/tables/) and --init shared (results/shared_init/tables/); reads no raw data, no labels and no checkpoints.
"old" is the start used so far (every seed from its own source-only model), "new" the shared start (every seed from
the supplied seed-52 source-only model). Both have the same five seeds; the seed-52 runs are the same runs.

Checks before anything is written:
- the seed-52 rows of the stage-2 tables are identical in both starts (same runs, same baseline);
- in each start, the baseline and branch rows of the soup summary equal the stage-2 summary;
- both starts have the same rows (transfers, groups, variants, comparisons) in the same order.

Tables:
    T11_branches_old_vs_new.csv            baseline, methods and controls per transfer and group, old and new
    T11_method_differences_old_vs_new.csv  each method minus its own control and minus the baseline, old and new
    T11_soup_old_vs_new.csv                soups, ensembles and derived variants (as in T8), old and new
    T11_soup_differences_old_vs_new.csv    the soup comparisons 1-6 and the context rows (as in T8), old and new
    T11_seed52_soup_reproduction.csv       seed 52: soups and ensembles computed again for the shared start vs. stage 3

Example: python make_shared_init_tables.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from make_results_tables import write_csv
from make_seed_tables import SUMMARY_METRICS, TABLES, tables_dir
from src.training.utils import SEED
from uda.data import check_output_dir
from uda.evaluation import METRICS

OLD, NEW = tables_dir("per_seed"), tables_dir("shared")
SUMMARY_KEY = ["scope", "family", "name", "variant"]
SUMMARY_VALUES = [f"{m}_{stat}" for m in SUMMARY_METRICS for stat in ("mean", "sd")]


def read(folder, name: str) -> pd.DataFrame:
    return pd.read_csv(folder / name, keep_default_na=False)


def text(value) -> str:
    return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)


def side_by_side(name: str, key: list[str], values: list[str], deltas: list[str]) -> tuple[list[str], list[list]]:
    """One row per row of the table: the key, then old and new of every value column, then new - old of `deltas`."""
    old, new = read(OLD, name), read(NEW, name)
    if not old[key].equals(new[key]):
        raise SystemExit(f"{name}: the two starts do not have the same rows")
    header = [*key, *(f"{start}_{v}" for v in values for start in ("old", "new")), *(f"new_minus_old_{d}" for d in deltas)]
    rows = []
    for (_, o), (_, n) in zip(old.iterrows(), new.iterrows()):
        rows.append([*(text(o[k]) for k in key), *(text(row[v]) for v in values for row in (o, n)),
                     *(f"{n[d] - o[d]:.6f}" for d in deltas)])
    return header, rows


def check_seed52() -> int:
    """The seed-52 runs and the seed-52 baseline are the same in both starts, so their rows must be identical."""
    old, new = read(OLD, "T3_runs_all_splits.csv"), read(NEW, "T3_runs_all_splits.csv")
    o = old[old["seed"] == SEED].reset_index(drop=True)
    n = new[new["seed"] == SEED].reset_index(drop=True)
    if not o.equals(n):
        raise SystemExit("The seed-52 rows of T3_runs_all_splits.csv differ between the two starts")
    return len(o)


def check_soup_summary(folder) -> int:
    """Baseline and branch rows of the soup summary must equal the stage-2 summary of the same start."""
    stage2, soup = read(folder, "T3_summary_target_test.csv"), read(folder, "T8_summary_target_test.csv")
    merged = stage2[SUMMARY_KEY + SUMMARY_VALUES].merge(soup[SUMMARY_KEY + SUMMARY_VALUES], on=SUMMARY_KEY,
                                                        suffixes=("_t3", "_t8"))
    if len(merged) != len(stage2) or any((merged[f"{v}_t3"] != merged[f"{v}_t8"]).any() for v in SUMMARY_VALUES):
        raise SystemExit(f"{folder}: the soup summary does not reproduce the stage-2 summary")
    return len(merged)


def seed52_soup_rows() -> list[list]:
    """Seed 52: the soups and ensembles of the shared start were computed again from the same ingredients."""
    key = ["family", "transfer", "seed", "variant", "split"]
    old, new = read(OLD, "T8_metrics_all_splits.csv"), read(NEW, "T8_metrics_all_splits.csv")
    old = old[old["seed"] == SEED].set_index(key)
    new = new[new["seed"] == SEED].set_index(key)
    if not old.index.equals(new.index):
        raise SystemExit("T8_metrics_all_splits.csv: the seed-52 rows of the two starts differ in their keys")
    rows = []
    for variant in dict.fromkeys(old.index.get_level_values("variant")):
        a, b = old.xs(variant, level="variant"), new.xs(variant, level="variant")
        difference = (a[list(METRICS)] - b[list(METRICS)]).abs()
        rows.append([variant, len(a), int((difference.max(axis=1) == 0).sum()), f"{difference.to_numpy().max():.6f}"])
    return rows


def main() -> None:
    tables = check_output_dir(TABLES)
    print(f"seed 52: {check_seed52()} rows of T3_runs_all_splits.csv identical in both starts")
    for folder in (OLD, NEW):
        print(f"{folder}: {check_soup_summary(folder)} baseline and branch rows of the soup summary = stage-2 summary")

    deltas = [f"{m}_mean" for m in SUMMARY_METRICS]
    write_csv(tables / "T11_branches_old_vs_new.csv",
              *side_by_side("T3_summary_target_test.csv", SUMMARY_KEY + ["n_transfers", "n_seeds"], SUMMARY_VALUES, deltas))
    values = [f"{m}_{stat}" for m in SUMMARY_METRICS for stat in ("diff_mean", "diff_sd", "seeds_method_higher")]
    write_csv(tables / "T11_method_differences_old_vs_new.csv",
              *side_by_side("T3_method_differences_target_test.csv",
                            ["scope", "family", "name", "method", "reference", "reference_variant", "n_transfers", "n_seeds"],
                            values, [f"{m}_diff_mean" for m in SUMMARY_METRICS]))
    write_csv(tables / "T11_soup_old_vs_new.csv",
              *side_by_side("T8_summary_target_test.csv", SUMMARY_KEY + ["n_transfers", "n_seeds"],
                            SUMMARY_VALUES + ["source_test_weighted_f1_mean", "source_test_weighted_f1_sd"],
                            deltas + ["source_test_weighted_f1_mean"]))
    values = [f"{m}_{stat}" for m in SUMMARY_METRICS for stat in ("diff_mean", "diff_sd", "seeds_a_higher")]
    write_csv(tables / "T11_soup_differences_old_vs_new.csv",
              *side_by_side("T8_differences_target_test.csv",
                            ["scope", "family", "name", "comparison", "a", "b", "b_chosen_with_target_labels",
                             "n_transfers", "n_seeds", "n_pairs"],
                            values + ["weighted_f1_pairs_a_higher", "weighted_f1_pairs_a_lower"],
                            [f"{m}_diff_mean" for m in SUMMARY_METRICS]))
    write_csv(tables / "T11_seed52_soup_reproduction.csv",
              ["variant", "rows_transfer_x_split", "rows_identical", "max_abs_difference_any_metric"], seed52_soup_rows())


if __name__ == "__main__":
    main()
