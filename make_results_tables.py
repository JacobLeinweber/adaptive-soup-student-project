"""Score the baseline and our runs on all 12 transfers and write the result tables (CSV).

Baseline: from the supplied predictions in models/<family>/<transfer>/predictions/*.npz.
Runs:     from results/source_only_continued/<transfer>/ (lambda = 0),
          results/lambda0_target_pass/<transfer>/ (lambda = 0 with target forward pass) and
          results/deep_coral/<transfer>/, if present; writes metrics.json next to each run.

Example: python make_results_tables.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.training.experiments import TRANSFERS
from src.training.train_transfer import ROOT
from uda.data import check_output_dir, checkpoint_dir
from uda.evaluation import METRICS, SPLITS, score_baseline, score_run, true_labels

RUNS = {"lambda0": ROOT / "results" / "source_only_continued",
        "lambda0_target_pass": ROOT / "results" / "lambda0_target_pass",
        "deep_coral": ROOT / "results" / "deep_coral"}
TABLES = ROOT / "results" / "tables"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print("Saved:", path.relative_to(ROOT))


def write_comparison(path: Path, methods: list[str], scores: dict) -> None:
    """One row per transfer x split with the metrics of every method side by side (empty if a run is missing)."""
    rows = []
    for t in TRANSFERS:
        for split in SPLITS:
            row = [t.family, t.name, split, scores[t.name, "baseline"][split]["n_windows"]]
            for method in methods:
                result = scores.get((t.name, method))
                row += [f"{result[split][m]:.6f}" if result else "" for m in METRICS]
            rows.append(row)
    write_csv(path, ["family", "transfer", "split", "n_windows", *(f"{method}_{m}" for method in methods for m in METRICS)], rows)


def main() -> None:
    tables = check_output_dir(TABLES)
    tables.mkdir(parents=True, exist_ok=True)
    scores = {}  # (transfer name, method) -> split -> metrics
    max_delta = 0.0
    for transfer in TRANSFERS:
        labels = true_labels(transfer)
        scores[transfer.name, "baseline"] = score_baseline(transfer, labels)
        # Sanity check against the supervisor's own metrics (macro-F1 there has no fixed class list).
        reference = json.loads((checkpoint_dir(transfer) / "metrics.json").read_text())["final_epoch"]
        for split in SPLITS:
            for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                delta = abs(scores[transfer.name, "baseline"][split][metric] - reference[split][metric])
                max_delta = max(max_delta, delta)
        for method, root in RUNS.items():
            run_dir = root / transfer.name
            if (run_dir / "config.json").is_file():
                result = score_run(run_dir, transfer, labels)
                scores[transfer.name, method] = result
                (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Baseline vs. supplied metrics.json: max. absolute difference {max_delta:.2e}")

    write_csv(
        tables / "baseline_all_transfers.csv",
        ["family", "transfer", "split", "n_windows", *METRICS],
        [[t.family, t.name, split, scores[t.name, "baseline"][split]["n_windows"],
          *(f"{scores[t.name, 'baseline'][split][m]:.6f}" for m in METRICS)]
         for t in TRANSFERS for split in SPLITS],
    )

    missing = [f"{method}/{t.name}" for t in TRANSFERS for method in RUNS if (t.name, method) not in scores]
    if len(missing) == len(TRANSFERS) * len(RUNS):
        return
    write_comparison(tables / "T1_baseline_lambda0_deep_coral.csv", ["baseline", "lambda0", "deep_coral"], scores)
    write_comparison(tables / "T1_baseline_lambda0_targetpass_deep_coral.csv", ["baseline", *RUNS], scores)
    if missing:
        print("Missing runs (empty cells):", ", ".join(missing))


if __name__ == "__main__":
    main()
