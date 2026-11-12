# Runs log

One entry per block of runs. "Code commit" is the commit recorded in the `config.json` of every run of the block (`repository_commit`); "results commit" is the commit that added the run folders and tables to the repository. All runs: 30 epochs, last epoch, Apple M1 with MPS, PyTorch 2.14.1. The prefixes of the table names (`T1`, `T5`, `T2`, `T3`, `T7`, `T8`) are the internal numbers of the work packages, in the order in which they were done.

## Overview

| Block | Date | Code commit | Results commit | Runs | Location | Main table |
|---|---|---|---|---:|---|---|
| Deep CORAL, first pass, seed 52 only | 2026-11-08 | `b4f34fd` | `bfe7168` | 36 | replaced by the next block | `results/tables/T1_baseline_lambda0_targetpass_deep_coral.csv` |
| Source-only models for the seeds 100, 445, 545, 881 | 2026-11-09 | `7db879b` | `cfcdaa4` | 48 | `results/source_only/seed_<seed>/<family>/<transfer>/` | `results/tables/T5_summary_target_test.csv` (as baseline) |
| Deep CORAL and its two controls, five seeds | 2026-11-09 | `7db879b` | `cfcdaa4` | 180 | `results/deep_coral/`, `results/lambda0_target_pass/`, `results/source_only_continued/` | `results/tables/T5_summary_target_test.csv` |
| DANN and its control, five seeds | 2026-11-09 to 2026-11-10 | `d9ca82f` | `b2a6e71`, probe tables `d923183` | 120 | `results/dann/`, `results/dann_lambda0_target_pass/` | `results/tables/T2_summary_target_test.csv` |
| DSAN and its control, five seeds | 2026-11-10 | `cdef98b` | `8349276` | 120 | `results/dsan/`, `results/dsan_lambda0_target_pass/` | `results/tables/T3_summary_target_test.csv` |
| Alignment-weight sensitivity in the paper convention, seed 52 | 2026-11-11 to 2026-11-12 | `bd0cd35` | `907631e`, tables script `dad603e` | 192 | `results/paper_convention/` | `results/tables/T7_sensitivity_summary.csv` |
| Uniform soups (methods, controls; two BatchNorm variants), output ensembles, source-val predictions of the branches; five seeds | 2026-11-12 | `586579f` | `7c5b142`, tables script `6c280b9` | 240 soups, 120 ensembles, 360 source-val predictions | `results/soup/`, `results/ensemble/`, `results/source_val_predictions/` | `results/tables/T8_summary_target_test.csv`, `T8_differences_target_test.csv` |

The supplied source-only models of seed 52 (12 transfers, in `models/`) were trained by the supervisor and carry no commit of this repository.

**Current main tables** (all variants together): `results/tables/T3_summary_target_test.csv`, `T3_method_differences_target_test.csv`, `T3_runs_all_splits.csv`, `T3_domain_probe_summary.csv`, `T3_costs_per_variant.csv`, `T3_commits_per_variant.csv`.

## Notes per block

### Deep CORAL, first pass (seed 52)

- 12 transfers × 3 variants (Deep CORAL, control with target pass, control without target data), each from the supplied checkpoint.
- Two thirds of these runs were computed before the training code was under version control. They were therefore recomputed in the next block with the committed code; all 36 recomputed runs were bitwise identical to the first pass (predictions, loss logs, checkpoints). In `cfcdaa4` the first-pass folders were replaced by the recomputed runs under `seed_52/`.
- Results of a single seed; superseded by the five-seed tables.

### Source-only models and Deep CORAL on five seeds

- 48 source-only models (4 seeds × 12 transfers), training time 29.8 min in total.
- 180 runs (5 seeds × 12 transfers × 3 variants): Deep CORAL 61.6 min, control with target pass 60.9 min, control without target data 31.4 min.
- Plausibility check before the batch: a source-only model trained by our script for seed 52 on PAMAP2 → RealWorld reached 91.85 % macro-F1 on source test against 96.40 % for the supplied checkpoint (different device; the allowed gap was 10 points).
- Tables written by `make_seed_tables.py`, script commit `7db879b`.

### DANN

- Trial run first: seed 52 on PAMAP2 → RealWorld, RealWorld male → female and RealWorld <30 → ≥30. All three criteria (stable source loss, adversarial loss near ln 2 per domain and no collapse of the discriminator, source-test macro-F1 at most 15 points below the baseline) were met without looking at target metrics. The three trial runs are part of the 60 DANN runs.
- 120 runs: DANN 90.0 min, control 59.8 min.
- Tables written by `make_dann_tables.py` (commits `945394c`, `fa11a65`), domain probe by `make_domain_probe.py` (commit `fa11a65`).

### DSAN

- Trial run first: seed 52 on PAMAP2 → RealWorld. Source loss stable (last five epochs 0.67 times epochs 2 to 5), LMMD finite in every epoch (0.24 to 0.88), source-test macro-F1 94.02 % against 96.40 % for the baseline. The trial run is part of the 60 DSAN runs.
- 120 runs, computed on 2026-11-10 between about 02:25 and 04:28 local time: DSAN 70.4 min, control 51.6 min.
- All 120 runs were checked without target labels (completeness, configuration, start checkpoint, log length, finite values, prediction shapes, loading of the checkpoint). No deviation.
- Tables written by `make_dsan_tables.py` (commit `3f5768e`); domain probe for all seven variants with the unchanged `make_domain_probe.py`. The probe values of the five earlier variants are identical to the DANN block.

### Alignment-weight sensitivity (paper convention, seed 52)

- 3 methods × 5 factors × 12 transfers plus 12 runs of the shared control, all from the supplied seed-52 checkpoints. Source weight 1 for every method, λ = factor × (AdaTime alignment weight / AdaTime source weight), factors 0.25, 0.5, 1, 2, 4.
- The runs save their source-validation predictions as well (for SRC).
- The batch was stopped once by hand after 131 runs (before a restart of the computer) and continued with the same command; the interrupted run (DANN ×4, RealWorld ≥30 → <30) was recomputed completely. All 192 `config.json` carry `bd0cd35`.
- Summed training time 254.7 min. These times come from two states of the computer (before and after the restart, part of them next to another job) and are not a cost measure.
- Tables written by `make_sensitivity_tables.py` and `make_dev_check.py` (commit `dad603e`); figures 7 to 9 and the section in `results.md` by `make_sensitivity_figures.py`.

### Uniform soup (stage 3, averaging at the end)

- No training. Per transfer and seed (12 × 5) the soup of the three methods and the soup of their three controls, each with averaged (`bn_avg`) and re-estimated (`bn_target`) batch-normalisation statistics, and the output ensemble of the same three runs. The averaging step is `create_method_soup.py` of the project partner (Denis Petrov), extended by `average_state_dicts`.
- Before averaging, every ingredient was checked to have started from the source-only checkpoint of its seed (SHA-256). All 120 checks passed.
- The source-val predictions of the 360 phase-2 branch runs were computed from their checkpoints (prediction only). The predictions of the branches on the four other splits were recomputed as a check and agreed with the saved ones in every run (agreement 1.0).
- Computed on Apple MPS in one batch of 12 min wall time, without interruption. Averaging takes about 11 ms per soup, the batch-normalisation re-estimation about 2 s.
- Tables written by `make_soup_tables.py` (commit `6c280b9`); figures 10 to 12 and the section in `results.md` by `make_soup_figures.py`.

## Code commit of every variant

From `results/tables/T3_commits_per_variant.csv`:

| Variant | Code commit | Runs with commit | Supplied runs without commit |
|---|---|---:|---:|
| Baseline (source-only) | `7db879b` | 48 | 12 |
| Deep CORAL | `7db879b` | 60 | 0 |
| CORAL control | `7db879b` | 60 | 0 |
| DANN | `d9ca82f` | 60 | 0 |
| DANN control | `d9ca82f` | 60 | 0 |
| DSAN | `cdef98b` | 60 | 0 |
| DSAN control | `cdef98b` | 60 | 0 |

## Known small inconsistency between tables

The count "seeds in which the method is higher" for Deep CORAL minus its control on RealWorld female → male is 4 in `T3_method_differences_target_test.csv` and 3 in `results.md`. One of the five per-seed differences is smaller than 10⁻⁶; the table script counts it with full precision, while `make_figures.py` reads the rounded CSV, where it becomes 0. All other counts agree.
