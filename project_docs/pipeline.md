# Pipeline

## Our code and the supervisor's code

The supervisor's code in `src/`, the data in `data/` and the supplied models in `models/` are used read-only and are not modified. Our code lives in the package `uda/` and in scripts in the repository root.

From the supervisor's code we use the list of transfers (`src/training/experiments.py`), the model `SelfHARCNN` (`src/training/model.py`), the data loading and the helpers of `src/training/train_transfer.py` and `src/training/utils.py` (seeding, JSON writing). **`run_transfer` is never called**, because it writes into `models/` and would overwrite the supplied checkpoints. All our outputs go to `results/`; `uda.data.check_output_dir` refuses output folders inside `models/`, `data/` or `src/`.

### Package `uda/`

| Module | Role |
|---|---|
| `uda/data.py` | Access to data and checkpoints on top of the supervisor's loaders. `load_labeled` returns windows and labels (used for the source only), `load_unlabeled` opens only `features.pkl` and returns windows without labels (used for the target). Also checkpoint paths, SHA-256, label maps. |
| `uda/features.py` | `extract_features`: the 96-dim features after pooling, without changing the model code. |
| `uda/losses.py` | The Deep CORAL loss. |
| `uda/training.py` | `train_continued`: the training loop for Deep CORAL and its controls. `CyclicTargetSampler`: the stream of target batches, reshuffled after every pass; also used by DANN and DSAN. |
| `uda/dann.py` | Gradient reversal layer, domain discriminator, reversal factor and `train_dann`. |
| `uda/dsan.py` | Gaussian kernel, class weights, LMMD loss and `train_dsan`. |
| `uda/evaluation.py` | The common evaluation: predictions, accuracy, macro-F1, weighted F1, balanced accuracy, per-class F1, confusion matrix. **The only module that reads target labels**, after training and for scoring only. |
| `uda/runs.py` | Bookkeeping: the seeds, the baseline hyper-parameters, `clean_commit` (provenance), `is_complete` (resume), the path of the source-only model of a seed, environment information, training log. |

### Scripts

| Script | Role |
|---|---|
| `evaluate_baseline.py` | Evaluates a supplied source-only checkpoint without retraining it. |
| `train_source_only.py` | Trains the source-only models for the seeds 100, 445, 545 and 881, following the steps of `run_transfer` one by one. Seed 52 uses the supplied checkpoint. |
| `run_deep_coral.py` | Deep CORAL and its two controls (`lambda0_target_pass`, `source_only_continued`). |
| `run_dann.py` | DANN and its control (`dann_lambda0_target_pass`). |
| `run_dsan.py` | DSAN and its control (`dsan_lambda0_target_pass`). |
| `make_results_tables.py` | First tables for seed 52 only (`T1_*.csv`, `baseline_all_transfers.csv`); superseded by the five-seed tables, kept for `write_csv`. |
| `make_seed_tables.py` | Five-seed tables for the baseline, Deep CORAL and its controls (`T5_*.csv`). |
| `make_dann_tables.py` | Adds DANN and its control (`T2_*.csv`). |
| `make_dsan_tables.py` | All three methods with their controls and the baseline (`T3_*.csv`); stops if the runs of a variant carry different code commits. This is the current main set of tables. |
| `make_domain_probe.py` | The domain probe (`<prefix>_domain_probe_*.csv`); needs the checkpoints. |
| `make_figures.py` | Figures and `project_docs/results.md`, built only from `results/tables/*.csv`. |

The run scripts open target data only as windows; they never open the label file of the target. The table scripts read the target labels, after all training has finished.

## Commands from zero to the figures

Run from the repository root, in this order. Every run script skips finished runs, so an interrupted call can simply be repeated. Times are the summed training times from `results/tables/T3_costs_per_variant.csv` and `T5_costs_per_variant.csv` on an Apple M1 (MPS); loading, prediction and writing come on top.

| Step | Command | Runs | Training time |
|---|---|---:|---:|
| 1 | `python train_source_only.py --all` | 48 | 30 min |
| 2 | `python run_deep_coral.py --all --all-variants` | 180 | 154 min (61.6 + 60.9 + 31.4) |
| 3 | `python run_dann.py --all --all-variants` | 120 | 150 min (90.0 + 59.8) |
| 4 | `python run_dsan.py --all --all-variants` | 120 | 122 min (70.4 + 51.6) |
| 5 | `python make_seed_tables.py`, `python make_dann_tables.py`, `python make_dsan_tables.py` | | under a minute each |
| 6 | `python make_domain_probe.py --prefix T3 --variants baseline deep_coral lambda0_target_pass dann dann_lambda0_target_pass dsan dsan_lambda0_target_pass` | | about 5 min |
| 7 | `python make_figures.py` | | seconds |

Useful options of the run scripts: `--transfer <name>` instead of `--all`, `--seeds 52 100`, `--control` (only a control; see `--help` of each script, Deep CORAL has two controls), `--device cuda`, `--output-dir <folder>` (for tests outside `results/`), `--overwrite`.

Steps 5 to 7 can be repeated at any time. Steps 5 and 6 need the prepared data (labels), step 6 also the checkpoints; step 7 needs only the tables.

## Where results are stored

```text
results/
├── source_only/seed_<seed>/<family>/<transfer>/      source-only models of the seeds 100, 445, 545, 881
├── deep_coral/seed_<seed>/<transfer>/
├── lambda0_target_pass/seed_<seed>/<transfer>/       control of Deep CORAL
├── source_only_continued/seed_<seed>/<transfer>/     control of Deep CORAL without target data
├── dann/seed_<seed>/<transfer>/
├── dann_lambda0_target_pass/seed_<seed>/<transfer>/
├── dsan/seed_<seed>/<transfer>/
├── dsan_lambda0_target_pass/seed_<seed>/<transfer>/
└── tables/                                           all CSV tables
```

Every run folder contains:

| File | Content |
|---|---|
| `config.json` | everything needed to identify the run (below); written last |
| `train_log.csv` | one row per epoch: the losses of the method, steps, target passes, seconds |
| `predictions/<split>.npz` | predicted classes for `source_test`, `target_train`, `target_val`, `target_test` |
| `model_last.pt` | weights of the HAR predictor after the last epoch; **local only, not in the repository** |
| `discriminator_last.pt` | DANN only: the discriminator; local only |

Fields of `config.json` (the method-specific ones differ in name): method, transfer, family, source, target, label map, seed; path and SHA-256 of the start checkpoint; loss weights; optimiser, epochs, batch size, learning rate, weight decay; method details (for example kernel settings or discriminator settings); steps per epoch and the number of source and target training windows; the reference to the AdaTime files and commit; device, PyTorch, NumPy and Python versions, platform and a note on non-determinism; `repository_commit`; start and end time (UTC) and `train_seconds`.

## Provenance

- **Clean commit.** Before every run, `clean_commit` checks with `git status --porcelain -- uda/ "*.py"` that the package `uda/` and all Python files match the last commit, including untracked files. Otherwise the run stops. The commit hash is written into `config.json` as `repository_commit`.
- **Consequence.** While a batch of runs is computing, nothing may be committed and no Python file may be added or changed in the repository; the next run of the batch would stop, or would carry a different commit.
- **Start checkpoint.** The SHA-256 of the start checkpoint is computed before and after the run and written into `config.json`.
- **One commit per variant.** `make_dsan_tables.py` stops before writing tables if the runs of a variant carry different code commits, and writes the commit of every variant to `results/tables/T3_commits_per_variant.csv`. The 12 supplied seed-52 baselines were trained by the supervisor and carry no commit of this repository.

## Resume and overwrite

A run counts as finished only if all its files exist and `config.json` can be read; `config.json` is written last. Repeating a command skips finished runs and recomputes started ones. After a correction of the code, affected runs must be recomputed explicitly with `--overwrite`; without it the old results would be kept and skipped.
