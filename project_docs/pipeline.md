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
| `uda/criteria.py` | Label-free criteria ported from AdaTime: SRC (`source_risk`, cross-entropy and metrics on source validation) and DEV (ported, not used, see `results.md`). |
| `uda/soup.py` | The ingredients of the two soups, the batch-normalisation re-estimation on target-train windows (`recompute_bn`), logits and softmax for the soups and the output ensembles. |
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
| `run_paper_convention.py` | Alignment-weight sensitivity in the convention of the papers (seed 52): three methods × five factors and one shared control. |
| `make_sensitivity_tables.py`, `make_dev_check.py` | SRC, choices and scores of the sensitivity (`T7_*.csv`); the check of the DEV importance weights. |
| `create_method_soup.py` | The soup script of the project partner (Denis Petrov): averages the checkpoints of three runs per transfer and seed. Its averaging step `average_state_dicts` is used by `run_soup.py`. |
| `run_soup.py` | Stage 3: per transfer and seed the soups of the methods and of the controls in both batch-normalisation variants, the two output ensembles, and the source-validation predictions of the six branches; checks the shared start checkpoint; writes predictions and `config.json`. |
| `make_soup_tables.py` | SRC and the "best branch" choice without target labels, then the scores and paired comparisons of the soups (`T8_*.csv`). |
| `make_figures.py` | Figures and `project_docs/results.md`, built only from `results/tables/*.csv`; adds the sections of `make_sensitivity_figures.py` and `make_soup_figures.py` when their tables exist. |

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
| 7 | `python run_paper_convention.py --all`, then `python make_sensitivity_tables.py` and `python make_dev_check.py` | 192 | about 4 h (see `runs_log.md`) |
| 8 | `python run_soup.py --all` | 240 soups, 120 ensembles | 12 min wall time on MPS (no training) |
| 9 | `python make_soup_tables.py` | | under a minute |
| 10 | `python make_figures.py` | | seconds |

Useful options of the run scripts: `--transfer <name>` instead of `--all`, `--seeds 52 100`, `--control` (only a control; see `--help` of each script, Deep CORAL has two controls), `--device cuda`, `--output-dir <folder>` (for tests outside `results/`), `--overwrite`.

Steps 5, 6, 9 and 10 can be repeated at any time. Steps 5, 6 and 9 need the prepared data (labels); steps 6 and 8 and the criteria of step 7 need the checkpoints; step 10 needs only the tables.

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
├── paper_convention/<method>/x<factor>/seed_52/<transfer>/   alignment-weight sensitivity, and control/seed_52/<transfer>/
├── soup/<soup_methods|soup_controls>/<bn_avg|bn_target>/seed_<seed>/<transfer>/
├── ensemble/<methods|controls>/seed_<seed>/<transfer>/
├── source_val_predictions/<branch>/seed_<seed>/<transfer>/   source-val predictions of the phase-2 branches, for SRC
└── tables/                                           all CSV tables
```

A soup folder holds `model_last.pt` (local only), `config.json` and `predictions/<split>.npz` for `source_val`, `source_test`, `target_train`, `target_val` and `target_test`; `source_val.npz` also holds the logits. An ensemble folder holds the same without a checkpoint; its `source_val.npz` holds the mean probabilities. The `config.json` of a soup lists every ingredient with path, SHA-256, start checkpoint and code commit, the start checkpoint, the averaging rule, the batch-normalisation variant (with batch size, number of batches and order of the re-estimation), the seconds of each step and the code commit.

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
- **Soups.** `run_soup.py` checks the clean commit before and after the whole batch; `make_soup_tables.py` stops unless all soups, ensembles and source-val predictions carry one commit and every soup's ingredients share its start checkpoint, and writes the commits of the outputs and of the ingredient runs to `results/tables/T8_commits.csv`.
- **One commit per variant.** `make_dsan_tables.py` stops before writing tables if the runs of a variant carry different code commits, and writes the commit of every variant to `results/tables/T3_commits_per_variant.csv`. The 12 supplied seed-52 baselines were trained by the supervisor and carry no commit of this repository.

## Resume and overwrite

A run counts as finished only if all its files exist and `config.json` can be read; `config.json` is written last. Repeating a command skips finished runs and recomputes started ones. After a correction of the code, affected runs must be recomputed explicitly with `--overwrite`; without it the old results would be kept and skipped.
