# Hyperparameter search (pre-experiment): protocol

Following the supervisor's advice in meeting 3, the loss weights are not assumed to carry over between datasets. As a pre-experiment, one transfer group (cross-dataset, six transfers) is searched with one seed (52), using AdaTime's procedure. **AdaTime is the reference protocol.** We deviate only where AdaTime demonstrably does not work or does not fit our setup; every deviation is listed below.

AdaTime: <https://github.com/emadeldeen24/AdaTime>, commit `2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d` (branch `adatime_v2`). All file and line references below are to this commit.

Code: `run_hp_search.py` (runs, start check, resume), `uda/hp_search.py` (combinations, training call, criteria), `make_hp_search_config.py` (writes the configuration once), `pack_hp_search.py` (ZIP archives). Configuration: `configs/hp_search_cross_dataset.json`, which fixes the search space with its sources, all 100 combinations per method, the input files with SHA-256 and the few-shot windows. Start card for running it on another machine: [hp_search_run_guide.md](hp_search_run_guide.md).

## Setup

| Item | Value |
|---|---|
| Transfers | the six cross-dataset transfers (PAMAP2, RealWorld, MHEALTH in both directions) |
| Methods | Deep CORAL, DANN, DSAN (our Stage-2 implementations, unchanged) |
| Seed | 52 |
| Start model | the supplied seed-52 source-only checkpoint of the transfer (`models/cross_dataset/<transfer>/model_last.pt`) |
| Fixed | 30 epochs, batch size 32, Adam with weight decay 1e-4, model after the last epoch, Stage-2 data pipeline and forward passes |
| Searched | learning rate, source-loss weight, alignment weight (per method) |
| Combinations | 100 per method: 0 = Stage-2 setting, 1 = AdaTime `hparams.py`, 2–99 = random draws |
| Same combinations for all transfers | yes, as AdaTime uses one setting for all scenarios of a dataset (`trainers/sweep.py:71-96`) |
| Order | combination by combination: combination *k* for all 6 transfers × 3 methods, then *k* + 1 |

The search is stopped at a fixed time, however far it got. Only combinations that are finished for all 18 transfer–method pairs are evaluated.

## Search space

AdaTime's wandb sweep receives only the per-method space `sweep_alg_hparams[da_method]` (`trainers/sweep.py:51-56`), with random search (`main_sweep.py:32`). Every parameter is drawn independently: from a list uniformly, or from a uniform distribution.

| Method | Parameter (our name) | AdaTime name | Space | Source |
|---|---|---|---|---|
| Deep CORAL | learning rate | `learning_rate` | {1e-2, 5e-3, 1e-3, 5e-4} | `configs/sweep_params.py:38` |
| | source weight | `src_cls_loss_wt` | U(0.1, 10) | `configs/sweep_params.py:39` |
| | alignment weight (CORAL) | `coral_wt` | U(0.1, 10) | `configs/sweep_params.py:40` |
| DANN | learning rate | `learning_rate` | {1e-2, 5e-3, 1e-3, 5e-4} | `configs/sweep_params.py:13` |
| | source weight | `src_cls_loss_wt` | U(0.1, 10) | `configs/sweep_params.py:14` |
| | alignment weight (domain loss) | `domain_loss_wt` | U(0.01, 10) | `configs/sweep_params.py:15` |
| DSAN | learning rate | `learning_rate` | {1e-2, 5e-3, 1e-3, 5e-4} | `configs/sweep_params.py:66` |
| | source weight | `src_cls_loss_wt` | U(0.1, 10) | `configs/sweep_params.py:67` |
| | alignment weight (LMMD) | `domain_loss_wt` | U(0.01, 10) | range of `mmd_wt`, `configs/sweep_params.py:68` (see below) |

**DSAN.** AdaTime's sweep varies `mmd_wt` for DSAN, but AdaTime's DSAN loss reads `domain_loss_wt` (`algorithms/algorithms.py:624`), which stays at 1.59 (`configs/hparams.py:60`). In AdaTime the LMMD weight is therefore never searched. We apply the range given for `mmd_wt` to the weight the loss uses.

The DANN discriminator uses the same learning rate and weight decay as the network (`algorithms/algorithms.py:306-310`), as in our implementation, so it follows the drawn learning rate.

## Reference combinations

| Id | Origin | Learning rate | Loss weights |
|---|---|---|---|
| 0 | Stage 2 | 3e-4 for all methods | AdaTime weights for HAR (CORAL 4.2304 / source 0.1013; DANN 2.9437 / 5.1390; DSAN 1.59 / 1.8745) |
| 1 | AdaTime `configs/hparams.py` | 5e-4 (Deep CORAL, lines 71-76), 1e-3 (DANN, lines 42-47; DSAN, lines 56-62) | the same weights |

Stage 2 ran with learning rate 3e-4, the learning rate of the supplied source-only training, not with AdaTime's learning rates. Combination 0 repeats the Stage-2 runs of seed 52 exactly and serves as a device check (CUDA here, Apple MPS in Stage 2). Combination 1 is AdaTime's own setting.

Random draws (ids 2–99): `numpy.random.default_rng([52, i])` with *i* the index of the method (Deep CORAL 0, DANN 1, DSAN 2); per combination in the order learning rate, source weight, alignment weight. The drawn values are stored in the configuration, so every machine runs the same values.

## Deviations from AdaTime

1. **Epochs and batch size are fixed** at 30 and 32. AdaTime trains 40 epochs from scratch (`configs/hparams.py:13`); our runs continue the source-only model for the Stage-2 budget. `configs/sweep_params.py:1-10` also defines a training space (`num_epochs` 3–6, `batch_size` 32/64, `disc_lr`, `weight_decay`, `step_size`, `gamma`), but AdaTime's sweep does not use it (`trainers/sweep.py:55`), so AdaTime does not vary them either.
2. **Weight decay** stays at 1e-4 (AdaTime's HAR value, not part of its method space).
3. **No learning-rate scheduler.** AdaTime builds `StepLR(step_size 50, gamma 0.5)`, which never steps within its 40 epochs.
4. **DSAN alignment weight**: the `mmd_wt` range is applied to `domain_loss_wt` (above).
5. **Combination 0** uses learning rate 3e-4 (Stage 2) instead of AdaTime's learning rates; combination 1 is AdaTime's setting.
6. **Random draws with numpy** instead of wandb, with the same sampling rule.
7. **SRC on source validation**, not on source test as in AdaTime (`trainers/abstract_trainer.py:246`), so that source test stays untouched by any selection.
8. **FST on target validation**: AdaTime draws 5 windows per class from the target **test** split (`trainers/abstract_trainer.py:147-148`, `dataloader/dataloader.py:106-128`). We draw them once from target validation with a fixed seed and use the same windows for every run, so that target test is never used for a selection.

## What every run stores

In `results/hp_search/cross_dataset_seed52/runs/c<id>/<method>/<transfer>/`:

| File | Content |
|---|---|
| `model_last.pt`, `discriminator_last.pt` (DANN) | checkpoints; not in git |
| `train_log.csv` | losses per epoch |
| `outputs.npz` | predictions on source val, source test, target train, target val, target test; cross-entropy per source-val window; DEV importance weight per source-val window |
| `report_only.json` | target train and target test (cross-entropy, accuracy, macro-F1, weighted F1, balanced accuracy): **for the report only** |
| `result.json` | written last: setting, training summary, criteria, device, PyTorch version, code commit, run time |

Criteria in `result.json` (and in `criteria.csv`, one row per finished run):

| Prefix | Criterion | Labels used |
|---|---|---|
| `src_` | SRC: cross-entropy, macro-F1 and the other metrics on source validation | source labels only |
| `source_test_` | source test (information) | source labels only |
| `dev_` | DEV as in `uda/criteria.py` (domain classifier seed 52), with the raw quantities in `outputs.npz` so that DEV can be recomputed | source labels only |
| `fst_` | few-shot target: 5 fixed labelled windows per class from target validation | **25 target labels** (few-shot) |
| `oracle_target_val_` | all of target validation | **target labels** (oracle, reference only) |

Target test appears only in `report_only.json` and `report_only_target.csv` and is never used to choose a combination, a method or a run. Training receives target windows only.

## Robustness

Before the first run, `run_hp_search.py` checks the Python packages, the device (including a test forward and backward pass), that code and configuration equal the last commit, every input file (size and SHA-256, no Git LFS pointer), the data and checkpoints themselves (shapes, label maps, strict loading, few-shot windows) and the free disk space, and stops with a list of all problems. Each file of a run is written under a temporary name and renamed; `result.json` comes last, so an interrupted run is recognised and redone, and finished runs are skipped when the same command is started again. A failing run is logged in `errors.log` and skipped; the search stops by itself only if the device stops working, the code changes, the disk fills up or 18 runs fail in a row. Progress and the estimated remaining time are printed after every run and written to `progress.json`.
