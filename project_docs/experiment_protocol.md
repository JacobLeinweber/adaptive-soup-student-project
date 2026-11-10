# Experiment protocol

This page fixes how every run of phase 2 was set up and evaluated. Numbers of windows are taken from the run configurations and the result tables.

## Transfers and splits

Twelve transfers in three families. Source and target have the same classes within a transfer. The splits are the ones prepared by the supervisor; our code does not create or change splits.

| Family | Transfer | Classes | Source train | Target train | Target test |
|---|---|---:|---:|---:|---:|
| Cross-dataset | PAMAP2 → RealWorld | 5 | 3 646 | 9 685 | 2 894 |
| Cross-dataset | RealWorld → PAMAP2 | 5 | 9 685 | 3 646 | 1 426 |
| Cross-dataset | PAMAP2 → MHEALTH | 5 | 3 646 | 1 170 | 702 |
| Cross-dataset | MHEALTH → PAMAP2 | 5 | 1 170 | 3 646 | 1 426 |
| Cross-dataset | RealWorld → MHEALTH | 5 | 9 685 | 1 170 | 702 |
| Cross-dataset | MHEALTH → RealWorld | 5 | 1 170 | 9 685 | 2 894 |
| Cross-gender | PAMAP2 male → female | 12 | 5 902 | 1 374 | 157 |
| Cross-gender | PAMAP2 female → male | 12 | 1 374 | 5 902 | 3 270 |
| Cross-gender | RealWorld male → female | 8 | 6 540 | 5 498 | 2 525 |
| Cross-gender | RealWorld female → male | 8 | 5 498 | 6 540 | 2 569 |
| Cross-age | RealWorld <30 → ≥30 | 8 | 7 677 | 4 209 | 1 270 |
| Cross-age | RealWorld ≥30 → <30 | 8 | 4 209 | 7 677 | 2 567 |

Each group (for example "PAMAP2 male") has a train, a validation and a test split by subject. The exception is "PAMAP2 female", which consists of a single subject and is split in time (80/10/10). Each group is normalised with the statistics of its own training split, the target as well; this uses no labels.

How the splits are used:

| Split | Use |
|---|---|
| Source train (windows and labels) | training |
| Target train (windows only) | training of the UDA methods and controls, without labels |
| Source test | evaluation |
| Target train, target validation, target test (labels) | evaluation only, after all training |

Source validation is not used by our runs, because there is no model selection (see below).

## Seeds and start models

Five paired seeds: 52, 100, 445, 545, 881. For every transfer and seed there is exactly one source-only model, and **every method and every control of that seed starts from it**.

| Seed | Source-only model | Trained on |
|---|---|---|
| 52 | the checkpoint supplied in `models/<family>/<transfer>/` | CUDA, by the supervisor |
| 100, 445, 545, 881 | trained by `train_source_only.py` with the same procedure as the supervisor's `run_transfer`, stored in `results/source_only/seed_<seed>/` | Apple MPS |

The seed of a run sets the global random state, the order of the source batches and the stream of target batches. The baseline of a seed in all tables is the source-only model of that seed.

## Training budget

Identical for the source-only models, all methods and all controls:

- Adam, learning rate 3e-4, weight decay 1e-4, batch size 32, 30 epochs.
- **One epoch = one pass over the source training windows.** The number of steps per epoch is the number of source batches (37 to 303, depending on the source).
- **The target is cycled.** In every step one target batch of the same size as the source batch is drawn from a stream over the target training windows; when the stream is exhausted, the target windows are reshuffled and the stream starts again. A small target is therefore seen several times per epoch, a large one only in part.
- The source order is a new permutation per epoch from a generator seeded with the run seed, exactly as in the baseline.
- Source and target batches go through the model in two separate forward passes in train mode, source first.

A UDA run thus has the same number of optimisation steps as its source-only model had, on top of it.

## Control runs

| Variant | Target forward pass | Alignment weight | Source loss weight | Purpose |
|---|---|---|---|---|
| `lambda0_target_pass` ("CORAL control") | yes | 0 | that of Deep CORAL (0.1013) | effect of the CORAL loss alone |
| `dann_lambda0_target_pass` ("DANN control") | yes | 0 | that of DANN (5.1390) | effect of the domain loss alone |
| `dsan_lambda0_target_pass` ("DSAN control") | yes | 0 | that of DSAN (1.8745) | effect of the LMMD loss alone |
| `source_only_continued` | no | 0 | that of Deep CORAL | effect of continued training without any target data (Deep CORAL only) |

A control with target pass differs from its method **only** in the alignment loss: the target windows still pass through the model in train mode, so they still change the batch-normalisation statistics. Comparing a method with the baseline mixes three effects (30 more epochs, target statistics in batch normalisation, alignment loss); comparing it with its own control isolates the third. Both comparisons are reported.

## Model selection

None. The model after the **last epoch** is used for every run. No metric on any split decides which epoch, which hyper-parameter or which method is reported. In particular, target labels are not used for selection.

## Metrics

Computed by one common module on four splits (source test, target train, target validation, target test): accuracy, macro-F1, weighted F1 and balanced accuracy. Per-class F1 and confusion matrices are stored for target test.

- **Main metric: weighted F1 on the target test split**, reported as mean ± sample standard deviation over the five seeds.
- For a group of transfers the value is first averaged over the transfers within each seed, then mean and SD are taken over the seeds. The SD of a group therefore describes the variation between seeds, not between transfers.
- Differences between two variants are paired by seed: mean ± SD of the five per-seed differences and the number of seeds with a positive sign.
- Costs: training time of the training loop per run.

## The rule "no target labels in training"

Target labels steer neither the loss, nor the choice of a checkpoint, a hyper-parameter or a method. They appear only in the final evaluation. The code enforces this in three ways:

1. The training functions (`train_continued`, `train_dann`, `train_dsan`) have a parameter for the target windows and **no parameter for target labels**.
2. The run scripts load the target with `uda.data.load_unlabeled`, which opens only `features.pkl`. They save the predictions and do not score them.
3. Scoring happens in separate table scripts through `uda.evaluation`, after all runs of a batch have finished.

For DANN and DSAN this was also checked at run time: a complete run was repeated with a Python audit hook on every file that is opened, and the label file of the target was never opened. The pseudo-labels of DSAN are the model's own predictions.

Decisions that were taken after target-test numbers were known (for example which runs to report) do not change any training setting; where a setting could have been reconsidered in the light of the results (the batch-normalisation rule, the reversal factor of DANN, the constant weight of DSAN), it was left as fixed beforehand.

## Domain probe

A diagnostic, computed after all runs; it selects nothing.

- For every run, the saved model produces the 96-dim features of the source-train and the target-train windows in eval mode.
- The larger domain is subsampled so that both have the same number of windows.
- A logistic regression on standardised features is evaluated with 5-fold stratified cross-validation; it predicts the domain (source or target) of a window. The scaler is fitted inside each fold.
- Reported is the mean accuracy over the folds: 50 % means the domains cannot be separated linearly, 100 % that they are perfectly separable.
- Only the domain of a window is used, no activity labels.

The probe needs the checkpoints, which are not in the repository; its results are in `results/tables/T3_domain_probe_runs.csv` and `T3_domain_probe_summary.csv`.
