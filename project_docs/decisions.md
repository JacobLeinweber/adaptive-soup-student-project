# Decisions

The main decisions of phase 2, each with its reason and, where relevant, the alternative that was rejected. Requirements from the project kickoff are marked as such; they take precedence over our own decisions.

## Scope

**Three methods, one per branch: Deep CORAL, DANN, DSAN.** One discrepancy-based global method, one adversarial method and one class-wise method, so that the branches of a later soup differ in mechanism. All three have a reference implementation for time series in AdaTime. Details in [methods.md](methods.md).

**All methods on all 12 transfers.** A method that helps on one transfer and harms on another can only be judged on the full set. The results confirm this: every method lowers the mean on at least one transfer.

**No soup yet.** The averaging is tested only after all three methods are available separately and their individual behaviour is documented.

## Requirements from the kickoff

- Target labels steer neither training nor the choice of checkpoints, methods or soup weights; they appear only in the final evaluation.
- Same CNN architecture, channels and activities for all branches, and the same source checkpoint per transfer; only the HAR predictor is averaged, not method-specific heads such as a discriminator.
- Five paired seeds (52, 100, 445, 545, 881), with independent branches from one shared checkpoint per seed.
- Main metric: weighted F1 on the held-out target test split, mean ± standard deviation over the five seeds; in addition macro-F1, accuracy, per-class results, confusion matrices and a cost analysis.

## Training set-up

**Every branch starts from the source-only model of its transfer and seed.** This is the kickoff requirement of a shared checkpoint. *Our reading:* "one shared checkpoint per seed" means that the source-only model itself is trained anew for each seed. Seed 52 uses the supplied checkpoint; for the other four seeds we trained source-only models with the procedure of the supervisor's training function, step by step and in the same order. *Rejected:* starting all five seeds from the one supplied checkpoint, because the seeds would then share their largest source of variation.

**The supervisor's `run_transfer` is never called.** It overwrites the checkpoints in `models/`. Our own loops write only to `results/`.

**Features for the alignment: the 96-dim output after pooling.** It can be read from the model without changing the model code, and it is the input of the classifier.

**Target labels cannot physically reach the training functions.** The training functions have no parameter for them, and the target is loaded by a function that opens only the feature file. A rule that is only written down can be broken by accident; a missing parameter cannot.

**One epoch = one pass over the source batches, the target is cycled.** This gives every method the same number of optimisation steps as the baseline, independent of the size of the target. *Rejected:* an epoch defined by the larger of the two domains, which would give transfers with a large target many more steps than the baseline had.

**Separate forward passes for source and target in train mode, the same for all methods.** This is what the reference implementation does. It means that batch normalisation sees target statistics. The rule was fixed before results were known and was **not** changed afterwards, since changing it because of target-test numbers would be a decision based on target labels.

**Loss, loss weights and schedules as in AdaTime; optimiser as in the baseline.** We take the loss code and the tuned weights for the HAR dataset of AdaTime, but keep Adam with learning rate 3e-4, weight decay 1e-4, batch size 32 and 30 epochs, so that all branches and the baseline share one budget. *Consequence:* the weights are not tuned for our model and data; see [limitations.md](limitations.md). Tuning is a separate later step and needs a criterion without target labels.

**DANN: reversal factor literally as in AdaTime.** The AdaTime formula makes the factor practically 1 from the first step, unlike the rising schedule of the paper. We kept the formula of the reference implementation and fixed a fallback **before** the trial run: had the trial run failed its stability criteria, the paper formula would have been used. It did not fail.

**DSAN: weight literally as in AdaTime.** Constant weight 1.59 from the first step, without the rising factor of the DSAN authors' code, for the same reason as for DANN.

**Model selection: last epoch.** There is no accepted label-free selection criterion in the project yet. Selecting an epoch by target performance is excluded; selecting by source validation was not done, to keep the procedure identical to the baseline.

## Controls

**Every method has its own control with the alignment weight set to 0, the same target forward pass and the same source loss weight.** The three methods use very different source loss weights (0.10, 5.14, 1.87), and continued training with target statistics in batch normalisation changes the result on its own. Without a matching control, the effect of the alignment loss cannot be separated from these two effects.

**A method is compared with both the baseline and its own control.** The baseline comparison answers "is the branch better than the starting point", the control comparison answers "is that due to the alignment loss".

## Trial runs before the full runs

**DANN was first run on one transfer per family, DSAN on one transfer, with criteria that use no target labels:** the source loss stays stable after the first epoch; the adversarial loss neither collapses nor explodes (DANN) or the LMMD values are finite (DSAN); macro-F1 on source test is at most 15 points below the baseline. Target metrics were not computed for this decision. The threshold of 15 points only excludes collapses: in the first evaluation, continued training alone already moved macro-F1 on source test by −9.7 to +13.3 points, depending on the transfer.

## Evaluation and reporting

**One evaluation module for the baseline and all runs,** so that every number in the tables is computed by the same code. The baseline of seed 52 is scored from the supplied predictions.

**Report mean ± SD over five seeds, paired differences and the number of positive seeds; no single-seed claims.** A first evaluation with seed 52 only suggested effects that reversed or vanished over five seeds.

**Domain probe as a diagnostic.** To see whether a method actually makes the features of the two domains more similar, a linear classifier is trained afterwards to separate source from target features. It explains, it does not select.

## Reproducibility and versioning

**Runs record the code commit and refuse to start on uncommitted code.** Every `config.json` names the commit of the code that produced the run. While a batch of runs is computing, nothing is committed. The table script checks that all runs of a variant carry the same commit.

**After the first commit of the code, the seed-52 runs were recomputed with the committed code.** The recomputed runs were bitwise identical to the earlier ones.

**Runs are resumable; recomputation after a fix needs `--overwrite`.**

**Training on Apple MPS.** The available machine is an Apple M1. PyTorch does not guarantee bitwise reproducibility on MPS; every `config.json` says so.

**In the repository: code, tables, predictions, logs, configurations. Not in the repository: checkpoints.** Large files in a fork would count against the Git LFS quota of the supervisor's repository. Where the checkpoints should be stored is an open question for the supervisor.
