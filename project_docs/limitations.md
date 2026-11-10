# Limitations

What the results in [results.md](results.md) do not show, and where they should be read with care.

## Data and splits

- **PAMAP2 female is a single subject.** The group is split in time (80/10/10), and its validation and test splits have only 157 windows with 5 to 20 windows per class. The results of PAMAP2 male → female (target test: 157 windows) and, to a lesser degree, PAMAP2 female → male (source of one subject) are not reliable.
- **Few subjects per split in general.** The test splits contain one to three subjects. A result on a target test split is a result for these few persons.
- **MHEALTH is small** (1 170 training windows). With MHEALTH as source an epoch has only 37 steps, and the source loss is close to zero from the start; for Deep CORAL the alignment term then dominates the total loss.
- **The baseline already contains a weak form of adaptation.** Every group is normalised with the statistics of its own training split, the target as well. This uses no labels, but "source-only" here means source-only training on per-domain normalised data.

## Seeds and start models

- **Device mix for seed 52.** The source-only models of seed 52 are the supplied checkpoints, trained on CUDA; the source-only models of the other four seeds were trained on Apple MPS. All UDA runs, of every seed, ran on MPS. Seed 52 is therefore not exchangeable with the other seeds in every respect.
- **Shared source-only models, runs are not independent.** For the seeds 100 to 881, cross-dataset transfers with the same source share one source-only model (the model does not depend on the target), so the six cross-dataset transfers of such a seed start from three different models. For seed 52 the six supplied models are all different. Transfers with the same source are correlated, and the 60 combinations of transfer and seed are not 60 independent experiments.
- **Five seeds.** Standard deviations over five values are themselves uncertain. For single transfers the SD of a paired difference reaches 20 points. No significance tests were computed. Where two variants differ by less than the SD of their paired difference, the order is not established; this holds in particular for DSAN against DANN over all transfers (+0.2 ± 3.1 points).
- **Reproducibility.** PyTorch does not guarantee bitwise identical results on MPS. A repeated run was bitwise identical on our machine for the transfers we checked; this is not shown for other machines or versions.

## Training and model selection

- **Last epoch, no model selection.** Every number is the model after epoch 30. The value of the last epoch is noisy, and a method might be better or worse at another epoch. There is no label-free selection criterion in the project yet.
- **Loss weights are not tuned.** All alignment weights and source loss weights are taken from AdaTime, where they were tuned for another dataset (UCIHAR), another backbone, a learning rate of 1e-3 and 128-dim features. Here the features have 96 dimensions and the learning rate is 3e-4. For Deep CORAL the same weight acts about 1.78 times more strongly at 96 dimensions.
- **The comparison between the methods is not a tuned comparison.** Because of the previous point, "DANN and DSAN are better than Deep CORAL" is a statement about these three configurations, not about the methods in general.
- **Very different source loss weights.** The three methods weight the source cross-entropy with 0.10, 5.14 and 1.87. With Adam this acts roughly like a different strength of weight decay. It affects comparisons between methods and with the baseline; the comparison of a method with its **own** control is not affected, because both use the same weight.
- **DANN deviates from the paper.** The gradient reversal factor follows the AdaTime formula and is practically 1 from the first step, instead of rising from 0 to 1 as in the paper. The paper schedule was not run.
- **DSAN deviates from the authors' code.** The LMMD weight is constant (1.59) from the first step, without the rising factor used in the code of the DSAN authors, and the optimiser is Adam instead of SGD. The variant with a rising weight was not run.
- **Rising source loss with DSAN.** In 23 of the 60 DSAN runs the source cross-entropy of the last five epochs is more than 1.5 times that of epochs 2 to 5 (DANN: 3 of 60). The source-test performance is not lower on average, but the alignment term works against the source loss at this weight. The single-seed trial run before the full runs did not show this.

## Interpretation

- **Negative transfer exists.** On MHEALTH → PAMAP2 all three methods are below the baseline on average. The cause was not investigated.
- **The gain is concentrated.** Three transfers (PAMAP2 ↔ RealWorld and RealWorld <30 → ≥30) carry about two thirds of the mean gain of DANN and DSAN over the baseline, and more for Deep CORAL.
- **Domain probe: DANN leaves the domains linearly separable.** On the DANN features a linear classifier separates source from target with 93.5 % accuracy, the same as on the baseline features; Deep CORAL lowers this slightly (92.0 %), DSAN clearly (81.7 %). The gain of DANN on the target is therefore **not** explained by a measurable loss of linear separability. **Where the gain of DANN comes from is open.**
- **The probe is limited.** It tests only linear separability and only globally, not per class. Its folds are random and not split by subject, and the windows overlap by 50 %, so its accuracy is rather too high in absolute terms. It is suited for comparing variants, which use the same windows and folds per seed.
- **Lower separability is not shown to cause the gain.** For DSAN the reduction in probe accuracy and the gain over the control are only weakly related across runs (rank correlation −0.25 over the 60 runs, which are not independent).
- **Pseudo-labels cannot be checked without target labels.** DSAN aligns per class according to the model's own predictions. Whether a class of the target is aligned with the right class of the source is not known during training. The confidence of the pseudo-labels is not a measure of their correctness.
- **No causes are claimed.** The tables show how the variants differ, not why.

## Scope

- Only phase 2 is covered. No soup has been built, so nothing here says whether averaging the branches works.
- The checkpoints are stored only locally and are not part of the repository.
