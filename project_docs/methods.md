# Methods

Three UDA methods are implemented, one per future branch of the model soup. All three share the same frame:

- **Model:** the supervisor's `SelfHARCNN`: three convolution blocks with batch normalisation and dropout, global pooling to a 96-dimensional feature vector $f$, and a classifier (one hidden layer) on top. The model code is not changed.
- **Start:** the source-only model of the transfer and seed. A method continues training this model.
- **Step:** one labelled source batch and one unlabelled target batch of the same size (32). They go through the model in **two separate forward passes in train mode**, source first. Batch normalisation therefore normalises each batch with its own statistics, and the running statistics are updated by both domains.
- **Optimiser:** Adam with the values of the baseline (learning rate 3e-4, weight decay 1e-4), 30 epochs, last epoch.
- **Loss, loss weights and schedules:** taken from the AdaTime implementation and its hyper-parameters for the HAR (UCIHAR) dataset, at commit `2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d` of <https://github.com/emadeldeen24/AdaTime> (branch `adatime_v2`). The weights were **not** tuned for this project, see [limitations.md](limitations.md).

Each method has a **control run**: the same loop, the same target forward pass and the same source loss weight, but the alignment weight is 0. The target windows then influence the model only through the batch-normalisation statistics. The difference between a method and its control is the effect of the alignment loss alone.

In the formulas, $x^s, y^s$ are a source batch with labels, $x^t$ a target batch without labels, $f^s, f^t$ their features (each $n \times d$ with $n = 32$, $d = 96$), and $\mathcal{L}_{CE}$ the cross-entropy of the classifier on the source batch.

## 1. Deep CORAL

**Idea.** Align the second-order statistics of the features: the covariance matrix of the source features and that of the target features should be equal. The alignment is global; classes play no role.

**Loss.** The paper defines

$$\mathcal{L}_{CORAL} = \frac{1}{4d^2}\,\lVert C_S - C_T \rVert_F^2 ,\qquad C = \frac{1}{n-1}\,F_c^\top F_c ,$$

where $F_c$ is the feature batch with the batch mean subtracted. AdaTime, and therefore this project, computes the scatter matrices without the factor $1/(n-1)$ and averages the squared difference over the $d^2$ entries before dividing by $4d^2$. The value used here is the paper value times $(n-1)^2/d^2$.

$$\mathcal{L} = 0.1013 \cdot \mathcal{L}_{CE} + 4.2304 \cdot \mathcal{L}_{CORAL}$$

| | |
|---|---|
| What is aligned | covariance of the 96-dim features of a source and a target batch |
| Extra components | none |
| Weights | `coral_wt` 4.2304, `src_cls_loss_wt` 0.1013 (AdaTime `configs/hparams.py:71-76`) |
| Loss code | AdaTime `models/loss.py:109-127`, ported to `uda/losses.py` without changing the arithmetic |
| Control | `lambda0_target_pass` (CORAL weight 0). A second control without any target data exists only for this method (`source_only_continued`). |

Because the feature dimension here is 96 and not 128 as in AdaTime, the same weight acts about 1.78 times more strongly than in AdaTime ($128^2/96^2$).

## 2. DANN

**Idea.** A second network, the domain discriminator, tries to tell from the features whether a window comes from source or target. The feature extractor is trained to make this impossible. This is done with a gradient reversal layer (GRL): it passes the features on unchanged and multiplies the gradient by $-\alpha$ on the way back. The alignment is global and adversarial.

**Loss.**

$$\mathcal{L} = 5.1390 \cdot \mathcal{L}_{CE} + 2.9437 \cdot \big(\mathcal{L}_{dom}(D(\mathrm{GRL}_\alpha(f^s)), 1) + \mathcal{L}_{dom}(D(\mathrm{GRL}_\alpha(f^t)), 0)\big)$$

$\mathcal{L}_{dom}$ is the cross-entropy of the discriminator $D$ with domain label 1 for source and 0 for target.

| | |
|---|---|
| What is aligned | the features of both domains, through a discriminator that should not be able to separate them |
| Extra components | discriminator (96 → 64 → 64 → 2, ReLU, no batch normalisation, no dropout) and the gradient reversal layer |
| Optimisers | Adam for the network as above; a second Adam for the discriminator with the same learning rate and weight decay and betas (0.5, 0.99) as in AdaTime |
| Weights | `domain_loss_wt` 2.9437, `src_cls_loss_wt` 5.1390 (AdaTime `configs/hparams.py:42-47`) |
| Code | AdaTime `models/models.py:229-261`, `algorithms/algorithms.py:327-364`, ported to `uda/dann.py` |
| Saved | `model_last.pt` holds only the HAR predictor (features and classifier); the discriminator is saved separately and is not part of a later soup |
| Control | `dann_lambda0_target_pass` (domain weight 0, no discriminator) |

**Deviation from the paper: the reversal factor $\alpha$.** In the paper (Ganin et al., section 5.2.2) the factor follows $\lambda_p = 2/(1+e^{-10p}) - 1$, where $p$ is the training progress from 0 to 1, so the factor rises from 0 to 1 over the training. AdaTime uses the same expression but computes $p$ as

$$p = \frac{\text{step} + \text{epoch} \cdot B}{E} + \frac{1}{B}$$

with $B$ batches per epoch, $E$ epochs, and the epoch counted from 1. With the batch counts of this project (37 to 303 per epoch) $p$ is above 1 from the first step, so **$\alpha$ is practically 1 from the first step on**. We adopted the AdaTime formula literally. Our DANN is therefore "DANN as implemented in AdaTime", without the rising schedule of the paper.

## 3. DSAN

**Idea.** Deep CORAL and DANN align the feature distributions as a whole. DSAN aligns them **per class** (per "subdomain"): the target features that the model assigns to a class should be distributed like the source features with that label. As there are no target labels, DSAN uses the model's own softmax outputs as soft pseudo-labels. The distance is the local maximum mean discrepancy (LMMD).

**Loss.** With a kernel $k$ and weights $w$,

$$\mathcal{L}_{LMMD} = \frac{1}{|C|} \sum_{c \in C} \Big[ \sum_{i,j} w^{s}_{ic} w^{s}_{jc}\, k(f^s_i, f^s_j) + \sum_{i,j} w^{t}_{ic} w^{t}_{jc}\, k(f^t_i, f^t_j) - 2 \sum_{i,j} w^{s}_{ic} w^{t}_{jc}\, k(f^s_i, f^t_j) \Big]$$

- $w^s_{ic}$ is the one-hot source label of window $i$, divided by the number of source windows of class $c$ in the batch.
- $w^t_{jc}$ is the predicted probability of class $c$ for target window $j$, divided by the sum of this probability over the batch. No gradient flows through the pseudo-labels.
- $C$ is the set of classes that occur among the source labels **and** among the hard predictions (arg max) of the target batch. Without such a class the loss of the step is 0.
- $k$ is a sum of five Gaussian kernels. The base bandwidth is the mean squared distance over all pairs of different windows in the stacked batch (no gradient); the five bandwidths are 0.25, 0.5, 1, 2 and 4 times this value.

$$\mathcal{L} = 1.8745 \cdot \mathcal{L}_{CE} + 1.59 \cdot \mathcal{L}_{LMMD}$$

| | |
|---|---|
| What is aligned | the 96-dim features of source and target per class, weighted with soft pseudo-labels |
| Extra components | none (no discriminator, no second optimiser) |
| Weights | `domain_loss_wt` 1.59, `src_cls_loss_wt` 1.8745 (AdaTime `configs/hparams.py:56-62`) |
| Code | AdaTime `models/loss.py:205-290`, `algorithms/algorithms.py:603-636`, ported to `uda/dsan.py` |
| Control | `dsan_lambda0_target_pass` (LMMD weight 0; the LMMD value is still logged) |

Three points about the adoption from AdaTime:

1. **Which weight.** The AdaTime entry for DSAN also contains `mmd_wt` = 2.0872. The DSAN code of AdaTime does not read it; it reads `domain_loss_wt`. We use 1.59.
2. **No rising weight.** In the code of the DSAN authors (<https://github.com/easezyc/deep-transfer-learning>, `UDA/pytorch1.0/DSAN/main.py`, checked on 2026-11-10) the LMMD term is multiplied by a factor that rises from 0 towards 1 during training, and the network is trained with SGD. AdaTime uses a constant weight and Adam. Our DSAN is "DSAN as implemented in AdaTime". TODO (verify): whether the paper text describes the same rising factor as the authors' code; only the code was checked.
3. **NaN guard.** As in AdaTime, the loss of a step is a constant 0 if the kernel matrix contains NaN. Our loop counts such steps; there was none in any run.

## 4. Uniform soup of the final checkpoints (stage 3)

**Idea.** A model soup averages the weights of several models that were fine-tuned from the same start model, and evaluates the averaged model like a single model (Wortsman et al.). Here the ingredients are the three branches of one transfer and seed: Deep CORAL, DANN and DSAN after their last epoch, all continued from the same source-only checkpoint. "Uniform" means every ingredient has weight 1/3; "at the end" means the branches are trained independently to the end and averaged once.

$$\theta_{soup} = \tfrac{1}{3}\left(\theta_{CORAL} + \theta_{DANN} + \theta_{DSAN}\right)$$
ight)$$

| | |
|---|---|
| What is averaged | every entry of the HAR predictor's `state_dict` (convolutions, batch-normalisation scale and shift, classifier) and the batch-normalisation running statistics; not the DANN discriminator |
| Precondition | all three ingredients started from the same source-only checkpoint (checked by SHA-256) |
| Soup of the controls | the same average of the three control runs; it shows what the soup gains without any alignment loss |
| Batch normalisation | `bn_avg`: the averaged running statistics. `bn_target`: after averaging, the running statistics are reset and re-estimated in one pass over the unlabelled target-train windows (cumulative average over batches of 32, `torch.optim.swa_utils.update_bn`); the weights are not changed |
| Output ensemble | as a comparison, the mean of the softmax outputs of the three ingredients; three forward passes instead of one |
| Code | averaging: `create_method_soup.py`, written by the project partner Denis Petrov; extended for the evaluation by `average_state_dicts` (keeps the integer counter `num_batches_tracked`) and by `run_soup.py` (checks, batch-normalisation variants, ensemble, predictions, `config.json`) |

**Why two batch-normalisation variants.** Averaging the running mean and variance of three models does not give the statistics of the averaged model: the averaged weights produce different activations than any ingredient. Re-estimating the statistics after averaging is common practice for weight averaging (for example in stochastic weight averaging). Doing it on target-train windows uses only unlabelled target data, like the target forward pass of the training. Which of the two variants is better is part of the results, not a choice made beforehand.

## Why these three methods

The plan from the project kickoff is to average branches that were trained with different UDA methods. We chose one method from each of three families, so that the branches differ in how they use the target data:

| Branch | Method | Family | Alignment |
|---|---|---|---|
| 1 | Deep CORAL | discrepancy-based | global, covariance of the features |
| 2 | DANN | adversarial | global, domain discriminator |
| 3 | DSAN | discrepancy-based | local, per class with soft pseudo-labels |

- **Different mechanisms.** A statistic-matching loss, an adversarial game and a class-wise loss are three different ways to use the unlabelled target data. TODO (verify): the task text assigns this grouping to kickoff slides 5 and 7; the slides were not at hand when this page was written, so the wording of the slides is not quoted here.
- **Compatible with a soup.** A model soup averages the weights of multiple models fine-tuned with different hyper-parameter configurations (Wortsman et al.); here the branches differ in the UDA method instead. All three methods keep the architecture of the HAR predictor unchanged and start from the same source-only model, and only DANN has an extra head, which is left out of the average. TODO (verify): that more diverse ingredients make a better soup is our working assumption, not a result we checked in the model-soups paper.
- **Class-wise alignment in the benchmark.** AdaTime groups DSAN with the methods that align the joint (marginal and conditional) distribution, and Deep CORAL and DANN with those that align only the marginal distribution. In its discussion the paper reports that methods with joint alignment tend to perform consistently better on its datasets (section 4.3 of the arXiv version, read in the ar5iv rendering). This was a reason to include one class-wise method.
- **Available reference code.** All three are implemented in AdaTime for time series, which gives a tested loss implementation and hyper-parameters for a HAR dataset.

## References

1. **Deep CORAL.** Baochen Sun and Kate Saenko. "Deep CORAL: Correlation Alignment for Deep Domain Adaptation." 2016. arXiv:1607.01719. Title, authors and year checked against arXiv. TODO (verify): the published venue (to our knowledge the ECCV 2016 Workshops, Springer LNCS) with volume, pages and DOI could not be checked against the publisher's page.
2. **DANN.** Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand and Victor Lempitsky. "Domain-Adversarial Training of Neural Networks." Journal of Machine Learning Research, vol. 17, pp. 1–35, 2016. arXiv:1505.07818. Checked against arXiv (journal reference as given there).
3. **DSAN.** Yongchun Zhu, Fuzhen Zhuang, Jindong Wang, Guolin Ke, Jingwu Chen, Jiang Bian, Hui Xiong and Qing He. "Deep Subdomain Adaptation Network for Image Classification." IEEE Transactions on Neural Networks and Learning Systems, vol. 32, no. 4, pp. 1713–1722, 2021. DOI 10.1109/TNNLS.2020.2988928. arXiv:2106.09388. Checked against arXiv and the Crossref record of the DOI.
4. **AdaTime.** Mohamed Ragab, Emadeldeen Eldele, Wee Ling Tan, Chuan-Sheng Foo, Zhenghua Chen, Min Wu, Chee-Keong Kwoh and Xiaoli Li. "ADATIME: A Benchmarking Suite for Domain Adaptation on Time Series Data." ACM Transactions on Knowledge Discovery from Data, vol. 17, no. 8, 2023. DOI 10.1145/3587937. arXiv:2203.08321. Checked against arXiv and the Crossref record of the DOI. TODO (verify): the article number in the journal issue (Crossref lists only pages 1–18).
5. **Model soups.** Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith and Ludwig Schmidt. "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time." Proceedings of the 39th International Conference on Machine Learning, PMLR 162:23965–23998, 2022. arXiv:2203.05482. Checked against arXiv and the PMLR page.

Code: AdaTime, <https://github.com/emadeldeen24/AdaTime>, commit `2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d`.
