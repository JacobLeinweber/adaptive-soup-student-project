"""Label-free selection criteria, ported from AdaTime: source risk (SRC) and deep embedded validation (DEV).

Neither criterion reads target labels. SRC uses the labelled source validation split. DEV uses the
source validation split and, as unlabelled data, the windows of source train and target train.

Source: https://github.com/emadeldeen24/AdaTime
        commit 2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d (branch adatime_v2, 2023-06-07).
        SRC: trainers/abstract_trainer.py:104-133 (evaluate) and 244-255 (calculate_risks):
             mean cross-entropy in eval mode, computed there on the source *test* split.
        DEV: utils.py:149-161 (get_dev_value), 164-177 (simple_MLP), 180-224 (get_weight_gpu),
             227-238 (calc_dev_risk). At this commit calc_dev_risk is defined but never called.

The domain classifier of DEV is a skorch NeuralNetClassifier in AdaTime. skorch is not part of this
project, so the classifier is rebuilt in plain PyTorch with skorch's defaults as used there.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

from src.training.model import SelfHARCNN
from .evaluation import compute_metrics
from .features import extract_features

# AdaTime get_weight_gpu: learning rates tried for the domain classifier, its epochs and the share used for training.
DEV_LEARNING_RATES = (1e-1, 5e-2, 1e-2)
DEV_EPOCHS = 30
DEV_TRAIN_SIZE = 0.8
# skorch defaults of NeuralNetClassifier, which AdaTime does not override.
SKORCH_BATCH_SIZE = 128
SKORCH_VALID_FOLDS = 5  # train_split=ValidSplit(5, stratified=True): one fifth is held out and not trained on


def features_and_logits(model: SelfHARCNN, x: np.ndarray, device: torch.device,
                        batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """96-dim features and class logits in eval mode (BatchNorm running statistics, no dropout), no gradients."""
    model.eval()
    features, logits = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            f = extract_features(model, batch)
            features.append(f.cpu().numpy())
            logits.append(model.classifier(f).cpu().numpy())
    return np.concatenate(features), np.concatenate(logits)


def per_window_ce(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cross-entropy of every window, in float64."""
    return F.cross_entropy(torch.as_tensor(logits, dtype=torch.float64), torch.as_tensor(labels, dtype=torch.long),
                           reduction="none").numpy()


def source_risk(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    """SRC on the source validation split: mean cross-entropy over the windows and the usual metrics.

    AdaTime averages the batch means of the cross-entropy; here it is the mean over all windows, so
    that the value does not depend on the batch size.
    """
    return {"ce": float(per_window_ce(logits, labels).mean()),
            **compute_metrics(labels, logits.argmax(1), num_classes)}


class DomainMLP(nn.Module):
    """AdaTime simple_MLP: d -> d // 2 -> 2 with ReLU, softmax output (class 1 = source, 0 = target)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.dense0 = nn.Linear(input_dim, input_dim // 2)
        self.nonlin = nn.ReLU()
        self.output = nn.Linear(input_dim // 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.output(self.nonlin(self.dense0(x))), dim=-1)


def fit_domain_classifier(x: np.ndarray, y: np.ndarray, lr: float) -> DomainMLP:
    """Train DomainMLP as skorch's NeuralNetClassifier does with AdaTime's settings.

    skorch defaults: one fifth is split off with a stratified 5-fold split (first fold, no shuffling)
    and only used for skorch's own validation loss, so it is not trained on; SGD without momentum;
    batches of 128, reshuffled every epoch (iterator_train__shuffle=True); NLL loss on the log of the
    softmax output plus machine epsilon. 30 epochs, no early stopping (callbacks="disable").
    The global torch generator drives the initialisation and the shuffling, as in skorch.
    """
    train_index, _ = next(StratifiedKFold(SKORCH_VALID_FOLDS).split(x, y))
    xs = torch.as_tensor(x[train_index], dtype=torch.float32)
    ys = torch.as_tensor(y[train_index], dtype=torch.long)
    classifier = DomainMLP(x.shape[1])
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr)
    eps = torch.finfo(torch.float32).eps
    classifier.train()
    for _ in range(DEV_EPOCHS):
        for batch in torch.randperm(len(xs)).split(SKORCH_BATCH_SIZE):
            loss = F.nll_loss(torch.log(classifier(xs[batch]) + eps), ys[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return classifier.eval()


def predict_proba(classifier: DomainMLP, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return classifier(torch.as_tensor(x, dtype=torch.float32)).numpy().astype(np.float64)


def dev_weights(source_features: np.ndarray, target_features: np.ndarray, validation_features: np.ndarray,
                seed: int) -> tuple[np.ndarray, dict]:
    """Importance weights p_target(x) / p_source(x) of the source validation windows (AdaTime get_weight_gpu).

    A domain classifier separates source-train (label 1) from target-train features (label 0). Of the
    three learning rates the one with the highest accuracy on a random 20 % hold-out is kept, and the
    weight of a validation window is P(target | x) / P(source | x) * N_source / N_target.
    The split, the initialisation and the shuffling are drawn from `seed` (AdaTime leaves them unseeded).
    """
    n_source, n_target = len(source_features), len(target_features)
    x = np.concatenate([source_features, target_features])
    y = np.concatenate([np.ones(n_source, dtype=np.int64), np.zeros(n_target, dtype=np.int64)])
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=DEV_TRAIN_SIZE, random_state=seed)
    accuracies, classifiers = [], []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for lr in DEV_LEARNING_RATES:
            classifier = fit_domain_classifier(x_train, y_train, lr)
            accuracies.append(float((predict_proba(classifier, x_test).argmax(1) == y_test).mean()))
            classifiers.append(classifier)
    best = accuracies.index(max(accuracies))
    probabilities = predict_proba(classifiers[best], validation_features)
    weights = probabilities[:, 0] / probabilities[:, 1] * n_source / n_target
    return weights, {"dev_lr": DEV_LEARNING_RATES[best], "dev_domain_accuracy": accuracies[best],
                     "dev_domain_accuracies": accuracies}


def dev_value(weights: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    """DEV risk with the control variate of You et al. (2019), as AdaTime get_dev_value; also eta."""
    weights = np.asarray(weights, dtype=np.float64)
    weighted_error = weights * np.asarray(errors, dtype=np.float64)
    cov = np.cov(np.stack([weighted_error, weights], axis=1), rowvar=False)[0][1]
    eta = -cov / np.var(weights, ddof=1)
    return float(np.mean(weighted_error) + eta * np.mean(weights) - eta), float(eta)


def dev_risk(source_features: np.ndarray, target_features: np.ndarray, validation_features: np.ndarray,
             validation_errors: np.ndarray, seed: int) -> dict:
    """DEV risk of a model from its features and the cross-entropy of every source validation window."""
    weights, details = dev_weights(source_features, target_features, validation_features, seed)
    if not np.isfinite(weights).all():
        return {"dev_risk": float("nan"), "dev_eta": float("nan"), **details,
                "dev_weight_mean": float("nan"), "dev_weight_max": float("nan"), "dev_weights_finite": False}
    risk, eta = dev_value(weights, validation_errors)
    return {"dev_risk": risk, "dev_eta": eta, **details, "dev_weight_mean": float(weights.mean()),
            "dev_weight_max": float(weights.max()), "dev_weights_finite": True}
