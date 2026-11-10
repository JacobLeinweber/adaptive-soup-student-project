"""DSAN: class-wise alignment with the local maximum mean discrepancy (LMMD), ported from AdaTime.

Source: https://github.com/emadeldeen24/AdaTime
        commit 2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d (branch adatime_v2, 2023-06-07).
        LMMD: models/loss.py, lines 205-290 (class LMMD_loss);
        training step: algorithms/algorithms.py, lines 603-636 (class DSAN).

As for Deep CORAL and DANN, source and target go through the model in two separate forward passes
in train mode, source first, and an epoch has as many steps as there are source batches.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.training.model import SelfHARCNN
from src.training.utils import seed_everything
from .features import extract_features
from .training import CyclicTargetSampler

KERNEL_MUL = 2.0  # AdaTime LMMD_loss defaults (models/loss.py:206); DSAN passes no other values
KERNEL_NUM = 5


def gaussian_kernel(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum of KERNEL_NUM Gaussian kernels over all pairs of the stacked batch, shape (ns + nt, ns + nt).

    The base bandwidth is the mean squared distance over all pairs of different windows. It is
    treated as a constant (no gradient). The kernels use the bandwidths base / 4, / 2, x 1, x 2, x 4.
    """
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)
    l2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
    bandwidth = l2_distance.detach().sum() / (n_samples ** 2 - n_samples)
    bandwidth = bandwidth / KERNEL_MUL ** (KERNEL_NUM // 2)
    return sum(torch.exp(-l2_distance / (bandwidth * KERNEL_MUL ** i)) for i in range(KERNEL_NUM))


def class_weights(source_labels: torch.Tensor, target_probabilities: torch.Tensor) -> tuple:
    """Pair weights (source-source, target-target, source-target) and the number of shared classes.

    Source windows are weighted by their one-hot label, target windows by their predicted class
    probabilities (soft pseudo-labels), each divided by the class total of its batch. Only classes
    count that occur among the source labels and among the hard target predictions of the batch;
    the weights are divided by the number of these shared classes. No gradient flows through the
    pseudo-labels.
    """
    num_classes = target_probabilities.size(1)
    source_vec = F.one_hot(source_labels, num_classes).to(target_probabilities.dtype)
    target_vec = target_probabilities.detach()
    target_hard = F.one_hot(target_vec.argmax(1), num_classes).sum(0)
    source_sum = source_vec.sum(0, keepdim=True)
    target_sum = target_vec.sum(0, keepdim=True)
    source_vec = source_vec / torch.where(source_sum == 0, torch.full_like(source_sum, 100.0), source_sum)
    target_vec = target_vec / torch.where(target_sum == 0, torch.full_like(target_sum, 100.0), target_sum)

    shared = ((source_sum.squeeze(0) > 0) & (target_hard > 0)).to(source_vec.dtype)
    source_vec = source_vec * shared
    target_vec = target_vec * shared
    n_shared = shared.sum()
    scale = n_shared.clamp(min=1)  # without a shared class all weights are already 0
    return (source_vec @ source_vec.t() / scale, target_vec @ target_vec.t() / scale,
            source_vec @ target_vec.t() / scale, n_shared)


def lmmd_loss(source: torch.Tensor, target: torch.Tensor, source_labels: torch.Tensor,
              target_probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """LMMD between a source and a target feature batch; also the number of shared classes and a NaN flag.

    As in AdaTime, the loss is a constant 0 (no gradient) if the kernel matrix contains NaN; the
    flag reports this case so that it does not go unnoticed.
    """
    n = source.size(0)
    weight_ss, weight_tt, weight_st, n_shared = class_weights(source_labels, target_probabilities)
    kernels = gaussian_kernel(source, target)
    if torch.isnan(kernels).any():
        return torch.zeros((), device=source.device), n_shared, True
    loss = torch.sum(weight_ss * kernels[:n, :n] + weight_tt * kernels[n:, n:] - 2 * weight_st * kernels[:n, n:])
    return loss, n_shared, False


def train_dsan(
    model: SelfHARCNN,
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    *,
    lmmd_weight: float,
    source_ce_weight: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Continue training `model` in place and return the log of every epoch.

    Loss per step: source_ce_weight * CE(source) + lmmd_weight * LMMD(source, target).
    The pseudo-labels of the target batch are the softmax outputs of the model itself in this step.
    lmmd_weight=0 is the control: the same loop and the same target forward pass, but the LMMD is
    only computed for the log (without gradient), so target windows reach the model only through
    the BatchNorm statistics.
    For the target only the windows x_target are passed in; there is no parameter for target labels.
    """
    seed_everything(seed)
    model.to(device).train()
    xs = torch.as_tensor(x_source, dtype=torch.float32, device=device)
    ys = torch.as_tensor(y_source, dtype=torch.long, device=device)
    xt = torch.as_tensor(x_target, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Source order exactly as in the baseline: new permutation per epoch from a generator with the seed.
    source_generator = torch.Generator().manual_seed(seed)
    target_sampler = CyclicTargetSampler(len(xt), seed)

    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        ce_sum = torch.zeros((), device=device)
        lmmd_sum = torch.zeros((), device=device)
        total_sum = torch.zeros((), device=device)
        confidence_sum = torch.zeros((), device=device)
        shared_sum = torch.zeros((), device=device)
        nan_steps = 0
        steps = 0
        windows = 0
        for order in torch.randperm(len(xs), generator=source_generator).split(batch_size):
            index = order.to(device)
            source_features = extract_features(model, xs[index])
            ce = F.cross_entropy(model.classifier(source_features), ys[index])
            # Same batch size as the source batch, as for Deep CORAL and DANN.
            target_index = target_sampler.take(len(order)).to(device)
            target_features = extract_features(model, xt[target_index])
            target_probabilities = F.softmax(model.classifier(target_features), dim=1)
            loss = source_ce_weight * ce
            if lmmd_weight != 0:
                lmmd, n_shared, kernel_nan = lmmd_loss(source_features, target_features, ys[index], target_probabilities)
                loss = loss + lmmd_weight * lmmd
            else:
                with torch.no_grad():
                    lmmd, n_shared, kernel_nan = lmmd_loss(source_features, target_features, ys[index], target_probabilities)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ce_sum += ce.detach()
            lmmd_sum += lmmd.detach()
            total_sum += loss.detach()
            confidence_sum += target_probabilities.detach().max(1).values.sum()
            shared_sum += n_shared
            nan_steps += int(kernel_nan)
            steps += 1
            windows += len(order)
        record = {
            "epoch": epoch,
            "source_ce": ce_sum.item() / steps,
            "lmmd": lmmd_sum.item() / steps,
            "pseudo_label_confidence": confidence_sum.item() / windows,  # mean highest class probability
            "shared_classes": shared_sum.item() / steps,
            "lmmd_nan_steps": nan_steps,
            "total": total_sum.item() / steps,
            "steps": steps,
            "target_passes": target_sampler.passes,
            "seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(record[key]) for key in ("source_ce", "lmmd", "total")):
            log(f"warning: non-finite loss in epoch {epoch}")
        history.append(record)
        log(f"epoch {epoch:2d}  source_ce {record['source_ce']:.5f}  lmmd {record['lmmd']:.5f}  "
            f"confidence {record['pseudo_label_confidence']:.3f}  shared {record['shared_classes']:.2f}  "
            f"nan_steps {nan_steps}  total {record['total']:.5f}  ({record['seconds']:.1f}s)")
    return history
