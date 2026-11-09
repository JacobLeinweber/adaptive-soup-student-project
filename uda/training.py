"""Training loop that continues from a supplied source-only checkpoint.

One step = one labeled source batch and, for Deep CORAL, one unlabeled target batch of the
same size. Source and target go through the feature extractor in two separate forward passes
in train mode, source first, as in AdaTime's Deep_Coral.training_epoch
(algorithms/algorithms.py, lines 188-205, commit 2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d).
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
from .losses import coral_loss


class CyclicTargetSampler:
    """Endless stream of target-train indices.

    Every pass over the target windows is a fresh permutation drawn from an own generator.
    take(k) returns the next k indices; a batch may span the end of one pass and the start
    of the next one, so every window is used once per pass.
    """

    def __init__(self, n_windows: int, seed: int):
        self.n_windows = n_windows
        self.generator = torch.Generator().manual_seed(seed)
        self.pending = torch.empty(0, dtype=torch.long)
        self.passes = 0

    def take(self, k: int) -> torch.Tensor:
        while len(self.pending) < k:
            self.pending = torch.cat([self.pending, torch.randperm(self.n_windows, generator=self.generator)])
            self.passes += 1
        batch, self.pending = self.pending[:k], self.pending[k:]
        return batch


def train_continued(
    model: SelfHARCNN,
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray | None,
    *,
    coral_weight: float,
    source_ce_weight: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Continue training `model` in place and return the mean losses of every epoch.

    Loss per step: source_ce_weight * CE(source) + coral_weight * CORAL(source, target).
    For the target only the windows x_target are passed in; there is no parameter for
    target labels. x_target=None is the lambda = 0 control ("source only, trained further"):
    no target window goes through the model, so not even the BatchNorm statistics see target data.
    x_target together with coral_weight=0 keeps the target forward pass but adds no CORAL gradient,
    so target windows reach the model only through the BatchNorm statistics.
    """
    if x_target is None and coral_weight != 0:
        raise ValueError("coral_weight must be 0 when no target data is given")
    seed_everything(seed)
    model.to(device).train()
    xs = torch.as_tensor(x_source, dtype=torch.float32, device=device)
    ys = torch.as_tensor(y_source, dtype=torch.long, device=device)
    xt = None if x_target is None else torch.as_tensor(x_target, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Source order exactly as in the baseline: new permutation per epoch from a generator with the seed.
    source_generator = torch.Generator().manual_seed(seed)
    target_sampler = None if xt is None else CyclicTargetSampler(len(xt), seed)

    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        ce_sum = torch.zeros((), device=device)
        coral_sum = torch.zeros((), device=device)
        total_sum = torch.zeros((), device=device)
        steps = 0
        for order in torch.randperm(len(xs), generator=source_generator).split(batch_size):
            index = order.to(device)
            source_features = extract_features(model, xs[index])
            ce = F.cross_entropy(model.classifier(source_features), ys[index])
            loss = source_ce_weight * ce
            if target_sampler is not None:
                # Same batch size as the source batch: the scatter matrices in coral_loss are only
                # comparable for equal n (AdaTime avoids the issue with drop_last=True).
                target_index = target_sampler.take(len(order)).to(device)
                target_features = extract_features(model, xt[target_index])
                coral = coral_loss(source_features, target_features)
                loss = loss + coral_weight * coral
                coral_sum += coral.detach()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ce_sum += ce.detach()
            total_sum += loss.detach()
            steps += 1
        record = {
            "epoch": epoch,
            "source_ce": ce_sum.item() / steps,
            "coral": coral_sum.item() / steps if target_sampler is not None else math.nan,
            "total": total_sum.item() / steps,
            "steps": steps,
            "target_passes": target_sampler.passes if target_sampler is not None else 0,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        log(f"epoch {epoch:2d}  source_ce {record['source_ce']:.5f}  coral {record['coral']:.3e}  "
            f"total {record['total']:.5f}  ({record['seconds']:.1f}s)")
    return history
