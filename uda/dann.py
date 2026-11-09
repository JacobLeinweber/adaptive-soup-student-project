"""DANN: adversarial domain adaptation with a gradient reversal layer, ported from AdaTime.

Source: https://github.com/emadeldeen24/AdaTime
        commit 2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d (branch adatime_v2, 2023-06-07).
        Discriminator: models/models.py, lines 229-248; gradient reversal: lines 252-261;
        training step: algorithms/algorithms.py, lines 327-364 (class DANN).

As for Deep CORAL, source and target go through the feature extractor in two separate forward
passes in train mode, source first, and an epoch has as many steps as there are source batches.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from src.training.model import SelfHARCNN
from src.training.utils import seed_everything
from .features import extract_features
from .training import CyclicTargetSampler

DISCRIMINATOR_HIDDEN = 64          # AdaTime disc_hid_dim for HAR (configs/data_model_configs.py:41)
DISCRIMINATOR_BETAS = (0.5, 0.99)  # AdaTime, discriminator optimizer only (algorithms/algorithms.py:310)


class GradientReversal(Function):
    """Identity on the forward pass; multiplies the gradient by -alpha on the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output.neg() * ctx.alpha, None


class DomainDiscriminator(nn.Module):
    """Two hidden layers and two outputs (domain 1 = source, 0 = target), no BatchNorm, no dropout."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, DISCRIMINATOR_HIDDEN), nn.ReLU(),
            nn.Linear(DISCRIMINATOR_HIDDEN, DISCRIMINATOR_HIDDEN), nn.ReLU(),
            nn.Linear(DISCRIMINATOR_HIDDEN, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layer(features)


def reversal_factor(step: int, epoch: int, num_batches: int, epochs: int) -> float:
    """Gradient reversal factor alpha, literally as in AdaTime (algorithms/algorithms.py:327-328).

    step counts from 0, epoch from 1. Unlike in the DANN paper, p is not the fraction of training
    done so far: it is divided by the number of epochs only, and 1 / num_batches is added.
    With the batch counts of this project alpha is practically 1 from the first step on.
    """
    p = float(step + epoch * num_batches) / epochs + 1 / num_batches
    return 2.0 / (1.0 + math.exp(-10 * p)) - 1


def train_dann(
    model: SelfHARCNN,
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    *,
    domain_weight: float,
    source_ce_weight: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> tuple[list[dict], DomainDiscriminator | None]:
    """Continue training `model` in place; return the log of every epoch and the discriminator.

    Loss per step: source_ce_weight * CE(source) + domain_weight * (domain CE source + domain CE target).
    domain_weight=0 is the control: the same loop without discriminator and domain loss, so target
    windows reach the model only through the BatchNorm statistics, and the returned discriminator is None.
    For the target only the windows x_target are passed in; there is no parameter for target labels.
    """
    seed_everything(seed)
    model.to(device).train()
    xs = torch.as_tensor(x_source, dtype=torch.float32, device=device)
    ys = torch.as_tensor(y_source, dtype=torch.long, device=device)
    xt = torch.as_tensor(x_target, dtype=torch.float32, device=device)
    optimizers = [torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)]
    discriminator = None
    if domain_weight != 0:
        # Built on the CPU: its initialization draws from the CPU generator of the run's seed and
        # does not advance the MPS generator that drives dropout.
        discriminator = DomainDiscriminator(model.classifier[0].in_features).to(device).train()
        optimizers.append(torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=weight_decay,
                                           betas=DISCRIMINATOR_BETAS))
    # Source order exactly as in the baseline: new permutation per epoch from a generator with the seed.
    source_generator = torch.Generator().manual_seed(seed)
    target_sampler = CyclicTargetSampler(len(xt), seed)
    # AdaTime uses max(len(src_loader), len(trg_loader)) with drop_last=True, although its DANN epoch
    # runs over the source batches only.
    num_batches = max(len(xs), len(xt)) // batch_size

    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        ce_sum = torch.zeros((), device=device)
        domain_sum = torch.zeros((), device=device)
        total_sum = torch.zeros((), device=device)
        correct_source = torch.zeros((), device=device)
        correct_target = torch.zeros((), device=device)
        alpha = math.nan
        steps = 0
        windows = 0
        for step, order in enumerate(torch.randperm(len(xs), generator=source_generator).split(batch_size)):
            index = order.to(device)
            source_features = extract_features(model, xs[index])
            ce = F.cross_entropy(model.classifier(source_features), ys[index])
            # Same batch size as the source batch, as for Deep CORAL.
            target_index = target_sampler.take(len(order)).to(device)
            target_features = extract_features(model, xt[target_index])
            loss = source_ce_weight * ce
            if discriminator is not None:
                alpha = reversal_factor(step, epoch, num_batches, epochs)
                source_domain = discriminator(GradientReversal.apply(source_features, alpha))
                target_domain = discriminator(GradientReversal.apply(target_features, alpha))
                domain_loss = (F.cross_entropy(source_domain, torch.ones(len(order), dtype=torch.long, device=device))
                               + F.cross_entropy(target_domain, torch.zeros(len(order), dtype=torch.long, device=device)))
                loss = loss + domain_weight * domain_loss
                domain_sum += domain_loss.detach()
                correct_source += (source_domain.argmax(1) == 1).sum()
                correct_target += (target_domain.argmax(1) == 0).sum()
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            ce_sum += ce.detach()
            total_sum += loss.detach()
            steps += 1
            windows += len(order)
        adversarial = discriminator is not None
        record = {
            "epoch": epoch,
            "source_ce": ce_sum.item() / steps,
            "domain_loss": domain_sum.item() / steps if adversarial else math.nan,  # sum over both domains
            "disc_acc_source": correct_source.item() / windows if adversarial else math.nan,
            "disc_acc_target": correct_target.item() / windows if adversarial else math.nan,
            "grl_alpha": alpha,  # value in the last step of the epoch
            "total": total_sum.item() / steps,
            "steps": steps,
            "target_passes": target_sampler.passes,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        log(f"epoch {epoch:2d}  source_ce {record['source_ce']:.5f}  domain_loss {record['domain_loss']:.4f}  "
            f"disc_acc s {record['disc_acc_source']:.3f} t {record['disc_acc_target']:.3f}  "
            f"alpha {record['grl_alpha']:.6f}  total {record['total']:.5f}  ({record['seconds']:.1f}s)")
    return history, discriminator
