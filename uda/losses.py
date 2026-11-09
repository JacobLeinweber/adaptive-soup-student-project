"""Deep CORAL loss, ported from AdaTime without changing the arithmetic.

Source: https://github.com/emadeldeen24/AdaTime
        commit 2d9be6bd0542e150518afaf8aa0c6ea0823b4a3d (branch adatime_v2, 2023-06-07),
        file models/loss.py, class CORAL, lines 109-127.
"""
from __future__ import annotations

import torch


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CORAL distance between a source and a target feature batch, both of shape (n, d).

    As in AdaTime, the two matrices are centred scatter matrices Xc^T Xc (not divided by n - 1),
    and the squared Frobenius norm of their difference is averaged over the d * d entries
    before the division by 4 d^2. Relative to the paper formula 1/(4 d^2) * ||C_S - C_T||_F^2
    with C = Xc^T Xc / (n - 1), this is the paper value times (n - 1)^2 / d^2.
    The scatter matrices only have the same scale if both batches have the same size n,
    so callers must pass equally sized batches.
    """
    d = source.size(1)

    # source covariance (scatter matrix)
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance (scatter matrix)
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # Frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    return loss
