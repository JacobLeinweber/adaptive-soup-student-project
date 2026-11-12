"""Uniform soup of the final checkpoints: ingredients, BatchNorm re-estimation and label-free outputs.

The averaging itself is create_method_soup.average_state_dicts (the soup script of the project partner).
Nothing in this module reads labels: the BatchNorm statistics are re-estimated from target-train windows only.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.optim.swa_utils import update_bn

from src.training.model import SelfHARCNN

# Ingredients of every soup, in this order (one run each per transfer and seed, all from the same start checkpoint).
SOUPS = {
    "soup_methods": ("deep_coral", "dann", "dsan"),
    "soup_controls": ("lambda0_target_pass", "dann_lambda0_target_pass", "dsan_lambda0_target_pass"),
}
# The output ensemble that belongs to each soup: same ingredients, averaged softmax outputs instead of weights.
ENSEMBLES = {"soup_methods": "methods", "soup_controls": "controls"}
# bn_avg: BatchNorm statistics averaged like the weights. bn_target: re-estimated on target train after averaging.
BN_VARIANTS = ("bn_avg", "bn_target")
# Batch size of the BatchNorm re-estimation: the batch size of the target forward pass in training.
BN_BATCH_SIZE = 32


def logits(model: SelfHARCNN, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """Class logits in eval mode (BatchNorm running statistics, no dropout), without gradients; float32."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs)


def softmax(values: np.ndarray) -> np.ndarray:
    """Row-wise softmax in float64."""
    shifted = np.asarray(values, dtype=np.float64) - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def recompute_bn(model: SelfHARCNN, x_target_train: np.ndarray, seed: int, device: torch.device,
                 batch_size: int = BN_BATCH_SIZE) -> int:
    """Re-estimate the BatchNorm running statistics on target-train windows; weights stay unchanged.

    torch.optim.swa_utils.update_bn resets running_mean, running_var and num_batches_tracked, sets the
    momentum to None (cumulative average over the batches), runs one pass in train mode without gradients
    and restores the momentum and the mode. The windows are shuffled with a generator from `seed`, because
    the prepared files are ordered by subject and activity and batch statistics of ordered batches would
    underestimate the variance. Dropout is active in train mode, as in the forward passes of training;
    its random state is drawn from `seed` too. Returns the number of batches.
    """
    order = torch.randperm(len(x_target_train), generator=torch.Generator().manual_seed(seed))
    x = torch.as_tensor(x_target_train, dtype=torch.float32)
    batches = [x[index] for index in order.split(batch_size)]
    model.to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        update_bn(batches, model, device=device)
    model.eval()
    return len(batches)
