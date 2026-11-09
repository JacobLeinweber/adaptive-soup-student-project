"""Feature access for SelfHARCNN without modifying src/training/model.py."""
from __future__ import annotations

import torch

from src.training.model import SelfHARCNN


def extract_features(model: SelfHARCNN, x: torch.Tensor) -> torch.Tensor:
    """Return the 96-dim pooled features that SelfHARCNN.forward passes to its classifier.

    x has shape (batch, 150, 18). The computation is the one in SelfHARCNN.forward up to and
    including the global average pooling, so model.classifier(extract_features(model, x))
    equals model(x).
    """
    return model.pool(model.features(x.transpose(1, 2))).flatten(1)
