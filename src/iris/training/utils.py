"""Utility helpers for reproducible medical training workflows."""

from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger("iris.training")


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def compute_class_weights(mask_batch: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute inverse-frequency weights to mitigate class imbalance.

    Args:
        mask_batch: Tensor of shape (B, K, D, H, W) with binary labels.
        eps: Numerical stability constant.

    Returns:
        Tensor of shape (B, K) representing per-class weights.
    """
    flattened = mask_batch.view(mask_batch.shape[0], mask_batch.shape[1], -1)
    voxel_counts = flattened.sum(dim=-1)
    inv = 1.0 / torch.clamp(voxel_counts, min=eps)
    inv[voxel_counts == 0] = 0.0
    norm = inv.sum(dim=1, keepdim=True).clamp_min(eps)
    weights = inv / norm
    return weights


def parameter_count(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def describe_datasets(datasets: Sequence[torch.utils.data.Dataset]) -> List[str]:
    descriptions = []
    for dataset in datasets:
        size = len(dataset)
        name = getattr(dataset, "dataset_name", dataset.__class__.__name__)
        split = getattr(dataset, "split", None)
        split_name = getattr(split, "value", split) if split is not None else "unknown"
        descriptions.append(f"{name} ({split_name}): {size} volumes")
    return descriptions


