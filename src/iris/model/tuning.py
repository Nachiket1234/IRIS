"""In-context tuning utilities for task embeddings (Section 3.3)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import IrisModel
from .memory import ClassMemoryBank


@contextmanager
def _frozen_parameters(module: nn.Module, *, requires_grad: bool = False):
    original = [param.requires_grad for param in module.parameters()]
    try:
        for param in module.parameters():
            param.requires_grad_(requires_grad)
        yield
    finally:
        for param, flag in zip(module.parameters(), original):
            param.requires_grad_(flag)


class DiceCrossEntropyLoss(nn.Module):
    """Combine Dice and BCE losses for volumetric multi-class segmentation."""

    def __init__(self, *, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        probs = torch.sigmoid(logits)

        dims = tuple(range(2, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        
        # Clamp to prevent numerical instability
        intersection = torch.clamp(intersection, min=0)
        denom = torch.clamp(denom, min=self.smooth)
        
        dice_per_class = 1.0 - (2.0 * intersection + self.smooth) / (denom + self.smooth)
        
        # Clamp dice to valid range [0, 1]
        dice_per_class = torch.clamp(dice_per_class, min=0.0, max=1.0)

        if class_weights is not None:
            weights = class_weights.to(dice_per_class.dtype)
            while weights.ndim < dice_per_class.ndim:
                weights = weights.unsqueeze(-1)
            dice = (dice_per_class * weights).sum() / weights.sum()
        else:
            dice = dice_per_class.mean()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            weight=None if class_weights is None else class_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            reduction="mean",
        )
        
        # Ensure both losses are valid
        dice = torch.clamp(dice, min=0.0, max=1.0)
        total_loss = dice + bce
        
        return total_loss


@dataclass
class InContextTuner:
    """
    Optimise task embeddings while freezing the IRIS core parameters.

    The tuner follows Section 3.3 of the IRIS paper: we treat the task tokens
    as the only trainable variables and perform a lightweight adaptation loop on
    a held-out query volume.
    """

    model: IrisModel
    lr: float = 1e-3
    steps: int = 20
    loss_fn: nn.Module = DiceCrossEntropyLoss()
    memory_bank: Optional[ClassMemoryBank] = None

    def tune(
        self,
        query_images: torch.Tensor,
        query_masks: torch.Tensor,
        initial_embeddings: torch.Tensor,
        *,
        class_ids: Optional[Sequence[Sequence[int]] | Sequence[int]] = None,
        steps: Optional[int] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """
        Run in-context optimisation of the task embeddings.

        Args:
            query_images: Tensor of shape (B, C_in, D, H, W).
            query_masks: Binary tensor of shape (B, K, D, H, W).
            initial_embeddings: Tensor of shape (B, K, m+1, C).
            class_ids: Optional class identifiers corresponding to each K entry.
            steps: Number of optimisation iterations (defaults to self.steps).
            update_memory: Whether to insert the tuned embeddings into the memory
                bank (if available) using EMA updates.

        Returns:
            Tuned task embeddings detached from the computational graph.
        """
        steps = steps or self.steps
        task_embeddings = nn.Parameter(initial_embeddings.detach().clone())

        self.model.eval()
        optimiser = torch.optim.Adam([task_embeddings], lr=self.lr)

        with _frozen_parameters(self.model):
            for _ in range(steps):
                optimiser.zero_grad()
                outputs = self.model(
                    query_images,
                    task_embeddings,
                )
                logits = outputs["logits"]
                loss = self.loss_fn(logits, query_masks)
                loss.backward()
                optimiser.step()

        tuned = task_embeddings.detach()

        if update_memory and class_ids is not None:
            bank = self.memory_bank or getattr(self.model, "memory_bank", None)
            if bank is not None:
                bank.update_episode(tuned, class_ids)

        return tuned

    def initialise_from_memory(
        self,
        class_ids: Sequence[int],
        *,
        fallback: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Retrieve initial task embeddings from the memory bank for inference.
        """
        bank = self.memory_bank or getattr(self.model, "memory_bank", None)
        if bank is None:
            raise RuntimeError("No memory bank available for initialisation.")
        return bank.retrieve(
            class_ids,
            default=fallback,
            device=device,
            dtype=dtype,
        )



