"""Class-specific memory bank for task embeddings (Section 3.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch


def _ensure_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}")
    if tensor.ndim < 2:
        raise ValueError(
            "Task embeddings must have shape (..., m+1, C). "
            f"Received tensor with ndim={tensor.ndim}"
        )
    return tensor


@dataclass
class ClassMemoryBank:
    """
    Exponential moving average (EMA) memory for class-specific task embeddings.

    Each entry stores the contextualised task tokens `T_k ∈ ℝ^{(m+1)×C}` for a
    semantic class `k`. The update rule follows Section 3.3 of the IRIS paper:

        T_k ← α T_k + (1 - α) T̂_k

    where `α = 0.999` is the momentum parameter and `T̂_k` is the newly observed
    task embedding extracted from a support/reference pair during training.
    """

    momentum: float = 0.999
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    storage: MutableMapping[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("Momentum α must lie in [0, 1).")

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def __contains__(self, class_id: int) -> bool:
        return int(class_id) in self.storage

    def get(self, class_id: int) -> Optional[torch.Tensor]:
        tensor = self.storage.get(int(class_id))
        if tensor is None:
            return None
        return tensor.detach().clone()

    def items(self) -> Iterable[tuple[int, torch.Tensor]]:
        for key, tensor in self.storage.items():
            yield key, tensor.detach().clone()

    def clear(self) -> None:
        self.storage.clear()

    # ------------------------------------------------------------------ #
    # Update & retrieval helpers
    # ------------------------------------------------------------------ #
    def update(self, class_id: int, embedding: torch.Tensor) -> torch.Tensor:
        """
        Update a single class entry using EMA.

        Args:
            class_id: Integer class label identifier (background should be 0 and
                is typically excluded by the caller).
            embedding: Tensor of shape (m+1, C) containing the new task tokens.

        Returns:
            The updated embedding stored in the memory bank.
        """
        class_id = int(class_id)
        embedding = _ensure_tensor(embedding).detach()
        if embedding.ndim != 2:
            raise ValueError(
                "Embedding must have rank 2 with shape (m+1, C); "
                f"received {tuple(embedding.shape)}"
            )

        if self.device is None:
            self.device = embedding.device
        if self.dtype is None:
            self.dtype = embedding.dtype

        embedding = embedding.to(device=self.device, dtype=self.dtype)

        if class_id in self.storage:
            stored = self.storage[class_id]
            if stored.shape != embedding.shape:
                raise ValueError(
                    f"Shape mismatch for class {class_id}: "
                    f"existing {tuple(stored.shape)} vs new {tuple(embedding.shape)}"
                )
            updated = self.momentum * stored + (1.0 - self.momentum) * embedding
        else:
            updated = embedding

        self.storage[class_id] = updated
        return updated.detach().clone()

    def update_episode(
        self,
        task_embeddings: torch.Tensor,
        class_ids: Sequence[Sequence[int]] | Sequence[int],
    ) -> None:
        """
        Apply EMA updates for all classes observed in an episode.

        Args:
            task_embeddings: Tensor of shape (B, K, m+1, C) or (K, m+1, C).
            class_ids: Either a flattened list of class IDs (len=K) or a nested
                sequence with outer length B specifying class IDs per batch item.
        """
        embeddings = _ensure_tensor(task_embeddings).detach()
        if embeddings.ndim == 3:  # (K, m+1, C)
            embeddings = embeddings.unsqueeze(0)

        if isinstance(class_ids[0], (list, tuple)):
            flat_class_ids: List[List[int]] = [
                [int(cid) for cid in sample_ids] for sample_ids in class_ids  # type: ignore[index]
            ]
        else:
            flat_class_ids = [list(map(int, class_ids))]  # type: ignore[arg-type]

        if embeddings.shape[0] != len(flat_class_ids):
            raise ValueError(
                "Mismatch between batch size of task embeddings and class_ids."
            )

        for sample_embeddings, sample_classes in zip(embeddings, flat_class_ids):
            if len(sample_classes) != sample_embeddings.shape[0]:
                raise ValueError(
                    "Number of class IDs per sample must match embedding count."
                )
            for class_id, class_embedding in zip(sample_classes, sample_embeddings):
                if int(class_id) == 0:
                    continue  # Background is not stored.
                self.update(int(class_id), class_embedding)

    def retrieve(
        self,
        class_ids: Sequence[int],
        *,
        default: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Fetch task embeddings for a list of classes.

        Args:
            class_ids: Iterable of class identifiers.
            default: Optional tensor to use when a class is missing. The tensor
                must broadcast to shape (m+1, C).
            device: Target device for the returned tensor.
            dtype: Target dtype for the returned tensor.

        Returns:
            Tensor of shape (len(class_ids), m+1, C).
        """
        retrieved: List[torch.Tensor] = []
        for class_id in class_ids:
            tensor = self.storage.get(int(class_id))
            if tensor is None:
                if default is None:
                    raise KeyError(
                        f"Class {class_id} is not present in the memory bank."
                    )
                tensor = default
            retrieved.append(tensor.detach())

        batch = torch.stack(retrieved, dim=0)
        if device is not None or dtype is not None:
            batch = batch.to(device=device or batch.device, dtype=dtype or batch.dtype)
        return batch

    # ------------------------------------------------------------------ #
    # Context ensemble utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def ensemble(embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Average multiple task embeddings to form a context ensemble.

        Args:
            embeddings: Sequence of tensors with identical shapes.

        Returns:
            A single tensor representing the averaged context.
        """
        if not embeddings:
            raise ValueError("Cannot build ensemble from an empty sequence.")

        stacked = torch.stack([_ensure_tensor(t).detach() for t in embeddings], dim=0)
        return stacked.mean(dim=0)

    def summary(self) -> Mapping[int, torch.Size]:
        """Return a lightweight summary of stored classes and tensor shapes."""
        return {class_id: tensor.shape for class_id, tensor in self.storage.items()}



