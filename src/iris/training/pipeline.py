"""Episodic training loop mirroring the IRIS paper (Section 3.2.3)."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from iris.data.augmentations import MedicalAugmentation
from iris.model import IrisModel
from iris.model.tuning import DiceCrossEntropyLoss
from iris.training.lamb import Lamb
from iris.training.utils import (
    LOGGER,
    compute_class_weights,
    ensure_directory,
    set_global_seed,
)


@dataclass
class EpisodicTrainingConfig:
    """Configuration parameters for episodic IRIS training."""

    base_learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-6
    total_iterations: int = 80_000
    warmup_iterations: int = 2_000
    batch_size: int = 32
    lr_decay_gamma: float = 0.98
    decay_interval: int = 5_000
    gradient_clip_norm: float = 1.0
    random_seed: int = 42
    device: Optional[str] = None
    log_every: int = 50
    eval_every: int = 2_000
    checkpoint_every: int = 5_000
    checkpoint_dir: str | Path = "checkpoints"
    volume_size: Tuple[int, int, int] = (128, 128, 128)
    query_noise_std: float = 0.05
    random_class_drop_prob: float = 0.15
    augmentation_kwargs: Dict[str, float] = field(
        default_factory=lambda: {
            "crop_size": (112, 112, 112),
            "intensity_shift": 0.1,
            "intensity_scale": 0.2,
            "rotation_range": (10.0, 10.0, 10.0),
            "translation_range": (8.0, 8.0, 8.0),
        }
    )


def _ensure_tensor_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Unsupported tensor shape for volumetric data: {tuple(tensor.shape)}")


class EpisodicTrainer:
    """
    Simulates in-context learning via episodic training across heterogeneous datasets.
    """

    def __init__(
        self,
        model: IrisModel,
        datasets: Sequence[torch.utils.data.Dataset],
        config: EpisodicTrainingConfig,
        *,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required for episodic training.")

        self.model = model
        self.datasets = list(datasets)
        self.config = config
        set_global_seed(config.random_seed)
        self.logger = logger or LOGGER
        self.device = torch.device(
            device
            or config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.loss_fn = DiceCrossEntropyLoss()

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=config.base_learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
        )

        self.iteration = 0
        self.checkpoint_dir = ensure_directory(config.checkpoint_dir)

        self.support_augmentation = MedicalAugmentation(**config.augmentation_kwargs)
        self.query_augmentation = MedicalAugmentation(**config.augmentation_kwargs)

        self.rng = np.random.default_rng(config.random_seed)

        self.logger.info("Episodic trainer initialised on %s", self.device)
        for description in self._dataset_descriptions():
            self.logger.info("Dataset: %s", description)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train(
        self,
        *,
        evaluation_hook=None,
    ) -> None:
        self.model.train()
        start_time = time.time()

        for iteration in range(1, self.config.total_iterations + 1):
            self.iteration = iteration
            episode = self._sample_batch()
            loss_value = self._run_episode(episode)

            self._apply_schedule(iteration)

            if iteration % self.config.log_every == 0:
                elapsed = time.time() - start_time
                self.logger.info(
                    "Iter %d/%d | Loss %.4f | LR %.6f | Elapsed %.1fs",
                    iteration,
                    self.config.total_iterations,
                    loss_value,
                    self.optimizer.param_groups[0]["lr"],
                    elapsed,
                )

            if iteration % self.config.checkpoint_every == 0:
                self.save_checkpoint(iteration)

            if evaluation_hook and iteration % self.config.eval_every == 0:
                self.model.eval()
                try:
                    evaluation_hook(iteration, self.model)
                finally:
                    self.model.train()

    def save_checkpoint(self, iteration: Optional[int] = None) -> Path:
        iteration = iteration or self.iteration
        checkpoint_path = self.checkpoint_dir / f"iris_iter_{iteration:06d}.pt"
        state = {
            "iteration": iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "rng_state": self.rng.bit_generator.state,
            "config": self.config,
        }
        if self.model.memory_bank is not None:
            state["memory_bank"] = {
                cid: tensor.cpu() for cid, tensor in self.model.memory_bank.items()
            }
        torch.save(state, checkpoint_path)
        self.logger.info("Checkpoint saved to %s", checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.iteration = checkpoint["iteration"]
        if "rng_state" in checkpoint:
            self.rng.bit_generator.state = checkpoint["rng_state"]
        if self.model.memory_bank is not None and "memory_bank" in checkpoint:
            for cid, tensor in checkpoint["memory_bank"].items():
                self.model.memory_bank.update(int(cid), tensor.to(self.device))
        self.logger.info("Loaded checkpoint from %s (iteration %d)", checkpoint_path, self.iteration)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _dataset_descriptions(self) -> List[str]:
        descriptions = []
        for dataset in self.datasets:
            name = getattr(dataset, "dataset_name", dataset.__class__.__name__)
            split = getattr(dataset, "split", None)
            split_name = getattr(split, "value", split) if split is not None else "unknown"
            descriptions.append(f"{name} ({split_name}) | {len(dataset)} volumes")
        return descriptions

    def _sample_dataset(self) -> torch.utils.data.Dataset:
        idx = int(self.rng.integers(len(self.datasets)))
        return self.datasets[idx]

    def _sample_indices(self, dataset: torch.utils.data.Dataset) -> Tuple[int, int]:
        if len(dataset) < 2:
            raise ValueError("Datasets must contain at least two samples for episodic training.")
        indices = self.rng.choice(len(dataset), size=2, replace=False)
        return int(indices[0]), int(indices[1])

    def _prepare_sample(
        self,
        sample: Dict[str, torch.Tensor],
        *,
        augmentation: Optional[MedicalAugmentation],
        apply_noise: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = sample["image"].float()
        mask = sample.get("mask")
        if mask is None:
            raise ValueError("Segmentation mask is required for episodic training.")

        entry = {
            "image": image.clone(),
            "mask": mask.clone().float(),
            "meta": dict(sample.get("meta", {})),
        }
        if augmentation is not None:
            entry = augmentation(entry)
        image = entry["image"]
        mask = torch.round(entry["mask"]).to(torch.int64)

        if apply_noise and self.config.query_noise_std > 0.0:
            noise = torch.randn_like(image) * self.config.query_noise_std
            image = (image + noise).clamp(0.0, 1.0)

        target_size = self.config.volume_size
        if image.shape[-3:] != target_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=target_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        if mask.shape[-3:] != target_size:
            mask = F.interpolate(
                mask.float().unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode="nearest",
            ).squeeze(0).squeeze(0).to(torch.int64)

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        return image, mask

    def _extract_class_masks(
        self,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        unique_classes = [int(c.item()) for c in torch.unique(mask) if int(c.item()) != 0]
        if len(unique_classes) == 0:
            return torch.zeros(0, *mask.shape, device=mask.device), []

        # Random class dropping to encourage robustness.
        if self.config.random_class_drop_prob > 0.0 and len(unique_classes) > 1:
            kept = [
                cls
                for cls in unique_classes
                if self.rng.random() > self.config.random_class_drop_prob
            ]
            if kept:
                unique_classes = kept

        bin_masks = []
        for cls in unique_classes:
            bin_mask = (mask == cls).float()
            if bin_mask.sum() == 0:
                continue
            bin_masks.append(bin_mask)

        if not bin_masks:
            return torch.zeros(0, *mask.shape, device=mask.device), []

        stacked = torch.stack(bin_masks, dim=0)
        return stacked, unique_classes

    def _pad_class_dimension(
        self,
        tensors: List[torch.Tensor],
        *,
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        max_classes = max(t.shape[0] for t in tensors)
        padded = []
        for tensor in tensors:
            if tensor.shape[0] == max_classes:
                padded.append(tensor)
                continue
            pad_shape = (max_classes - tensor.shape[0],) + tensor.shape[1:]
            pad_tensor = torch.full(pad_shape, pad_value, device=tensor.device, dtype=tensor.dtype)
            padded.append(torch.cat([tensor, pad_tensor], dim=0))
        return torch.stack(padded, dim=0)

    def _sample_batch(self):
        support_images: List[torch.Tensor] = []
        support_masks: List[torch.Tensor] = []
        query_images: List[torch.Tensor] = []
        query_masks: List[torch.Tensor] = []
        episode_class_ids: List[List[int]] = []

        while len(support_images) < self.config.batch_size:
            dataset = self._sample_dataset()
            support_idx, query_idx = self._sample_indices(dataset)
            support_raw = dataset[support_idx]
            query_raw = dataset[query_idx]

            support_image, support_mask = self._prepare_sample(
                support_raw,
                augmentation=self.support_augmentation,
                apply_noise=False,
            )
            support_mask = support_mask.to(support_image.device)
            class_masks, class_ids = self._extract_class_masks(support_mask)
            if len(class_ids) == 0:
                continue

            query_image, query_mask_raw = self._prepare_sample(
                query_raw,
                augmentation=self.query_augmentation,
                apply_noise=True,
            )
            query_mask_raw = query_mask_raw.to(query_image.device)
            query_class_masks = []
            for cls in class_ids:
                class_mask = (query_mask_raw == cls).float()
                query_class_masks.append(class_mask)
            query_class_tensor = torch.stack(query_class_masks, dim=0)

            support_images.append(support_image)
            support_masks.append(class_masks)
            query_images.append(query_image)
            query_masks.append(query_class_tensor)
            episode_class_ids.append(class_ids)

        support_batch = torch.stack(support_images, dim=0).to(self.device)
        query_batch = torch.stack(query_images, dim=0).to(self.device)
        support_mask_batch = self._pad_class_dimension(support_masks).to(self.device)
        query_mask_batch = self._pad_class_dimension(query_masks).to(self.device)

        return {
            "support_images": support_batch,
            "support_masks": support_mask_batch,
            "query_images": query_batch,
            "query_masks": query_mask_batch,
            "class_ids": episode_class_ids,
        }

    def _run_episode(self, batch: Dict[str, torch.Tensor]) -> float:
        self.optimizer.zero_grad(set_to_none=True)

        support_images = batch["support_images"]
        support_masks = batch["support_masks"]
        query_images = batch["query_images"]
        query_masks = batch["query_masks"]

        support_dict = self.model.encode_support(support_images, support_masks)
        task_embeddings = support_dict["task_embeddings"]

        outputs = self.model(query_images, task_embeddings)
        logits = outputs["logits"]

        class_weights = compute_class_weights(query_masks)
        loss = self.loss_fn(logits, query_masks, class_weights=class_weights)
        loss.backward()

        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

        self.optimizer.step()

        if self.model.memory_bank is not None:
            # Update memory bank per sample to handle variable class counts
            batch_size = task_embeddings.shape[0]
            for i in range(batch_size):
                sample_embeddings = task_embeddings[i].detach()  # (K, C)
                sample_class_ids = batch["class_ids"][i]  # List[int] of length K
                # Only update if we have valid embeddings and class IDs
                if sample_embeddings.shape[0] == len(sample_class_ids) and len(sample_class_ids) > 0:
                    self.model.update_memory_bank(
                        sample_embeddings.unsqueeze(0),  # (1, K, C)
                        [sample_class_ids]  # List[List[int]]
                    )

        return float(loss.item())

    def _apply_schedule(self, iteration: int) -> None:
        if iteration <= self.config.warmup_iterations:
            scale = iteration / max(1, self.config.warmup_iterations)
        else:
            decay_steps = max(0, iteration - self.config.warmup_iterations)
            decay_factor = math.pow(
                self.config.lr_decay_gamma,
                decay_steps / max(1, self.config.decay_interval),
            )
            scale = decay_factor

        lr = self.config.base_learning_rate * scale
        for group in self.optimizer.param_groups:
            group["lr"] = lr


