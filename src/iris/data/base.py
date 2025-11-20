"""
Shared abstractions for heterogeneous medical imaging datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_medical_volume
from .preprocessing import preprocess_image_and_mask


class DatasetSplit(str, Enum):
    """Canonical split names used across datasets."""

    TRAIN = "train"
    VALID = "val"
    TEST = "test"


@dataclass
class VolumeRecord:
    """
    Metadata describing a single medical imaging volume and its segmentation mask.

    Attributes:
        image_path: Path to the image volume (NIfTI, MHD, etc.).
        mask_path: Optional path to the segmentation mask volume.
        subject_id: Unique identifier for the subject / study.
        dataset_name: Identifier of the originating dataset.
        modality: Imaging modality (e.g., "CT", "MRI", "PET").
        anatomy: Anatomical region or clinical target (e.g., "abdomen").
        classes: Sorted list of label IDs within the mask (excluding background).
        metadata: Additional dataset-specific metadata.
    """

    image_path: Path
    mask_path: Optional[Path]
    subject_id: str
    dataset_name: str
    modality: str
    anatomy: str
    classes: Sequence[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation for logging or caching."""
        return {
            "image_path": str(self.image_path),
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "subject_id": self.subject_id,
            "dataset_name": self.dataset_name,
            "modality": self.modality,
            "anatomy": self.anatomy,
            "classes": list(self.classes),
            "metadata": dict(self.metadata),
        }


def default_split_strategy(
    records: Sequence[VolumeRecord],
    train_ratio: float = 0.75,
    val_ratio: float = 0.05,
    random_seed: int = 42,
) -> Dict[DatasetSplit, List[VolumeRecord]]:
    """
    Deterministically split records into train/val/test partitions.

    Args:
        records: Sequence of dataset records.
        train_ratio: Fraction of samples assigned to the training split.
        val_ratio: Fraction assigned to validation (evaluation split).
        random_seed: Seed for deterministic shuffling.

    Returns:
        Mapping from DatasetSplit to list of records.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    total = len(records)
    train_count = max(1, int(round(train_ratio * total)))
    val_count = max(0, int(round(val_ratio * total)))

    if train_count + val_count > total:
        overflow = train_count + val_count - total
        if val_count >= overflow:
            val_count -= overflow
        else:
            overflow -= val_count
            val_count = 0
            train_count = max(1, train_count - overflow)

    train_cutoff = train_count
    val_cutoff = train_cutoff + val_count

    split_map: Dict[DatasetSplit, List[VolumeRecord]] = {
        DatasetSplit.TRAIN: [],
        DatasetSplit.VALID: [],
        DatasetSplit.TEST: [],
    }

    for i, idx in enumerate(indices):
        record = records[idx]
        if i < train_cutoff:
            split_map[DatasetSplit.TRAIN].append(record)
        elif i < val_cutoff:
            split_map[DatasetSplit.VALID].append(record)
        else:
            split_map[DatasetSplit.TEST].append(record)
    return split_map


class MedicalDataset(Dataset):
    """
    Base dataset implementation handling medical volume loading and preprocessing.

    Subclasses should implement `discover_records` to enumerate the dataset
    specific files and optionally override `configure_preprocessing` for dataset
    specific preprocessing tweaks.
    """

    dataset_name: str = "medical"
    modality: str = "CT"
    anatomy: str = "generic"
    target_classes: Optional[Sequence[int]] = None

    def __init__(
        self,
        root: Path | str,
        split: DatasetSplit = DatasetSplit.TRAIN,
        *,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        augmentation: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        target_size: Tuple[int, int, int] = (128, 128, 128),
        spacing: Optional[Tuple[float, float, float]] = None,
        cache_data: bool = False,
        random_seed: int = 42,
        split_strategy: Callable[
            [Sequence[VolumeRecord]], Dict[DatasetSplit, List[VolumeRecord]]
        ] = default_split_strategy,
        allow_missing_masks: bool = False,
        **preprocess_overrides: Any,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.split = split
        self.transforms = transforms
        self.augmentation = augmentation
        self.target_size = target_size
        self.spacing = spacing
        self.cache_data = cache_data
        self.random_seed = random_seed
        self.allow_missing_masks = allow_missing_masks
        self._preprocess_overrides = preprocess_overrides

        records = list(self.discover_records())
        if not records:
            raise RuntimeError(
                f"No volume records discovered for dataset {self.dataset_name} at {self.root}"
            )

        # If train_ratio or val_ratio provided in kwargs, create partial split_strategy
        if 'train_ratio' in preprocess_overrides or 'val_ratio' in preprocess_overrides:
            split_params = {}
            if 'train_ratio' in preprocess_overrides:
                split_params['train_ratio'] = preprocess_overrides['train_ratio']
            if 'val_ratio' in preprocess_overrides:
                split_params['val_ratio'] = preprocess_overrides['val_ratio']
            split_params['random_seed'] = random_seed
            split_strategy = partial(default_split_strategy, **split_params)

        self._records_by_split = split_strategy(records)
        if split not in self._records_by_split:
            available = ", ".join(s.value for s in self._records_by_split.keys())
            raise KeyError(f"Split {split} not available. Found splits: {available}")

        self.records = self._records_by_split[split]
        self._cache: Dict[str, Dict[str, Any]] = {}

    # --------------------------------------------------------------------- #
    # Discovery & configuration hooks
    # --------------------------------------------------------------------- #
    def discover_records(self) -> Iterable[VolumeRecord]:
        """
        Enumerate dataset records by inspecting the root directory.

        Subclasses must implement this method to return VolumeRecord instances.
        """
        raise NotImplementedError

    def configure_preprocessing(self) -> Dict[str, Any]:
        """
        Provide dataset-specific preprocessing configuration overrides.
        """
        return {
            "target_size": self.target_size,
            "target_spacing": self.spacing,
            "modality": self.modality,
            "clip_values": None,
            "mri_percentiles": (1.0, 99.0),
        }

    # --------------------------------------------------------------------- #
    # Dataset API
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        cache_key = record.subject_id

        if self.cache_data and cache_key in self._cache:
            sample = self._cache[cache_key].copy()
        else:
            sample = self._load_record(record)
            if self.cache_data:
                self._cache[cache_key] = sample.copy()

        if self.augmentation and self.split == DatasetSplit.TRAIN:
            sample = self.augmentation(sample)

        if self.transforms:
            sample = self.transforms(sample)

        sample["meta"]["index"] = index
        sample["meta"]["split"] = self.split.value
        sample["meta"]["dataset"] = self.dataset_name
        return sample

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _load_record(self, record: VolumeRecord) -> Dict[str, Any]:
        if record.mask_path is None and not self.allow_missing_masks:
            raise FileNotFoundError(
                f"Record {record.subject_id} is missing mask. "
                "Set allow_missing_masks=True to bypass."
            )

        image, image_meta = load_medical_volume(record.image_path)
        mask = None
        mask_meta = None
        if record.mask_path:
            mask, mask_meta = load_medical_volume(record.mask_path, ensure_nd=True)

        preprocessing_config = self.configure_preprocessing()
        preprocessing_config.update(self._preprocess_overrides)
        processed = preprocess_image_and_mask(
            image=image,
            mask=mask,
            image_meta=image_meta,
            mask_meta=mask_meta,
            modality=record.modality or self.modality,
            target_size=preprocessing_config.get("target_size"),
            target_spacing=preprocessing_config.get("target_spacing"),
            random_state=self.random_seed,
            metadata={**record.metadata, "classes": record.classes},
            clip_values=preprocessing_config.get("clip_values"),
            mri_percentiles=preprocessing_config.get("mri_percentiles", (1.0, 99.0)),
        )

        processed["meta"].update(
            {
                "subject_id": record.subject_id,
                "dataset_name": record.dataset_name,
                "anatomy": record.anatomy,
                "modality": record.modality,
            }
        )
        return processed


def ensure_unique_subject_ids(records: Iterable[VolumeRecord]) -> None:
    """
    Validate that subject IDs are unique to avoid cache collisions.
    """
    seen: Dict[str, str] = {}
    for record in records:
        if record.subject_id in seen:
            previous = seen[record.subject_id]
            raise ValueError(
                f"Duplicate subject_id {record.subject_id} detected. "
                f"Existing path: {previous}, new path: {record.image_path}"
            )
        seen[record.subject_id] = str(record.image_path)

