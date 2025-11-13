"""
SegTHOR thoracic organ segmentation dataset loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from ..base import MedicalDataset, VolumeRecord, ensure_unique_subject_ids
from ..factory import register_dataset

SEGTHOR_CLASSES = {
    1: "esophagus",
    2: "heart",
    3: "trachea",
    4: "aorta",
}


def _base_name(path: Path) -> str:
    name = path.name
    for suffix in [".nii.gz", ".nii", ".mhd", ".mha"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


@register_dataset("segthor")
class SegTHORDataset(MedicalDataset):
    """
    Loader for the SegTHOR thoracic CT segmentation dataset.

    Expected directory layout:
        root/
            images/
            labels/
    """

    dataset_name = "segthor"
    modality = "CT"
    anatomy = "thorax"
    target_classes: Sequence[int] = tuple(sorted(SEGTHOR_CLASSES.keys()))

    def __init__(
        self,
        root: Path | str,
        *,
        image_dir: str | None = None,
        label_dir: str | None = None,
        **kwargs,
    ) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        super().__init__(root, **kwargs)

    def discover_records(self) -> Iterable[VolumeRecord]:
        image_root = Path(self.image_dir) if self.image_dir else self.root / "images"
        label_root = Path(self.label_dir) if self.label_dir else self.root / "labels"

        if not image_root.exists():
            raise FileNotFoundError(f"SegTHOR images not found at {image_root}")
        if not label_root.exists() and not self.allow_missing_masks:
            raise FileNotFoundError(f"SegTHOR labels not found at {label_root}")

        image_paths = sorted(image_root.glob("*.nii*"))
        label_paths = sorted(label_root.glob("*.nii*")) if label_root.exists() else []
        label_map = {_base_name(p): p for p in label_paths}

        records: List[VolumeRecord] = []
        for image_path in image_paths:
            subject_id = _base_name(image_path)
            mask_path = label_map.get(subject_id)
            if mask_path is None and not self.allow_missing_masks:
                continue

            record = VolumeRecord(
                image_path=image_path,
                mask_path=mask_path,
                subject_id=subject_id,
                dataset_name=self.dataset_name,
                modality=self.modality,
                anatomy=self.anatomy,
                classes=self.target_classes,
                metadata={"class_names": SEGTHOR_CLASSES},
            )
            records.append(record)

        ensure_unique_subject_ids(records)
        return records

    def configure_preprocessing(self):
        config = super().configure_preprocessing()
        config.update({"clip_values": (-1000.0, 1000.0), "modality": "CT"})
        return config

