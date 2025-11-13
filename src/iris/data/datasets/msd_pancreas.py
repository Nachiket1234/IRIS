"""
Medical Segmentation Decathlon (MSD) Pancreas dataset loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from ..base import MedicalDataset, VolumeRecord, ensure_unique_subject_ids
from ..factory import register_dataset


def _without_suffix(path: Path) -> str:
    name = path.name
    for suffix in [".nii.gz", ".nii"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


@register_dataset("msd_pancreas")
class MSDPancreasDataset(MedicalDataset):
    """
    Loader for the MSD Pancreas segmentation dataset.

    Expected directory layout (default):
        root/
            imagesTr/
            labelsTr/
            imagesTs/ (optional)
    """

    dataset_name = "msd_pancreas"
    modality = "CT"
    anatomy = "abdomen"
    target_classes: Sequence[int] = (1, 2)  # pancreas, tumor

    def discover_records(self) -> Iterable[VolumeRecord]:
        image_root = self.root / "imagesTr"
        label_root = self.root / "labelsTr"

        if not image_root.exists():
            raise FileNotFoundError(f"MSD Pancreas images not found at {image_root}")
        if not label_root.exists() and not self.allow_missing_masks:
            raise FileNotFoundError(f"MSD Pancreas labels not found at {label_root}")

        image_paths = sorted(image_root.glob("*.nii.gz"))
        label_paths = sorted(label_root.glob("*.nii.gz")) if label_root.exists() else []
        label_map = {_without_suffix(p): p for p in label_paths}

        records: List[VolumeRecord] = []
        for image_path in image_paths:
            subject_id = _without_suffix(image_path)
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
                metadata={"task": "pancreas_tumor"},
            )
            records.append(record)

        ensure_unique_subject_ids(records)
        return records

    def configure_preprocessing(self):
        config = super().configure_preprocessing()
        config.update({"clip_values": (-150.0, 300.0), "modality": "CT"})
        return config

