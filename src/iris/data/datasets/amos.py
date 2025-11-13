"""
AMOS (Abdominal Multi-Organ Segmentation) dataset loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from ..base import MedicalDataset, VolumeRecord, ensure_unique_subject_ids
from ..factory import register_dataset

AMOS_CLASSES = {
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "portal_vein_and_splenic_vein",
    11: "pancreas",
    12: "right_adrenal_gland",
    13: "left_adrenal_gland",
    14: "duodenum",
    15: "left_ventricle",
}


def _strip_suffix(path: Path) -> str:
    name = path.name
    for suffix in [".nii.gz", ".nii", ".mha", ".mhd"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


@register_dataset("amos")
class AMOSDataset(MedicalDataset):
    """
    Loader for the AMOS abdominal CT/MRI segmentation dataset.

    Expected directory layout (default):
        root/
            imagesTr/
            labelsTr/
            imagesTs/
            labelsTs/
    """

    dataset_name = "amos"
    modality = "CT"
    anatomy = "abdomen"
    target_classes: Sequence[int] = tuple(sorted(AMOS_CLASSES.keys()))

    def __init__(
        self,
        root: Path | str,
        *,
        use_mri: bool = False,
        image_dir: Optional[str] = None,
        label_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.use_mri = use_mri
        self.image_dir = image_dir
        self.label_dir = label_dir
        super().__init__(root, **kwargs)

    def discover_records(self) -> Iterable[VolumeRecord]:
        image_root = (
            Path(self.image_dir) if self.image_dir else self.root / "imagesTr"
        )
        label_root = (
            Path(self.label_dir) if self.label_dir else self.root / "labelsTr"
        )

        if not image_root.exists():
            raise FileNotFoundError(f"AMOS image directory not found: {image_root}")
        if not label_root.exists() and not self.allow_missing_masks:
            raise FileNotFoundError(f"AMOS label directory not found: {label_root}")

        image_paths = sorted(image_root.glob("*.nii*"))
        label_paths = sorted(label_root.glob("*.nii*")) if label_root.exists() else []
        label_map = {_strip_suffix(p): p for p in label_paths}

        records: List[VolumeRecord] = []
        for image_path in image_paths:
            subject_id = _strip_suffix(image_path)
            mask_path = label_map.get(subject_id)
            if mask_path is None and not self.allow_missing_masks:
                continue

            modality = "MRI" if self.use_mri and "MRI" in subject_id.upper() else "CT"
            record = VolumeRecord(
                image_path=image_path,
                mask_path=mask_path,
                subject_id=subject_id,
                dataset_name=self.dataset_name,
                modality=modality,
                anatomy=self.anatomy,
                classes=self.target_classes,
                metadata={"class_names": AMOS_CLASSES, "use_mri": self.use_mri},
            )
            records.append(record)

        ensure_unique_subject_ids(records)
        return records

    def configure_preprocessing(self):
        config = super().configure_preprocessing()
        config.update(
            {
                "modality": "MRI" if self.use_mri else "CT",
                "clip_values": (-175.0, 275.0) if not self.use_mri else None,
            }
        )
        return config

