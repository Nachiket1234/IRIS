"""
ACDC (Automatic Cardiac Diagnosis Challenge) dataset loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from ..base import DatasetSplit, MedicalDataset, VolumeRecord, ensure_unique_subject_ids
from ..factory import register_dataset


def _derive_subject_id(path: Path) -> str:
    # patientXXX_frameYY.nii.gz -> patientXXX_frameYY
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        name = name[: -len(".nii")]
    return name.replace("_gt", "")


@register_dataset("acdc")
class ACDCDataset(MedicalDataset):
    """
    Loader for the ACDC cardiac MRI segmentation dataset.

    Expected directory layout (default):
        root/
            training/
                patient001/
                    patient001_frame01.nii.gz
                    patient001_frame01_gt.nii.gz
                    ...
            testing/
                ...
    """

    dataset_name = "acdc"
    modality = "MRI"
    anatomy = "cardiac"
    target_classes = (1, 2, 3)  # RV, Myocardium, LV

    def __init__(
        self,
        root: Path | str,
        split: DatasetSplit = DatasetSplit.TRAIN,
        *,
        subset: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.subset = subset
        super().__init__(root, split, **kwargs)

    def discover_records(self) -> Iterable[VolumeRecord]:
        root = self.root
        if self.subset:
            search_roots = [root / self.subset]
        else:
            search_roots = []
            training_root = root / "training"
            testing_root = root / "testing"
            if training_root.exists():
                search_roots.append(training_root)
            if testing_root.exists():
                search_roots.append(testing_root)
            if not search_roots:
                search_roots.append(root)

        image_paths: List[Path] = []
        mask_paths: List[Path] = []
        for base in search_roots:
            if base.exists():
                for path in base.rglob("*_frame*.nii.gz"):
                    if "_gt" in path.stem:
                        mask_paths.append(path)
                    else:
                        image_paths.append(path)

        mask_map = {_derive_subject_id(p): p for p in mask_paths}
        records: List[VolumeRecord] = []
        for image_path in image_paths:
            subject_id = _derive_subject_id(image_path)
            mask_path = mask_map.get(subject_id)
            if mask_path is None and not self.allow_missing_masks:
                continue
            record = VolumeRecord(
                image_path=image_path,
                mask_path=mask_path,
                subject_id=subject_id,
                dataset_name=self.dataset_name,
                modality=self.modality,
                anatomy=self.anatomy,
                classes=self.target_classes or (),
                metadata={"subset": self.subset, "src_path": str(image_path.parent)},
            )
            records.append(record)

        ensure_unique_subject_ids(records)
        return records

    def configure_preprocessing(self):
        config = super().configure_preprocessing()
        config.update({"modality": "MRI"})
        return config

