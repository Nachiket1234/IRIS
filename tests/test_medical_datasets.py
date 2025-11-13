import numpy as np
import nibabel as nib

from iris.data import DatasetSplit, build_dataset
from iris.data.augmentations import MedicalAugmentation
from iris.data.samplers import EpisodicBatchSampler


def _write_nifti(path, data, voxel_size=(1.5, 1.5, 2.0)):
    affine = np.diag(list(voxel_size) + [1.0])
    image = nib.Nifti1Image(data.astype(np.float32), affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def _prepare_acdc_dataset(tmp_path):
    training_root = tmp_path / "training" / "patient001"
    training_root.mkdir(parents=True, exist_ok=True)
    image = np.random.rand(16, 16, 16).astype(np.float32)
    mask = np.random.randint(0, 4, size=(16, 16, 16), dtype=np.int16)
    _write_nifti(training_root / "patient001_frame01.nii.gz", image)
    _write_nifti(training_root / "patient001_frame01_gt.nii.gz", mask)

    # Additional sample for episodic sampler testing
    training_root2 = tmp_path / "training" / "patient002"
    training_root2.mkdir(parents=True, exist_ok=True)
    image2 = np.random.rand(16, 16, 16).astype(np.float32)
    mask2 = np.random.randint(0, 4, size=(16, 16, 16), dtype=np.int16)
    _write_nifti(training_root2 / "patient002_frame01.nii.gz", image2)
    _write_nifti(training_root2 / "patient002_frame01_gt.nii.gz", mask2)


def test_acdc_dataset_loading(tmp_path):
    _prepare_acdc_dataset(tmp_path)
    dataset = build_dataset(
        "acdc",
        root=str(tmp_path),
        split=DatasetSplit.TRAIN,
        target_size=(8, 8, 8),
        cache_data=False,
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert "image" in sample and "mask" in sample and "meta" in sample
    assert sample["image"].shape == (1, 8, 8, 8)
    assert sample["mask"].shape == (8, 8, 8)
    assert sample["meta"]["dataset_name"] == "acdc"
    assert set(sample["meta"]["unique_classes"]).issubset({0, 1, 2, 3})


def test_episodic_sampler(tmp_path):
    _prepare_acdc_dataset(tmp_path)
    dataset = build_dataset(
        "acdc",
        root=str(tmp_path),
        split=DatasetSplit.TRAIN,
        target_size=(8, 8, 8),
        cache_data=False,
    )

    indices = list(range(len(dataset)))
    dataset_names = [dataset.dataset_name for _ in indices]
    sampler = EpisodicBatchSampler(
        indices=indices,
        dataset_names=dataset_names,
        n_support=1,
        n_query=1,
        episodes_per_epoch=2,
    )

    for batch in sampler:
        assert len(batch) == 2
        assert len(set(batch)) == 2  # support and query distinct


def test_medical_augmentation(tmp_path):
    _prepare_acdc_dataset(tmp_path)
    dataset = build_dataset(
        "acdc",
        root=str(tmp_path),
        split=DatasetSplit.TRAIN,
        target_size=(16, 16, 16),
        cache_data=False,
    )

    augmentation = MedicalAugmentation(crop_size=(12, 12, 12))
    sample = dataset[0]
    augmented = augmentation(sample)
    assert augmented["image"].shape == (1, 12, 12, 12)
    assert augmented["mask"].shape == (12, 12, 12)
    assert "augmentation" in augmented["meta"]

