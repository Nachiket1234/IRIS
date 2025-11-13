"""
High-level data loading utilities for the IRIS medical segmentation model.

This package provides dataset abstractions, preprocessing utilities, and
dataset-specific loaders for the real-world medical imaging datasets referenced
in the IRIS research paper. The goal is to offer a unified interface that can
handle heterogeneous modalities, anatomical regions, and annotation schemes
while supporting episodic sampling for continual and few-shot adaptation.
"""

from .base import DatasetSplit, MedicalDataset, VolumeRecord
from .factory import DATASET_REGISTRY, build_dataset
from .samplers import EpisodicBatchSampler

# Ensure dataset modules are imported so registration side effects run.
from . import datasets as _datasets  # noqa: F401

__all__ = [
    "DatasetSplit",
    "MedicalDataset",
    "VolumeRecord",
    "DATASET_REGISTRY",
    "build_dataset",
    "EpisodicBatchSampler",
]

