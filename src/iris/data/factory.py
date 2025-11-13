"""
Dataset registry & builders for medical datasets referenced in the IRIS paper.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import DatasetSplit, MedicalDataset


class DatasetRegistry(dict):
    """
    Simple registry mapping dataset identifiers to dataset classes.
    """

    def register(self, name: str, dataset_cls: Type[MedicalDataset]) -> None:
        key = name.lower()
        if key in self:
            raise KeyError(f"Dataset '{name}' is already registered.")
        self[key] = dataset_cls

    def build(
        self,
        name: str,
        root: str,
        split: DatasetSplit,
        **kwargs: Any,
    ) -> MedicalDataset:
        key = name.lower()
        if key not in self:
            available = ", ".join(self.keys()) or "<empty>"
            raise KeyError(f"Dataset '{name}' is not registered. Available: {available}")
        dataset_cls = self[key]
        return dataset_cls(root=root, split=split, **kwargs)


DATASET_REGISTRY = DatasetRegistry()


def register_dataset(name: str):
    """
    Decorator to register dataset classes in the global registry.
    """

    def decorator(cls: Type[MedicalDataset]) -> Type[MedicalDataset]:
        DATASET_REGISTRY.register(name, cls)
        return cls

    return decorator


def build_dataset(
    name: str,
    root: str,
    split: str | DatasetSplit,
    **kwargs: Any,
) -> MedicalDataset:
    """
    Convenience builder that accepts split as a string or DatasetSplit.
    """
    split_enum = (
        split if isinstance(split, DatasetSplit) else DatasetSplit(split.lower())
    )
    return DATASET_REGISTRY.build(name=name, root=root, split=split_enum, **kwargs)

