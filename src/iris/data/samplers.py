"""
Samplers and batching utilities for episodic training.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from .base import DatasetSplit


class EpisodicBatchSampler:
    """
    Sample reference-query pairs from the same dataset split.

    Each episode samples `n_support` support volumes and `n_query` query volumes
    for a given dataset name, mirroring the episodic evaluation used in the
    IRIS paper. Episodes can be constrained to a single dataset or span multiple
    datasets based on the provided index metadata.
    """

    def __init__(
        self,
        indices: Sequence[int],
        *,
        dataset_names: Sequence[str],
        n_support: int,
        n_query: int,
        episodes_per_epoch: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if len(indices) != len(dataset_names):
            raise ValueError("indices and dataset_names must have the same length")
        if n_support <= 0 or n_query <= 0:
            raise ValueError("n_support and n_query must be positive integers")

        self.indices = list(indices)
        self.dataset_names = list(dataset_names)
        self.n_support = n_support
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        self.rng = rng or np.random.default_rng()

        self._indices_by_dataset: Dict[str, List[int]] = defaultdict(list)
        for idx, name in zip(self.indices, self.dataset_names):
            self._indices_by_dataset[name].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        dataset_keys = list(self._indices_by_dataset.keys())
        for _ in range(self.episodes_per_epoch):
            dataset = str(self.rng.choice(dataset_keys))
            candidates = self._indices_by_dataset[dataset]
            if len(candidates) < self.n_support + self.n_query:
                raise ValueError(
                    f"Dataset {dataset} does not have enough samples for the requested episode size."
                )
            shuffled = self.rng.permutation(candidates)
            support = shuffled[: self.n_support]
            query = shuffled[self.n_support : self.n_support + self.n_query]
            yield support.tolist() + query.tolist()

    def __len__(self) -> int:
        return self.episodes_per_epoch


def group_indices_by_split(
    dataset,
    split: DatasetSplit,
) -> Tuple[List[int], List[str]]:
    """
    Utility to extract indices and dataset names for a specific split.
    """
    indices = []
    dataset_names = []
    for idx, record in enumerate(dataset.records):
        indices.append(idx)
        dataset_names.append(record.dataset_name)
    return indices, dataset_names

