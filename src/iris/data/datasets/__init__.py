"""
Dataset-specific loaders for the IRIS medical datasets.
"""

from .acdc import ACDCDataset
from .amos import AMOSDataset
from .msd_pancreas import MSDPancreasDataset
from .segthor import SegTHORDataset

__all__ = [
    "ACDCDataset",
    "AMOSDataset",
    "MSDPancreasDataset",
    "SegTHORDataset",
]

