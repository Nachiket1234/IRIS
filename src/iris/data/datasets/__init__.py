"""
Dataset-specific loaders for the IRIS medical datasets.
"""

from .acdc import ACDCDataset
from .amos import AMOSDataset
from .brain_tumor import BrainTumorDataset
from .chest_xray_masks import ChestXrayMasksDataset
from .covid_ct import COVIDCTDataset
from .drive import DRIVEDataset
from .isic import ISICDataset
from .kvasir import KvasirDataset
from .msd_pancreas import MSDPancreasDataset
from .segthor import SegTHORDataset

__all__ = [
    "ACDCDataset",
    "AMOSDataset",
    "BrainTumorDataset",
    "ChestXrayMasksDataset",
    "COVIDCTDataset",
    "DRIVEDataset",
    "ISICDataset",
    "KvasirDataset",
    "MSDPancreasDataset",
    "SegTHORDataset",
]

