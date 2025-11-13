"""
Core IRIS architecture components as described in Section 3.2 of the paper.

This subpackage exposes the medical-imaging-optimised 3D UNet encoder, task
encoding module, bidirectional mask decoder, and the cohesive IrisModel that
binds them together for episodic segmentation.
"""

from .core import IrisModel
from .decoder import MaskDecoder
from .encoder import Medical3DUNetEncoder
from .memory import ClassMemoryBank
from .task_encoding import TaskEncodingModule
from .tuning import DiceCrossEntropyLoss, InContextTuner

__all__ = [
    "IrisModel",
    "MaskDecoder",
    "Medical3DUNetEncoder",
    "TaskEncodingModule",
    "ClassMemoryBank",
    "InContextTuner",
    "DiceCrossEntropyLoss",
]


