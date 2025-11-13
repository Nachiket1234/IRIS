"""
Training and evaluation utilities for the IRIS medical segmentation model.
"""

from .demo import ClinicalDemoConfig, MedicalDemoRunner
from .evaluation import EvaluationConfig, MedicalEvaluationSuite
from .lamb import Lamb
from .pipeline import EpisodicTrainer, EpisodicTrainingConfig
from .utils import set_global_seed
from .visualization import (
    extract_middle_slices,
    plot_performance_dashboard,
    plot_training_curves,
    render_multi_planar_views,
)

__all__ = [
    "EpisodicTrainer",
    "EpisodicTrainingConfig",
    "MedicalEvaluationSuite",
    "EvaluationConfig",
    "Lamb",
    "set_global_seed",
    "MedicalDemoRunner",
    "ClinicalDemoConfig",
    "render_multi_planar_views",
    "extract_middle_slices",
    "plot_performance_dashboard",
    "plot_training_curves",
]


