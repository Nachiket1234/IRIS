"""Visualization utilities for medical evaluation and demonstrations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt  # pragma: no cover
except ImportError:  # pragma: no cover
    plt = None


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "Matplotlib is required for medical visualisations. "
            "Install it via `pip install matplotlib`."
        )


def _to_numpy(volume: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(volume, torch.Tensor):
        array = volume.detach().cpu().numpy()
    else:
        array = np.asarray(volume)
    return np.squeeze(array)


def extract_middle_slices(volume: torch.Tensor | np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract axial, coronal, and sagittal middle slices from a 3D volume.
    """
    array = _to_numpy(volume)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected 3D volume, received shape {array.shape}")
    depth, height, width = array.shape
    axial = array[depth // 2]
    coronal = array[:, height // 2, :]
    sagittal = array[:, :, width // 2]
    return {"axial": axial, "coronal": coronal, "sagittal": sagittal}


def render_multi_planar_views(
    image: torch.Tensor | np.ndarray,
    *,
    prediction: Optional[torch.Tensor | np.ndarray] = None,
    target: Optional[torch.Tensor | np.ndarray] = None,
    reference: Optional[torch.Tensor | np.ndarray] = None,
    titles: Optional[Sequence[str]] = None,
    cmap: str = "gray",
    alpha: float = 0.4,
    figsize: tuple[int, int] = (15, 5),
) -> "plt.Figure":
    """
    Render axial, coronal, and sagittal views with optional overlays.
    """
    _require_matplotlib()
    slices = extract_middle_slices(image)
    pred_slices = extract_middle_slices(prediction) if prediction is not None else None
    target_slices = extract_middle_slices(target) if target is not None else None
    reference_slices = extract_middle_slices(reference) if reference is not None else None

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    default_titles = ["Axial", "Coronal", "Sagittal"]

    for idx, (plane, slice_img) in enumerate(slices.items()):
        ax = axes[idx]
        ax.imshow(slice_img, cmap=cmap)
        ax.axis("off")
        title = titles[idx] if titles and idx < len(titles) else default_titles[idx]
        overlays = []
        if target_slices is not None:
            overlays.append(("GT", target_slices[plane]))
        if pred_slices is not None:
            overlays.append(("Pred", pred_slices[plane]))
        if reference_slices is not None:
            overlays.append(("Ref", reference_slices[plane]))
        for label, overlay in overlays:
            ax.imshow(np.ma.masked_where(overlay <= 0.5, overlay), cmap="jet", alpha=alpha)
            title += f" | {label}"
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_performance_dashboard(
    evaluation_results: Dict[str, Dict[str, dict]],
    *,
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """
    Create a dashboard summarising mean Dice per strategy across datasets.
    """
    _require_matplotlib()

    strategy_names: Sequence[str] = []
    dataset_labels: Sequence[str] = []
    dice_matrix: list[list[float]] = []

    for dataset, payload in evaluation_results.items():
        dataset_labels.append(dataset)
        strategies = payload.get("strategies", {})
        if not strategy_names:
            strategy_names = list(strategies.keys())
        row = [strategies[s]["dice_mean"] for s in strategy_names]
        dice_matrix.append(row)

    dice_array = np.array(dice_matrix)
    x = np.arange(len(dataset_labels))
    width = 0.8 / max(1, len(strategy_names))

    fig, ax = plt.subplots(figsize=figsize)
    for idx, strategy in enumerate(strategy_names):
        offset = (idx - (len(strategy_names) - 1) / 2) * width
        ax.bar(x + offset, dice_array[:, idx], width=width, label=strategy)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Dice")
    ax.set_ylim(0, 1)
    ax.set_title("Strategy-wise Dice Performance")
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_training_curves(
    losses: Sequence[float],
    *,
    secondary_metric: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 4),
) -> "plt.Figure":
    """
    Visualise training loss (and optional secondary metric) over iterations.
    """
    _require_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(losses, label=labels[0] if labels else "Segmentation Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.3)

    if secondary_metric is not None:
        ax2 = ax.twinx()
        ax2.plot(secondary_metric, color="tab:orange", label=labels[1] if labels and len(labels) > 1 else "Metric")
        ax2.set_ylabel("Secondary Metric")
        lines, legends = ax.get_legend_handles_labels()
        lines2, legends2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, legends + legends2, loc="upper right")
    elif labels:
        ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

