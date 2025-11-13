"""
Preprocessing utilities for 3D medical imaging volumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import zoom

Array3D = np.ndarray


@dataclass
class PreprocessingConfig:
    target_size: Optional[Tuple[int, int, int]] = (128, 128, 128)
    target_spacing: Optional[Tuple[float, float, float]] = None
    modality: str = "CT"
    clip_values: Optional[Tuple[float, float]] = None
    mri_percentiles: Tuple[float, float] = (1.0, 99.0)
    random_state: int = 42
    metadata: Optional[Dict[str, Any]] = None


def _resample_volume(
    volume: Array3D,
    *,
    current_spacing: Optional[Tuple[float, float, float]],
    target_spacing: Optional[Tuple[float, float, float]],
    order: int,
) -> Array3D:
    if target_spacing is None or current_spacing is None:
        return volume

    zoom_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(target_spacing))
    ]
    return zoom(volume, zoom=zoom_factors, order=order)


def _resize_volume(
    volume: Array3D,
    *,
    target_size: Optional[Tuple[int, int, int]],
    order: int,
) -> Array3D:
    if target_size is None:
        return volume
    factors = [
        target_size[i] / volume.shape[i] if volume.shape[i] > 0 else 1.0
        for i in range(len(target_size))
    ]
    return zoom(volume, zoom=factors, order=order)


def _normalize_ct(
    volume: Array3D,
    clip_values: Optional[Tuple[float, float]],
) -> Array3D:
    lower, upper = clip_values or (-1024.0, 1024.0)
    volume = np.clip(volume, lower, upper)
    volume = (volume - lower) / (upper - lower)
    return volume.astype(np.float32)


def _normalize_mri(
    volume: Array3D,
    percentiles: Tuple[float, float],
) -> Array3D:
    low, high = np.percentile(volume, percentiles)
    volume = np.clip(volume, low, high)
    volume = volume - low
    denom = high - low if high > low else 1.0
    volume = volume / denom
    return volume.astype(np.float32)


def normalize_intensity(
    volume: Array3D,
    modality: str,
    *,
    clip_values: Optional[Tuple[float, float]] = None,
    mri_percentiles: Tuple[float, float] = (1.0, 99.0),
) -> Array3D:
    modality_upper = modality.upper()
    if modality_upper == "CT":
        return _normalize_ct(volume, clip_values)
    if modality_upper == "MRI":
        return _normalize_mri(volume, mri_percentiles)
    if modality_upper == "PET":
        # Standard uptake value normalization to zero mean & unit variance.
        volume = volume.astype(np.float32)
        mean = float(volume.mean())
        std = float(volume.std()) or 1.0
        return (volume - mean) / std
    # Default: min-max normalisation
    volume = volume.astype(np.float32)
    volume_min = float(volume.min())
    volume_max = float(volume.max())
    if volume_max > volume_min:
        volume = (volume - volume_min) / (volume_max - volume_min)
    else:
        volume = np.zeros_like(volume, dtype=np.float32)
    return volume


def preprocess_image_and_mask(
    *,
    image: Array3D,
    mask: Optional[Array3D],
    image_meta: Dict[str, Any],
    mask_meta: Optional[Dict[str, Any]],
    modality: str,
    target_size: Optional[Tuple[int, int, int]],
    target_spacing: Optional[Tuple[float, float, float]],
    random_state: int,
    metadata: Optional[Dict[str, Any]] = None,
    clip_values: Optional[Tuple[float, float]] = None,
    mri_percentiles: Tuple[float, float] = (1.0, 99.0),
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    current_image_spacing = image_meta.get("spacing")
    current_mask_spacing = mask_meta.get("spacing") if mask_meta else current_image_spacing

    resampled_image = _resample_volume(
        image,
        current_spacing=current_image_spacing,
        target_spacing=target_spacing,
        order=3,
    )

    resampled_mask = (
        _resample_volume(
            mask,
            current_spacing=current_mask_spacing,
            target_spacing=target_spacing,
            order=0,
        )
        if mask is not None
        else None
    )

    resized_image = _resize_volume(resampled_image, target_size=target_size, order=3)
    resized_mask = (
        _resize_volume(resampled_mask, target_size=target_size, order=0)
        if resampled_mask is not None
        else None
    )

    normalized_image = normalize_intensity(
        resized_image,
        modality,
        clip_values=clip_values,
        mri_percentiles=mri_percentiles,
    )

    if resized_mask is not None:
        resized_mask = resized_mask.astype(np.int16)
        unique_classes = np.unique(resized_mask)
    else:
        unique_classes = np.array([], dtype=np.int16)

    sample = {
        "image": torch.from_numpy(normalized_image).unsqueeze(0),  # add channel dim
        "mask": torch.from_numpy(resized_mask) if resized_mask is not None else None,
        "meta": {
            "affine": image_meta.get("affine"),
            "spacing": target_spacing or current_image_spacing,
            "original_spacing": current_image_spacing,
            "original_shape": image_meta.get("original_shape"),
            "target_size": target_size,
        "unique_classes": unique_classes.tolist(),
        "rng_seed": random_state,
            **(metadata or {}),
        },
    }
    return sample

