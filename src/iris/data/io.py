"""
I/O utilities for reading heterogeneous medical imaging formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "The nibabel package is required for medical image loading. "
        "Install it via `pip install nibabel`."
    ) from exc

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    sitk = None


SUPPORTED_EXTENSIONS = {".nii", ".nii.gz", ".mhd", ".mha"}


@dataclass
class VolumeMetadata:
    """Metadata captured during volume loading."""

    affine: Optional[np.ndarray] = None
    spacing: Optional[Tuple[float, float, float]] = None
    original_shape: Optional[Tuple[int, int, int]] = None
    orientation: Optional[str] = None
    header: Optional[Any] = None


def _resolve_path(path: Path | str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Volume path not found: {resolved}")
    return resolved


def _load_with_nibabel(path: Path) -> Tuple[np.ndarray, VolumeMetadata]:
    img = nib.load(str(path))
    canonical = nib.as_closest_canonical(img)
    data = canonical.get_fdata(dtype=np.float32)
    affine = canonical.affine
    header = canonical.header
    spacing = tuple(float(x) for x in header.get_zooms()[:3])
    orientation = "".join(nib.aff2axcodes(affine))

    metadata = VolumeMetadata(
        affine=affine,
        spacing=spacing,
        original_shape=tuple(int(x) for x in data.shape),
        orientation=orientation,
        header=header,
    )
    return data, metadata


def _load_with_simpleitk(path: Path) -> Tuple[np.ndarray, VolumeMetadata]:  # pragma: no cover
    if sitk is None:
        raise RuntimeError(
            f"SimpleITK is required to load {path.suffix} files. "
            "Install it via `pip install SimpleITK`."
        )
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    image = reader.Execute()
    data = sitk.GetArrayFromImage(image).astype(np.float32)
    data = np.transpose(data, (2, 1, 0))  # SimpleITK uses z, y, x ordering
    spacing = tuple(float(x) for x in image.GetSpacing())

    metadata = VolumeMetadata(
        affine=None,
        spacing=spacing,
        original_shape=tuple(int(x) for x in data.shape),
        orientation=None,
        header=None,
    )
    return data, metadata


def load_medical_volume(
    path: Path | str,
    *,
    ensure_nd: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a medical volume using the appropriate backend.

    Args:
        path: Path to the medical image volume.
        ensure_nd: Enforce that the returned array is 3D. If the array is 2D,
            a singleton axis will be appended.

    Returns:
        Tuple of volume data and metadata dictionary.
    """
    resolved = _resolve_path(path)
    suffix = resolved.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS and resolved.suffixes[-2:] != [".nii", ".gz"]:
        raise ValueError(f"Unsupported medical volume extension: {resolved.suffix}")

    if resolved.suffixes[-2:] == [".nii", ".gz"] or suffix in {".nii"}:
        array, metadata = _load_with_nibabel(resolved)
    else:
        array, metadata = _load_with_simpleitk(resolved)

    if ensure_nd and array.ndim == 2:
        array = array[np.newaxis, ...]

    meta_dict: Dict[str, Any] = {
        "affine": metadata.affine,
        "spacing": metadata.spacing,
        "original_shape": metadata.original_shape,
        "orientation": metadata.orientation,
        "header": metadata.header,
        "path": str(resolved),
    }
    return array, meta_dict

