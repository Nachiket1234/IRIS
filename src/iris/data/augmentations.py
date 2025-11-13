"""
Data augmentation utilities tailored for 3D medical imaging.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import affine_transform


def _random_crop(
    image: torch.Tensor,
    mask: Optional[torch.Tensor],
    crop_size: Tuple[int, int, int],
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    _, depth, height, width = image.shape
    cz, cy, cx = crop_size
    if cz > depth or cy > height or cx > width:
        return image, mask

    z = rng.integers(0, depth - cz + 1)
    y = rng.integers(0, height - cy + 1)
    x = rng.integers(0, width - cx + 1)

    cropped_image = image[:, z : z + cz, y : y + cy, x : x + cx]
    if mask is None:
        cropped_mask = None
    elif mask.ndim == 4:
        cropped_mask = mask[:, z : z + cz, y : y + cy, x : x + cx]
    else:
        cropped_mask = mask[z : z + cz, y : y + cy, x : x + cx]
    return cropped_image, cropped_mask


def _random_flip(
    image: torch.Tensor,
    mask: Optional[torch.Tensor],
    flip_axes: Tuple[bool, bool, bool],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if flip_axes[0]:
        image = torch.flip(image, dims=[1])
        if mask is not None:
            dim = 1 if mask.ndim == 4 else 0
            mask = torch.flip(mask, dims=[dim])
    if flip_axes[1]:
        image = torch.flip(image, dims=[2])
        if mask is not None:
            dim = 2 if mask.ndim == 4 else 1
            mask = torch.flip(mask, dims=[dim])
    if flip_axes[2]:
        image = torch.flip(image, dims=[3])
        if mask is not None:
            dim = 3 if mask.ndim == 4 else 2
            mask = torch.flip(mask, dims=[dim])
    return image, mask


def _random_intensity_shift(
    image: torch.Tensor,
    max_shift: float,
    max_scale: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    shift = rng.uniform(-max_shift, max_shift)
    scale = rng.uniform(1.0 - max_scale, 1.0 + max_scale)
    augmented = image * scale + shift
    return augmented.clamp(0.0, 1.0)


def _affine_matrix(
    angles: Tuple[float, float, float],
    scales: Tuple[float, float, float],
    translations: Tuple[float, float, float],
) -> np.ndarray:
    rx, ry, rz = angles
    sx, sy, sz = scales
    tx, ty, tz = translations

    cx, sx_sin = np.cos(rx), np.sin(rx)
    cy, sy_sin = np.cos(ry), np.sin(ry)
    cz, sz_sin = np.cos(rz), np.sin(rz)

    rot_x = np.array(
        [[1, 0, 0], [0, cx, -sx_sin], [0, sx_sin, cx]], dtype=np.float32
    )
    rot_y = np.array(
        [[cy, 0, sy_sin], [0, 1, 0], [-sy_sin, 0, cy]], dtype=np.float32
    )
    rot_z = np.array(
        [[cz, -sz_sin, 0], [sz_sin, cz, 0], [0, 0, 1]], dtype=np.float32
    )

    rotation = rot_z @ rot_y @ rot_x
    scale_mat = np.diag([sx, sy, sz]).astype(np.float32)
    affine = rotation @ scale_mat

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = affine
    matrix[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return matrix


def _apply_affine(
    tensor: torch.Tensor,
    matrix: np.ndarray,
    order: int,
) -> torch.Tensor:
    array = tensor.cpu().numpy()
    had_channel = False
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
        had_channel = True
    matrix3x3 = matrix[:3, :3]
    offset = matrix[:3, 3]
    transformed = affine_transform(
        array,
        matrix3x3,
        offset=offset,
        order=order,
        mode="nearest",
        cval=0.0,
    )
    if had_channel:
        transformed_tensor = torch.from_numpy(transformed).unsqueeze(0).type_as(tensor)
    else:
        transformed_tensor = torch.from_numpy(transformed).type_as(tensor)
    return transformed_tensor


class MedicalAugmentation:
    """
    Composable augmentation callable for 3D medical volumes.
    """

    def __init__(
        self,
        *,
        crop_size: Optional[Tuple[int, int, int]] = (112, 112, 112),
        flip_prob: float = 0.5,
        intensity_shift: float = 0.1,
        intensity_scale: float = 0.15,
        rotation_range: Tuple[float, float, float] = (10.0, 10.0, 10.0),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_range: Tuple[float, float, float] = (5.0, 5.0, 5.0),
        random_class_drop_prob: float = 0.2,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.intensity_shift = intensity_shift
        self.intensity_scale = intensity_scale
        self.rotation_range = tuple(np.radians(x) for x in rotation_range)
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.random_class_drop_prob = random_class_drop_prob
        self.rng = rng or np.random.default_rng()

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image: torch.Tensor = sample["image"]
        mask: Optional[torch.Tensor] = sample.get("mask")

        if self.crop_size is not None:
            image, mask = _random_crop(image, mask, self.crop_size, self.rng)

        flip_axes = tuple(self.rng.random(3) < self.flip_prob)
        image, mask = _random_flip(image, mask, flip_axes)

        image = _random_intensity_shift(
            image, max_shift=self.intensity_shift, max_scale=self.intensity_scale, rng=self.rng
        )

        angles = tuple(self.rng.uniform(-r, r) for r in self.rotation_range)
        scales = tuple(self.rng.uniform(*self.scale_range) for _ in range(3))
        translations = tuple(
            self.rng.uniform(-t, t) for t in self.translation_range
        )
        affine_matrix = _affine_matrix(angles, scales, translations)
        image = _apply_affine(image, affine_matrix, order=3)
        if mask is not None:
            mask = _apply_affine(mask.float(), affine_matrix, order=0).to(mask.dtype)

        if mask is not None and self.random_class_drop_prob > 0.0:
            mask = self._random_class_drop(mask)

        sample["image"] = image
        sample["mask"] = mask
        sample["meta"]["augmentation"] = {
            "flip_axes": flip_axes,
            "angles": angles,
            "scales": scales,
            "translations": translations,
        }
        return sample

    def _random_class_drop(self, mask: torch.Tensor) -> torch.Tensor:
        unique_classes = torch.unique(mask)
        unique_classes = unique_classes[unique_classes != 0]
        if len(unique_classes) == 0:
            return mask
        if self.rng.random() < self.random_class_drop_prob:
            drop_class = int(self.rng.choice(unique_classes.cpu().numpy()))
            mask = mask.clone()
            mask[mask == drop_class] = 0
        return mask

