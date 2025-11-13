"""Utility ops for 3D medical imaging models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pixel_shuffle_3d(x: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    """
    3D extension of torch.nn.PixelShuffle.

    Args:
        x: Tensor of shape (B, C, D, H, W).
        upscale_factor: Spatial up-scaling factor ``r``. ``C`` must be divisible
            by ``r ** 3``.
    Returns:
        Tensor of shape (B, C / r^3, D * r, H * r, W * r).
    """
    if upscale_factor == 1:
        return x
    b, c, d, h, w = x.shape
    r = upscale_factor
    r3 = r ** 3
    if c % r3 != 0:
        raise ValueError(
            f"Channel dimension {c} is not divisible by upscale_factor ** 3 ({r3})."
        )
    x = x.view(b, c // r3, r, r, r, d, h, w)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return x.view(b, c // r3, d * r, h * r, w * r)


def pixel_unshuffle_3d(x: torch.Tensor, downscale_factor: int) -> torch.Tensor:
    """
    Inverse of :func:`pixel_shuffle_3d`.

    Args:
        x: Tensor of shape (B, C, D, H, W).
        downscale_factor: Spatial down-scaling factor ``r``.
    Returns:
        Tensor of shape (B, C * r^3, D / r, H / r, W / r).
    """
    if downscale_factor == 1:
        return x
    b, c, d, h, w = x.shape
    r = downscale_factor
    if d % r != 0 or h % r != 0 or w % r != 0:
        raise ValueError(
            f"Spatial dims {(d, h, w)} are not divisible by downscale_factor {r}."
        )
    x = x.view(b, c, d // r, r, h // r, r, w // r, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    r3 = r ** 3
    return x.view(b, c * r3, d // r, h // r, w // r)


def upsample_to_reference(
    features: torch.Tensor,
    reference: torch.Tensor,
    mode: str = "trilinear",
) -> torch.Tensor:
    """
    Upsample ``features`` to match the spatial size of ``reference``.

    Args:
        features: Tensor of shape (B, C, d, h, w).
        reference: Tensor whose last three dims provide the target size.
        mode: Interpolation mode, defaults to ``trilinear``.
    """
    target_size = reference.shape[-3:]
    return F.interpolate(features, size=target_size, mode=mode, align_corners=False)


