"""Medical imaging optimised 3D UNet encoder with residual downsampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class ResidualBlock(nn.Module):
    """Residual block tailored for 3D medical imaging volumes."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        if in_channels != out_channels or stride != 1:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity
        out = self.act(out)
        return out


@dataclass
class EncoderOutput:
    features: torch.Tensor
    skip_connections: Sequence[torch.Tensor]


class Medical3DUNetEncoder(nn.Module):
    """
    Four-stage residual 3D UNet encoder tuned for 128^3 medical volumes.

    The encoder yields the deepest features together with multi-resolution skip
    connections for the decoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        stages: int = 4,
    ) -> None:
        super().__init__()
        if stages != 4:
            raise ValueError("The encoder is specified for exactly four downsampling stages.")

        channels: List[int] = [base_channels * (2 ** i) for i in range(stages + 1)]

        self.stem = ResidualBlock(in_channels, channels[0], stride=1)

        self.down_blocks = nn.ModuleList()
        for idx in range(stages):
            block = ResidualBlock(
                channels[idx],
                channels[idx + 1],
                stride=2,
            )
            self.down_blocks.append(block)

    @property
    def output_channels(self) -> int:
        downsampled = next(reversed(self.down_blocks))
        return downsampled.conv2.out_channels  # type: ignore[attr-defined]

    @property
    def downsample_ratio(self) -> int:
        return 2 ** len(self.down_blocks)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        skips: List[torch.Tensor] = []
        out = self.stem(x)
        skips.append(out)
        for block in self.down_blocks:
            out = block(out)
            skips.append(out)

        features = skips[-1]
        return EncoderOutput(features=features, skip_connections=skips[:-1])


