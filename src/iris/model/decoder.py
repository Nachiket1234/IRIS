"""Mask decoding module implementing Section 3.2.2 of the IRIS paper."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResidualBlock


class FiLMLayer(nn.Module):
    """Feature-wise modulators driven by task embeddings."""

    def __init__(self, channels: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, channels * 2)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.linear(embedding).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class DecoderStage(nn.Module):
    """Single upsampling stage with skip connection fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ResidualBlock(out_channels + skip_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return x


class BidirectionalCrossAttention(nn.Module):
    """Cross-attention exchanging information between query features and task tokens."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.tokens_to_features = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.features_to_tokens = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.token_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        features: torch.Tensor,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Tensor of shape (B', N, C).
            tokens: Tensor of shape (B', M, C).
        Returns:
            Tuple of updated features and tokens with the same shapes.
        """
        tokens_prime, _ = self.features_to_tokens(tokens, features, features)
        features_prime, _ = self.tokens_to_features(features, tokens_prime, tokens_prime)
        tokens_prime, _ = self.token_self_attn(tokens_prime, tokens_prime, tokens_prime)
        return features_prime, tokens_prime


@dataclass
class DecoderOutput:
    logits: torch.Tensor
    updated_tokens: torch.Tensor


class MaskDecoder(nn.Module):
    """
    Implements Equation (5)-(6) with bidirectional cross-attention and UNet decoding.
    """

    def __init__(
        self,
        *,
        encoder_channels: Sequence[int],
        num_query_tokens: int,
        num_attention_heads: int,
        final_upsample_target: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels must include at least two stages.")

        deepest_channels = encoder_channels[-1]
        skip_channels = list(reversed(encoder_channels[:-1]))

        if deepest_channels % num_attention_heads != 0:
            num_attention_heads = max(1, math.gcd(deepest_channels, num_attention_heads))

        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=deepest_channels,
            num_heads=num_attention_heads,
        )

        stages = []
        films = []
        in_channels = deepest_channels
        for skip_ch in skip_channels:
            stage = DecoderStage(
                in_channels=in_channels,
                skip_channels=skip_ch,
                out_channels=skip_ch,
            )
            stages.append(stage)
            films.append(FiLMLayer(skip_ch, deepest_channels))
            in_channels = skip_ch

        self.decoder_stages = nn.ModuleList(stages)
        self.modulators = nn.ModuleList(films)
        self.final_conv = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.final_target = final_upsample_target

    def forward(
        self,
        query_features: torch.Tensor,
        skip_connections: Sequence[torch.Tensor],
        task_embeddings: torch.Tensor,
    ) -> DecoderOutput:
        """
        Args:
            query_features: Tensor ``F_q`` of shape (B, C, d, h, w).
            skip_connections: Sequence of skip features ordered from shallow to deep.
            task_embeddings: Tensor ``T`` of shape (B, K, m+1, C).

        Returns:
            DecoderOutput containing segmentation logits of shape (B, K, D, H, W).
        """
        b, c, d, h, w = query_features.shape
        k = task_embeddings.shape[1]
        m = task_embeddings.shape[2]

        features = query_features.unsqueeze(1).expand(-1, k, -1, -1, -1, -1)
        features = features.reshape(b * k, c, d, h, w)

        tokens = task_embeddings.reshape(b * k, m, c)

        # Bidirectional cross-attention
        features_seq = features.view(b * k, c, d * h * w).transpose(1, 2)
        features_seq, tokens = self.cross_attention(features_seq, tokens)
        features = features_seq.transpose(1, 2).view(b * k, c, d, h, w)

        summary = tokens.mean(dim=1)

        skips = [s.unsqueeze(1).expand(-1, k, -1, -1, -1, -1) for s in skip_connections]
        skips = [s.reshape(b * k, s.shape[2], s.shape[3], s.shape[4], s.shape[5]) for s in skips]
        skips = list(reversed(skips))

        for stage, film, skip in zip(self.decoder_stages, self.modulators, skips):
            features = stage(features, skip)
            features = film(features, summary)

        logits = self.final_conv(features)
        logits = nn.functional.interpolate(
            logits,
            size=self.final_target,
            mode="trilinear",
            align_corners=False,
        )
        logits = logits.view(b, k, *logits.shape[-3:])
        return DecoderOutput(logits=logits, updated_tokens=tokens.view(b, k, -1, c))


