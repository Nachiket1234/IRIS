"""High-level IRIS core model assembling encoder, task encoder, and decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .decoder import MaskDecoder
from .encoder import Medical3DUNetEncoder
from .memory import ClassMemoryBank
from .task_encoding import TaskEncodingModule

if TYPE_CHECKING:
    from .tuning import InContextTuner


class IrisModel(nn.Module):
    """
    Implements the IRIS architecture for episodic medical image segmentation.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        base_channels: int = 32,
        num_query_tokens: int = 8,
        num_attention_heads: int = 8,
        volume_shape: Tuple[int, int, int] = (128, 128, 128),
        use_memory_bank: bool = True,
        memory_momentum: float = 0.999,
    ) -> None:
        super().__init__()
        self.volume_shape = volume_shape
        self.encoder = Medical3DUNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            stages=4,
        )
        encoder_channels = [
            base_channels * (2 ** i) for i in range(5)
        ]
        self.task_encoder = TaskEncodingModule(
            feature_channels=encoder_channels[-1],
            num_query_tokens=num_query_tokens,
            num_attention_heads=num_attention_heads,
            downsample_ratio=self.encoder.downsample_ratio,
        )
        self.mask_decoder = MaskDecoder(
            encoder_channels=encoder_channels,
            num_query_tokens=num_query_tokens,
            num_attention_heads=num_attention_heads,
            final_upsample_target=volume_shape,
        )
        self.memory_bank: Optional[ClassMemoryBank] = (
            ClassMemoryBank(momentum=memory_momentum) if use_memory_bank else None
        )

    def encode_support(
        self,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoder_out = self.encoder(support_images)
        return self.task_encoder(encoder_out.features, support_masks)

    def update_memory_bank(
        self,
        task_embeddings: torch.Tensor,
        class_ids: Sequence[Sequence[int]] | Sequence[int],
    ) -> None:
        if self.memory_bank is None:
            return
        self.memory_bank.update_episode(task_embeddings.detach(), class_ids)

    def retrieve_memory_embeddings(
        self,
        class_ids: Sequence[int],
        *,
        fallback: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is disabled for this model instance.")
        return self.memory_bank.retrieve(
            class_ids,
            default=fallback,
            device=device,
            dtype=dtype,
        )

    def create_in_context_tuner(
        self,
        *,
        lr: float = 1e-3,
        steps: int = 20,
    ) -> "InContextTuner":
        from .tuning import InContextTuner

        return InContextTuner(
            model=self,
            lr=lr,
            steps=steps,
            memory_bank=self.memory_bank,
        )

    def forward(
        self,
        query_images: torch.Tensor,
        task_embeddings: torch.Tensor,
        *,
        skip_connections: Sequence[torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_out = self.encoder(query_images)
        skips = skip_connections or encoder_out.skip_connections
        decoder_out = self.mask_decoder(
            encoder_out.features,
            skips,
            task_embeddings,
        )
        return {
            "logits": decoder_out.logits,
            "tokens": decoder_out.updated_tokens,
            "skip_connections": encoder_out.skip_connections,
        }


