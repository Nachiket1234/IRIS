import torch

from iris.model.core import IrisModel
from iris.model.decoder import MaskDecoder
from iris.model.encoder import Medical3DUNetEncoder
from iris.model.task_encoding import TaskEncodingModule


def _random_masks(batch: int, classes: int, volume_shape: tuple[int, int, int]) -> torch.Tensor:
    D, H, W = volume_shape
    masks = torch.zeros(batch, classes, D, H, W, dtype=torch.float32)
    for b in range(batch):
        for k in range(classes):
            z_start = (k * D) // (2 * classes)
            z_end = min(D, z_start + max(1, D // (classes + 1)))
            masks[b, k, z_start:z_end] = 1.0
    return masks


def test_medical_encoder_outputs():
    torch.manual_seed(0)
    encoder = Medical3DUNetEncoder(in_channels=1, base_channels=32, stages=4)
    x = torch.randn(1, 1, 64, 64, 64)
    out = encoder(x)
    assert out.features.shape == (1, encoder.output_channels, 4, 4, 4)
    assert len(out.skip_connections) == 4
    assert out.skip_connections[0].shape[1] == 32


def test_task_encoding_shapes():
    torch.manual_seed(0)
    encoder = Medical3DUNetEncoder(in_channels=1, base_channels=32, stages=4)
    support = torch.randn(2, 1, 64, 64, 64)
    encoder_out = encoder(support)
    masks = _random_masks(batch=2, classes=3, volume_shape=(64, 64, 64))
    task_encoder = TaskEncodingModule(
        feature_channels=encoder.output_channels,
        num_query_tokens=6,
        num_attention_heads=4,
        downsample_ratio=encoder.downsample_ratio,
    )
    result = task_encoder(encoder_out.features, masks)
    task_embeddings = result["task_embeddings"]
    assert task_embeddings.shape == (2, 3, 7, encoder.output_channels)
    assert result["foreground_embeddings"].shape == (2, 3, 1, encoder.output_channels)


def test_mask_decoder_shapes():
    torch.manual_seed(0)
    encoder = Medical3DUNetEncoder(in_channels=1, base_channels=32, stages=4)
    query = torch.randn(2, 1, 64, 64, 64)
    encoder_out = encoder(query)
    masks = _random_masks(batch=2, classes=2, volume_shape=(64, 64, 64))
    task_encoder = TaskEncodingModule(
        feature_channels=encoder.output_channels,
        num_query_tokens=5,
        num_attention_heads=5,
        downsample_ratio=encoder.downsample_ratio,
    )
    task_embeddings = task_encoder(encoder_out.features, masks)["task_embeddings"]
    decoder = MaskDecoder(
        encoder_channels=[32, 64, 128, 256, encoder.output_channels],
        num_query_tokens=5,
        num_attention_heads=5,
        final_upsample_target=(64, 64, 64),
    )
    output = decoder(
        encoder_out.features,
        encoder_out.skip_connections,
        task_embeddings,
    )
    assert output.logits.shape == (2, 2, 64, 64, 64)
    assert output.updated_tokens.shape[0] == 2


def test_iris_model_end_to_end():
    torch.manual_seed(0)
    model = IrisModel(
        in_channels=1,
        base_channels=32,
        num_query_tokens=6,
        num_attention_heads=6,
        volume_shape=(64, 64, 64),
    )
    support = torch.randn(2, 1, 64, 64, 64)
    support_masks = _random_masks(batch=2, classes=2, volume_shape=(64, 64, 64))
    task_dict = model.encode_support(support, support_masks)
    query = torch.randn(2, 1, 64, 64, 64)
    outputs = model(query, task_dict["task_embeddings"])
    assert outputs["logits"].shape == (2, 2, 64, 64, 64)
    assert "tokens" in outputs
    assert len(outputs["skip_connections"]) == 4


