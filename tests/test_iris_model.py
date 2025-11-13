import torch

from iris.model import IrisModel


def test_task_encoding_shapes():
    model = IrisModel(volume_shape=(64, 64, 64))
    support = torch.randn(1, 1, 64, 64, 64)
    mask = torch.randint(0, 2, (1, 2, 64, 64, 64), dtype=torch.float32)

    encoder_out = model.encoder(support)
    task_info = model.task_encoder(encoder_out.features, mask)

    assert task_info["task_embeddings"].shape == (1, 2, 9, model.encoder.output_channels)
    assert task_info["foreground_embeddings"].shape == (1, 2, 1, model.encoder.output_channels)
    assert task_info["context_tokens"].shape == (1, 2, 8, model.encoder.output_channels)


def test_full_forward_pass_multiclass():
    model = IrisModel(volume_shape=(64, 64, 64))
    support = torch.randn(1, 1, 64, 64, 64)
    query = torch.randn(1, 1, 64, 64, 64)
    mask = torch.randint(0, 2, (1, 2, 64, 64, 64), dtype=torch.float32)

    support_enc = model.encoder(support)
    task_info = model.task_encoder(support_enc.features, mask)

    outputs = model(query, task_info["task_embeddings"])
    assert outputs["logits"].shape == (1, 2, 64, 64, 64)
    assert outputs["tokens"].shape[0] == 1
    assert outputs["tokens"].shape[1] == 2

