import torch

from iris.model import ClassMemoryBank, InContextTuner, IrisModel


def test_memory_bank_ema_update():
    bank = ClassMemoryBank(momentum=0.5)
    first = torch.ones(3, 4)
    second = torch.zeros(3, 4)

    bank.update(1, first)
    updated = bank.update(1, second)

    expected = 0.5 * first + 0.5 * second
    assert torch.allclose(updated, expected)
    assert torch.allclose(bank.get(1), expected)


def test_memory_bank_episode_update_and_retrieve():
    bank = ClassMemoryBank(momentum=0.0)
    embeddings = torch.randn(2, 2, 3, 4)
    class_ids = [[0, 5], [3, 7]]

    bank.update_episode(embeddings, class_ids)

    retrieved = bank.retrieve([5, 7])
    assert retrieved.shape == (2, 3, 4)
    assert torch.allclose(retrieved[0], embeddings[0, 1])
    assert torch.allclose(retrieved[1], embeddings[1, 1])


def test_in_context_tuner_updates_embeddings_and_memory():
    torch.manual_seed(0)
    model = IrisModel(
        in_channels=1,
        base_channels=8,
        num_query_tokens=4,
        num_attention_heads=2,
        volume_shape=(32, 32, 32),
        memory_momentum=0.0,
    )

    tuner = InContextTuner(model=model, lr=1e-2, steps=1)

    query_images = torch.randn(1, 1, 32, 32, 32)
    query_masks = torch.randint(0, 2, (1, 1, 32, 32, 32)).float()

    num_tokens = tuner.model.task_encoder.query_tokens.shape[1]
    feature_dim = tuner.model.task_encoder.query_tokens.shape[2]
    initial_embeddings = torch.randn(1, 1, num_tokens + 1, feature_dim)
    tuned = tuner.tune(
        query_images,
        query_masks,
        initial_embeddings,
        class_ids=[[1]],
        steps=1,
    )

    assert tuned.shape == initial_embeddings.shape
    assert model.memory_bank is not None
    retrieved = model.memory_bank.get(1)
    assert retrieved is not None
    assert retrieved.shape == tuned[0, 0].shape


