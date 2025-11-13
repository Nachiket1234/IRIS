import torch

from iris.model import IrisModel
from iris.training import EpisodicTrainer, EpisodicTrainingConfig, MedicalEvaluationSuite, EvaluationConfig, set_global_seed


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, length: int, classes: int = 2, volume_shape=(32, 32, 32)) -> None:
        self.length = length
        self.classes = classes
        self.volume_shape = volume_shape
        self.dataset_name = "synthetic"
        self.split = type("Split", (), {"value": "train"})()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        image = torch.rand(1, *self.volume_shape)
        mask = torch.zeros(self.volume_shape, dtype=torch.int64)
        for cls in range(1, self.classes + 1):
            region = torch.randint(0, 2, self.volume_shape)
            mask[region.bool()] = cls
        if mask.sum() == 0:
            mask[0, 0, 0] = 1
        return {"image": image, "mask": mask, "meta": {"index": idx}}


def test_lamb_training_step(tmp_path):
    set_global_seed(0)
    model = IrisModel(
        in_channels=1,
        base_channels=8,
        num_query_tokens=4,
        num_attention_heads=2,
        volume_shape=(32, 32, 32),
    )
    dataset = SyntheticDataset(length=10, classes=1, volume_shape=(32, 32, 32))

    config = EpisodicTrainingConfig(
        total_iterations=2,
        batch_size=2,
        base_learning_rate=1e-3,
        warmup_iterations=1,
        checkpoint_dir=tmp_path,
        log_every=1,
        eval_every=10,
        checkpoint_every=10,
        augmentation_kwargs={"crop_size": (24, 24, 24)},
    )

    trainer = EpisodicTrainer(model, [dataset], config)
    trainer.train()

    assert trainer.iteration == config.total_iterations


def test_evaluation_suite_runs(tmp_path):
    set_global_seed(0)
    model = IrisModel(
        in_channels=1,
        base_channels=8,
        num_query_tokens=4,
        num_attention_heads=2,
        volume_shape=(32, 32, 32),
    )
    dataset = SyntheticDataset(length=5, classes=1, volume_shape=(32, 32, 32))

    eval_config = EvaluationConfig(
        in_distribution=[dataset],
        out_of_distribution=[],
        novel_classes=[],
        num_episodes=2,
        ensemble_size=1,
        strategies=("one_shot",),
    )

    evaluator = MedicalEvaluationSuite(model, eval_config)
    results = evaluator.evaluate()
    assert "in_distribution" in results
    assert "synthetic" in results["in_distribution"]
    strategies = results["in_distribution"]["synthetic"]["strategies"]
    assert "one_shot" in strategies
    assert "dice_mean" in strategies["one_shot"]


