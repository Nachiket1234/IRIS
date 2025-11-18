"""
Train IRIS on higher quality synthetic medical images with better resolution.
"""
import json
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F

from iris.model import IrisModel
from iris.training import (
    EpisodicTrainingConfig,
    EpisodicTrainer,
    set_global_seed,
)


class HighQualityMedicalDataset(torch.utils.data.Dataset):
    """
    High-quality synthetic 3D medical volume dataset with better resolution and contrast.
    """

    def __init__(
        self,
        length: int,
        classes: int = 3,
        volume_shape=(96, 96, 96),  # Larger for better quality
        modality: str = "CT",
        seed_offset: int = 0,
    ) -> None:
        self.length = length
        self.classes = classes
        self.volume_shape = volume_shape
        self.modality = modality
        self.seed_offset = seed_offset
        self.dataset_name = "high_quality_medical"
        self.split = type("Split", (), {"value": "train"})()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        depth, height, width = self.volume_shape
        g = torch.Generator().manual_seed((idx + self.seed_offset) * 1000 + 42)

        # Create base image with realistic intensity distribution
        if self.modality == "CT":
            # CT: More realistic Hounsfield units with better contrast
            base_intensity = torch.randn(1, depth, height, width, generator=g) * 150 + 0
            base_intensity = torch.clamp(base_intensity, -1000, 1000)
        else:  # MRI
            base_intensity = torch.rand(1, depth, height, width, generator=g)

        mask = torch.zeros(depth, height, width, dtype=torch.int64)

        # Create larger, more visible organ-like structures
        for cls in range(1, self.classes + 1):
            # Random center
            center_z = int(torch.randint(10, depth - 10, (1,), generator=g).item())
            center_y = int(torch.randint(10, height - 10, (1,), generator=g).item())
            center_x = int(torch.randint(10, width - 10, (1,), generator=g).item())

            # Larger, more visible structures
            radius_z = int(torch.randint(12, 20, (1,), generator=g).item())
            radius_y = int(torch.randint(12, 20, (1,), generator=g).item())
            radius_x = int(torch.randint(12, 20, (1,), generator=g).item())

            # Create ellipsoid mask with smoother boundaries
            z_coords, y_coords, x_coords = torch.meshgrid(
                torch.arange(depth, dtype=torch.float32),
                torch.arange(height, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32),
                indexing="ij",
            )

            dist = (
                ((z_coords - center_z) / radius_z) ** 2
                + ((y_coords - center_y) / radius_y) ** 2
                + ((x_coords - center_x) / radius_x) ** 2
            )

            # Smoother boundaries using sigmoid-like transition
            organ_mask = dist <= 1.2
            mask[organ_mask] = cls

            # Add distinct intensity for each organ with better contrast
            if self.modality == "CT":
                # Different HU ranges for different organs
                organ_intensities = {1: 50, 2: 150, 3: 250}  # Distinct HU values
                organ_intensity = organ_intensities.get(cls, torch.randn(1, generator=g).item() * 100 + 100)
            else:
                organ_intensity = (cls / self.classes) * 0.4 + 0.3

            base_intensity[0, organ_mask] = organ_intensity

        # Add some noise for realism
        noise = torch.randn(1, depth, height, width, generator=g) * 10
        base_intensity = base_intensity + noise

        # Normalize to [0, 1] for model input
        if self.modality == "CT":
            image = (base_intensity + 1000) / 2000.0
        else:
            image = base_intensity

        image = torch.clamp(image, 0.0, 1.0)

        # Ensure at least one foreground class exists
        if mask.sum() == 0:
            mask[depth // 2, height // 2, width // 2] = 1

        return {
            "image": image,
            "mask": mask,
            "meta": {
                "index": idx,
                "modality": self.modality,
                "classes": list(range(1, self.classes + 1)),
            },
        }


def main():
    output_dir = Path("demo_outputs/high_quality_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "training_results.txt"

    if results_file.exists():
        results_file.unlink()

    def log(msg: str):
        print(msg)
        with results_file.open("a", encoding="utf-8") as fp:
            fp.write(msg + "\n")

    log("=" * 80)
    log("IRIS Training on High-Quality Synthetic 3D Medical Dataset")
    log("=" * 80)
    log("")

    set_global_seed(42)

    # Model with better capacity
    model = IrisModel(
        in_channels=1,
        base_channels=16,
        num_query_tokens=4,
        num_attention_heads=4,
        volume_shape=(96, 96, 96),
        use_memory_bank=False,
        memory_momentum=0.999,
    )

    log(f"Model initialized:")
    log(f"  - Base channels: 16")
    log(f"  - Volume shape: 96×96×96 (higher resolution)")
    log(f"  - Memory bank: Disabled")
    log("")

    # Training dataset
    train_dataset = HighQualityMedicalDataset(
        length=30,
        classes=3,
        volume_shape=(96, 96, 96),
        modality="CT",
        seed_offset=0,
    )

    log(f"Training dataset: {len(train_dataset)} volumes")
    log("")

    # More training iterations for better quality
    train_config = EpisodicTrainingConfig(
        total_iterations=50,  # More iterations
        batch_size=2,
        base_learning_rate=1e-3,
        weight_decay=1e-5,
        warmup_iterations=5,
        log_every=10,
        eval_every=25,
        checkpoint_every=50,
        checkpoint_dir=str(output_dir / "checkpoints"),
        volume_size=(96, 96, 96),
        augmentation_kwargs={
            "crop_size": (80, 80, 80),
            "intensity_shift": 0.05,
            "intensity_scale": 0.1,
            "rotation_range": (5.0, 5.0, 5.0),
            "translation_range": (5.0, 5.0, 5.0),
        },
        random_class_drop_prob=0.1,
        random_seed=42,
    )

    log("Starting training...")
    log(f"  - Total iterations: {train_config.total_iterations}")
    log(f"  - Batch size: {train_config.batch_size}")
    log(f"  - Learning rate: {train_config.base_learning_rate}")
    log("")

    trainer = EpisodicTrainer(
        model, [train_dataset], train_config, device="cpu"
    )
    trainer.train()

    log("")
    log("Training completed!")
    log("")

    checkpoint_path = trainer.save_checkpoint()
    log(f"Checkpoint saved: {checkpoint_path}")
    log("")
    log("=" * 80)
    log("Training Complete!")
    log("=" * 80)


if __name__ == "__main__":
    main()

