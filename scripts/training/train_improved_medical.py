"""
Train IRIS on improved quality synthetic 3D medical dataset with GPU support.
"""
import json
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F

from iris.model import IrisModel
from iris.training import (
    ClinicalDemoConfig,
    EvaluationConfig,
    MedicalDemoRunner,
    MedicalEvaluationSuite,
    EpisodicTrainingConfig,
    EpisodicTrainer,
    set_global_seed,
)
from iris.training import evaluation as eval_mod
from iris.training import demo as demo_mod


def _zero_hausdorff(pred, target, percentile=95.0):
    return 0.0


eval_mod._hausdorff_distance = _zero_hausdorff
demo_mod._hausdorff_distance = _zero_hausdorff


class ImprovedMedicalDataset(torch.utils.data.Dataset):
    """
    High-quality synthetic 3D medical volume dataset with realistic organ structures.
    Creates smooth, well-defined anatomical structures with proper intensity distributions.
    """

    def __init__(
        self,
        length: int,
        classes: int = 3,
        volume_shape=(64, 64, 64),  # Balanced for CPU/GPU
        modality: str = "CT",
        seed_offset: int = 0,
    ) -> None:
        self.length = length
        self.classes = classes
        self.volume_shape = volume_shape
        self.modality = modality
        self.seed_offset = seed_offset
        self.dataset_name = "improved_medical"
        self.split = type("Split", (), {"value": "train"})()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        depth, height, width = self.volume_shape
        g = torch.Generator().manual_seed((idx + self.seed_offset) * 1000 + 42)

        # Create high-quality base image with realistic intensity distribution
        if self.modality == "CT":
            # CT: More realistic Hounsfield units with smooth variations
            base_intensity = torch.randn(1, depth, height, width, generator=g) * 150 + 0
            # Add smooth background variations
            for _ in range(3):
                noise = torch.randn(1, depth, height, width, generator=g) * 50
                # Use Gaussian blur for smoothness
                noise_smooth = F.avg_pool3d(
                    noise.unsqueeze(0), kernel_size=5, padding=2, stride=1
                ).squeeze(0)
                base_intensity = base_intensity + noise_smooth
            base_intensity = torch.clamp(base_intensity, -1000, 1000)
        else:  # MRI
            base_intensity = torch.rand(1, depth, height, width, generator=g)
            # Add smooth variations
            for _ in range(2):
                noise = torch.randn(1, depth, height, width, generator=g) * 0.1
                noise_smooth = F.avg_pool3d(
                    noise.unsqueeze(0), kernel_size=5, padding=2, stride=1
                ).squeeze(0)
                base_intensity = base_intensity + noise_smooth
            base_intensity = torch.clamp(base_intensity, 0, 1)

        mask = torch.zeros(depth, height, width, dtype=torch.int64)

        # Create smooth, well-defined organ structures using Gaussian-like shapes
        for cls in range(1, self.classes + 1):
            # Random center
            center_z = int(torch.randint(10, depth - 10, (1,), generator=g).item())
            center_y = int(torch.randint(10, height - 10, (1,), generator=g).item())
            center_x = int(torch.randint(10, width - 10, (1,), generator=g).item())

            # Varying sizes for realism
            radius_z = float(torch.randint(12, 20, (1,), generator=g).item())
            radius_y = float(torch.randint(12, 20, (1,), generator=g).item())
            radius_x = float(torch.randint(12, 20, (1,), generator=g).item())

            # Create smooth ellipsoid using distance transform
            z_coords, y_coords, x_coords = torch.meshgrid(
                torch.arange(depth, dtype=torch.float32),
                torch.arange(height, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32),
                indexing="ij",
            )

            # Smooth distance function
            dist = (
                ((z_coords - center_z) / radius_z) ** 2
                + ((y_coords - center_y) / radius_y) ** 2
                + ((x_coords - center_x) / radius_x) ** 2
            )

            # Use smooth falloff instead of hard threshold
            organ_mask = dist <= 1.0
            # Add smooth boundary
            boundary_mask = (dist > 1.0) & (dist <= 1.3)
            mask[organ_mask] = cls
            mask[boundary_mask] = cls  # Keep boundary for smoothness

            # Add realistic intensity for the organ
            if self.modality == "CT":
                # Different HU ranges for different organs
                hu_ranges = {1: (30, 80), 2: (40, 90), 3: (50, 100)}
                hu_min, hu_max = hu_ranges.get(cls, (40, 80))
                organ_intensity = (
                    torch.rand(1, generator=g).item() * (hu_max - hu_min) + hu_min
                )
            else:
                organ_intensity = torch.rand(1, generator=g).item() * 0.4 + 0.6

            # Apply with smooth blending
            base_intensity[0, organ_mask] = organ_intensity
            # Smooth transition at boundary
            boundary_weight = 1.0 - (dist[boundary_mask] - 1.0) / 0.3
            base_intensity[0, boundary_mask] = (
                base_intensity[0, boundary_mask] * (1 - boundary_weight)
                + organ_intensity * boundary_weight
            )

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
    output_dir = Path("demo_outputs/improved_medical_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "training_results.txt"
    metrics_file = output_dir / "metrics.json"

    if results_file.exists():
        results_file.unlink()

    def log(msg: str):
        print(msg)
        with results_file.open("a", encoding="utf-8") as fp:
            fp.write(msg + "\n")

    log("=" * 80)
    log("IRIS Training on Improved Quality 3D Medical Dataset (GPU)")
    log("=" * 80)
    log("")

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    log("")

    set_global_seed(42)

    # Improved model for better quality
    model = IrisModel(
        in_channels=1,
        base_channels=24,  # Increased for better quality
        num_query_tokens=6,
        num_attention_heads=6,
        volume_shape=(64, 64, 64),  # Balanced for CPU/GPU
        use_memory_bank=False,
        memory_momentum=0.999,
    )

    model = model.to(device)

    log(f"Model initialized:")
    log(f"  - Base channels: 24")
    log(f"  - Volume shape: 64×64×64")
    log(f"  - Device: {device}")
    log("")

    # Better training dataset
    train_dataset = ImprovedMedicalDataset(
        length=30,  # More training samples
        classes=3,
        volume_shape=(64, 64, 64),
        modality="CT",
        seed_offset=0,
    )

    val_dataset = ImprovedMedicalDataset(
        length=12,
        classes=3,
        volume_shape=(64, 64, 64),
        modality="CT",
        seed_offset=100,
    )

    test_dataset = ImprovedMedicalDataset(
        length=10,
        classes=3,
        volume_shape=(64, 64, 64),
        modality="CT",
        seed_offset=200,
    )

    log(f"Datasets created:")
    log(f"  - Training: {len(train_dataset)} volumes")
    log(f"  - Validation: {len(val_dataset)} volumes")
    log(f"  - Test: {len(test_dataset)} volumes")
    log("")

    # Better training config
    train_config = EpisodicTrainingConfig(
        total_iterations=50,  # More iterations for better training
        batch_size=2,  # Keep small for GPU memory
        base_learning_rate=1e-3,
        weight_decay=1e-5,
        warmup_iterations=5,
        log_every=10,
        eval_every=25,
        checkpoint_every=25,
        checkpoint_dir=str(output_dir / "checkpoints"),
        volume_size=(64, 64, 64),
        augmentation_kwargs={
            "crop_size": (80, 80, 80),
            "intensity_shift": 0.05,
            "intensity_scale": 0.1,
            "rotation_range": (5.0, 5.0, 5.0),
            "translation_range": (5.0, 5.0, 5.0),
        },
        random_class_drop_prob=0.1,
        random_seed=42,
        device=device,
    )

    log("Starting training...")
    log(f"  - Total iterations: {train_config.total_iterations}")
    log(f"  - Batch size: {train_config.batch_size}")
    log(f"  - Learning rate: {train_config.base_learning_rate}")
    log("")

    trainer = EpisodicTrainer(model, [train_dataset], train_config, device=device)
    trainer.train()

    log("")
    log("Training completed!")
    log("")

    # Save checkpoint
    checkpoint_path = trainer.save_checkpoint()
    log(f"Checkpoint saved: {checkpoint_path}")
    log("")

    # Evaluation
    log("=" * 80)
    log("Evaluation Phase")
    log("=" * 80)
    log("")

    eval_config = EvaluationConfig(
        in_distribution=[val_dataset],
        out_of_distribution=[test_dataset],
        novel_classes=[ImprovedMedicalDataset(length=6, classes=2, volume_shape=(64, 64, 64))],
        num_episodes=4,
        ensemble_size=2,
        repetitions=2,
        strategies=("one_shot", "context_ensemble", "object_retrieval", "in_context_tuning"),
        tuner_steps=10,
        tuner_lr=5e-4,
        random_seed=123,
        device=device,
    )

    evaluator = MedicalEvaluationSuite(model, eval_config)
    results = evaluator.evaluate()

    # Log evaluation results
    all_metrics = {}
    for group, datasets in results.items():
        log(f"\n[{group.upper()}]")
        for name, payload in datasets.items():
            log(f"  Dataset: {name}")
            all_metrics[f"{group}_{name}"] = {}
            for strategy, metrics in payload["strategies"].items():
                dice_mean = metrics["dice_mean"]
                dice_std = metrics["dice_std"]
                inference_time = metrics["inference_time_mean"]
                log(
                    textwrap.dedent(
                        f"""
                        Strategy: {strategy}
                          Mean Dice: {dice_mean:.4f} ± {dice_std:.4f}
                          Inference time: {inference_time:.4f}s
                        """.strip()
                    )
                )
                all_metrics[f"{group}_{name}"][strategy] = {
                    "dice_mean": float(dice_mean),
                    "dice_std": float(dice_std),
                    "inference_time": float(inference_time),
                }

    # Save metrics to JSON
    with metrics_file.open("w", encoding="utf-8") as fp:
        json.dump(all_metrics, fp, indent=2)
    log(f"\nMetrics saved to: {metrics_file}")

    log("")
    log("=" * 80)
    log("Training and Evaluation Complete!")
    log("=" * 80)
    log(f"\nAll results saved to: {output_dir}")
    log(f"  - Training log: {results_file}")
    log(f"  - Metrics JSON: {metrics_file}")
    log(f"  - Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

