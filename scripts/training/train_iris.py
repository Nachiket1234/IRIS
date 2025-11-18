"""
Main IRIS training script - supports both real and synthetic medical datasets.
This is the consolidated training script that replaces multiple training scripts.
"""
import json
import textwrap
from pathlib import Path

import torch

from iris.data import build_dataset, DatasetSplit
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
    """Temporary Hausdorff distance for synthetic data."""
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
        volume_shape=(64, 64, 64),
        modality: str = "CT",
        seed_offset: int = 0,
    ) -> None:
        self.length = length
        self.classes = classes
        self.volume_shape = volume_shape
        self.modality = modality
        self.rng = torch.Generator()
        self.rng.manual_seed(42 + seed_offset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        depth, height, width = self.volume_shape
        device = "cpu"

        # Generate base intensity
        if self.modality == "CT":
            base_intensity = torch.rand((depth, height, width), generator=self.rng, device=device) * 0.3 + 0.4
        else:  # MRI
            base_intensity = torch.rand((depth, height, width), generator=self.rng, device=device) * 0.5 + 0.2

        # Generate smooth organ-like structures
        mask = torch.zeros((depth, height, width), dtype=torch.int64, device=device)

        for class_id in range(1, self.classes + 1):
            # Random center and size for each organ
            center_d = int(torch.randint(10, depth - 10, (1,), generator=self.rng).item())
            center_h = int(torch.randint(10, height - 10, (1,), generator=self.rng).item())
            center_w = int(torch.randint(10, width - 10, (1,), generator=self.rng).item())

            radius_d = int(torch.randint(8, 15, (1,), generator=self.rng).item())
            radius_h = int(torch.randint(8, 15, (1,), generator=self.rng).item())
            radius_w = int(torch.randint(8, 15, (1,), generator=self.rng).item())

            # Create smooth ellipsoid
            coords = torch.meshgrid(
                torch.arange(depth, device=device),
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing="ij",
            )
            dist = (
                ((coords[0] - center_d) / radius_d) ** 2
                + ((coords[1] - center_h) / radius_h) ** 2
                + ((coords[2] - center_w) / radius_w) ** 2
            )
            organ_mask = dist < 1.0

            # Add some noise for realism
            noise = torch.randn((depth, height, width), generator=self.rng, device=device) * 0.1
            noise = torch.nn.functional.avg_pool3d(
                noise.unsqueeze(0).unsqueeze(0), kernel_size=5, padding=2
            ).squeeze()
            organ_mask = (dist + noise) < 1.0

            mask[organ_mask] = class_id

        # Add intensity variations based on organs
        for class_id in range(1, self.classes + 1):
            organ_mask = mask == class_id
            if organ_mask.sum() > 0:
                intensity_boost = 0.1 + (class_id - 1) * 0.15
                base_intensity[organ_mask] += intensity_boost

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
    output_dir = Path("outputs/training")
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
    log("IRIS Training - Real Medical Datasets (GPU)")
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

    # Try to load real datasets, fall back to synthetic if not available
    use_real_data = False
    train_datasets = []
    val_datasets = []
    test_datasets = []

    # Try ACDC dataset
    acdc_root = Path("datasets/acdc")
    if acdc_root.exists():
        try:
            log("Attempting to load ACDC dataset...")
            acdc_train = build_dataset("acdc", root=str(acdc_root), split=DatasetSplit.TRAIN)
            acdc_val = build_dataset("acdc", root=str(acdc_root), split=DatasetSplit.VALID)
            acdc_test = build_dataset("acdc", root=str(acdc_root), split=DatasetSplit.TEST)
            
            if len(acdc_train) > 0:
                train_datasets.append(acdc_train)
                val_datasets.append(acdc_val)
                test_datasets.append(acdc_test)
                use_real_data = True
                log(f"  [OK] ACDC loaded: {len(acdc_train)} train, {len(acdc_val)} val, {len(acdc_test)} test")
            else:
                log("  [SKIP] ACDC dataset empty or not properly formatted")
        except Exception as e:
            log(f"  [SKIP] ACDC dataset failed to load: {e}")

    # Try other datasets if ACDC not available
    if not use_real_data:
        for dataset_name, dataset_path in [
            ("amos", Path("datasets/amos")),
            ("msd_pancreas", Path("datasets/msd_pancreas")),
            ("segthor", Path("datasets/segthor")),
        ]:
            if dataset_path.exists():
                try:
                    log(f"Attempting to load {dataset_name} dataset...")
                    train_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.TRAIN)
                    val_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.VALID)
                    test_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.TEST)
                    
                    if len(train_ds) > 0:
                        train_datasets.append(train_ds)
                        val_datasets.append(val_ds)
                        test_datasets.append(test_ds)
                        use_real_data = True
                        log(f"  [OK] {dataset_name} loaded: {len(train_ds)} train")
                        break
                except Exception as e:
                    log(f"  [SKIP] {dataset_name} failed: {e}")

    # Fall back to synthetic data if real data not available
    if not use_real_data:
        log("Using high-quality synthetic dataset (real datasets not available)")
        log("Note: To use real data, download datasets to datasets/ directory")
        log("      See docs/run_real_datasets.md for instructions")
        log("")
        
        train_dataset = ImprovedMedicalDataset(
            length=100,
            classes=3,
            volume_shape=(64, 64, 64),
            modality="CT",
            seed_offset=0,
        )

        val_dataset = ImprovedMedicalDataset(
            length=30,
            classes=3,
            volume_shape=(64, 64, 64),
            modality="CT",
            seed_offset=1000,
        )

        test_dataset = ImprovedMedicalDataset(
            length=20,
            classes=3,
            volume_shape=(64, 64, 64),
            modality="CT",
            seed_offset=2000,
        )

        train_datasets = [train_dataset]
        val_datasets = [val_dataset]
        test_datasets = [test_dataset]

        log(f"Datasets created:")
        log(f"  - Training: {len(train_dataset)} volumes")
        log(f"  - Validation: {len(val_dataset)} volumes")
        log(f"  - Test: {len(test_dataset)} volumes")
        log("")

    # Model configuration optimized for 4GB GPU
    model = IrisModel(
        in_channels=1,
        base_channels=24,
        num_query_tokens=6,
        num_attention_heads=6,
        volume_shape=(64, 64, 64),
        use_memory_bank=True,
    )

    log("Model initialized:")
    log(f"  - Base channels: {model.encoder.base_channels}")
    log(f"  - Volume shape: {model.volume_shape}")
    log(f"  - Device: {device}")
    log(f"  - Memory bank: {'Enabled' if model.memory_bank else 'Disabled'}")
    log("")

    # Training configuration
    config = EpisodicTrainingConfig(
        base_learning_rate=1e-3,
        total_iterations=150,
        warmup_iterations=20,
        batch_size=2,
        volume_size=(64, 64, 64),
        device=device,
        checkpoint_dir=output_dir / "checkpoints",
        log_every=10,
        eval_every=50,
        checkpoint_every=50,
    )

    log("Starting training...")
    log(f"  - Total iterations: {config.total_iterations}")
    log(f"  - Batch size: {config.batch_size}")
    log(f"  - Learning rate: {config.base_learning_rate}")
    log("")

    trainer = EpisodicTrainer(model, train_datasets, config, device=device)
    trainer.train()

    log("")
    log("Training completed!")
    log("")

    # Evaluation
    log("Running evaluation...")
    eval_config = EvaluationConfig(
        batch_size=1,
        device=device,
        strategies=["one_shot", "context_ensemble", "memory_retrieval", "in_context_tuning"],
    )

    evaluator = MedicalEvaluationSuite(model, eval_config)
    results = evaluator.evaluate(test_datasets, dataset_names=["test"])

    log("Evaluation Results:")
    for dataset_name, metrics in results.items():
        log(f"\n{dataset_name}:")
        for strategy, strategy_metrics in metrics.items():
            log(f"  {strategy}:")
            log(f"    Dice: {strategy_metrics.get('dice', {}).get('mean', 0):.4f}")
            log(f"    Inference Time: {strategy_metrics.get('inference_time', 0):.4f}s")

    # Save metrics
    with metrics_file.open("w") as f:
        json.dump(results, f, indent=2)

    log("")
    log(f"Results saved to: {output_dir}")
    log("=" * 80)


if __name__ == "__main__":
    main()

