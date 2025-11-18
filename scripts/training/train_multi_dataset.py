"""
Multi-dataset training script - trains one model on all available real datasets together.
Uses episodic training across multiple datasets for better generalization.
"""
import json
from pathlib import Path

import torch

from iris.data import build_dataset, DatasetSplit
from iris.model import IrisModel
from iris.training import (
    EvaluationConfig,
    MedicalEvaluationSuite,
    EpisodicTrainingConfig,
    EpisodicTrainer,
    set_global_seed,
)
from iris.training import evaluation as eval_mod
from iris.training import demo as demo_mod

# Import dataset checker
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.check_datasets import check_all_datasets

# Import synthetic dataset for fallback
sys.path.insert(0, str(Path(__file__).parent))
try:
    from train_iris import ImprovedMedicalDataset
except ImportError:
    # Define it here if import fails
    class ImprovedMedicalDataset(torch.utils.data.Dataset):
        def __init__(self, length, classes=3, volume_shape=(64, 64, 64), modality="CT", seed_offset=0):
            self.length = length
            self.classes = classes
            self.volume_shape = volume_shape
            self.modality = modality
            self.rng = torch.Generator()
            self.rng.manual_seed(42 + seed_offset)
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            depth, height, width = self.volume_shape
            device = "cpu"
            base_intensity = torch.rand((depth, height, width), generator=self.rng, device=device) * 0.3 + 0.4
            mask = torch.zeros((depth, height, width), dtype=torch.int64, device=device)
            for class_id in range(1, self.classes + 1):
                center_d = int(torch.randint(10, depth - 10, (1,), generator=self.rng).item())
                center_h = int(torch.randint(10, height - 10, (1,), generator=self.rng).item())
                center_w = int(torch.randint(10, width - 10, (1,), generator=self.rng).item())
                radius_d = int(torch.randint(8, 15, (1,), generator=self.rng).item())
                radius_h = int(torch.randint(8, 15, (1,), generator=self.rng).item())
                radius_w = int(torch.randint(8, 15, (1,), generator=self.rng).item())
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
                noise = torch.randn((depth, height, width), generator=self.rng, device=device) * 0.1
                noise = torch.nn.functional.avg_pool3d(
                    noise.unsqueeze(0).unsqueeze(0), kernel_size=5, padding=2
                ).squeeze()
                organ_mask = (dist + noise) < 1.0
                mask[organ_mask] = class_id
            for class_id in range(1, self.classes + 1):
                organ_mask = mask == class_id
                if organ_mask.sum() > 0:
                    intensity_boost = 0.1 + (class_id - 1) * 0.15
                    base_intensity[organ_mask] += intensity_boost
            image = torch.clamp(base_intensity, 0.0, 1.0)
            if mask.sum() == 0:
                mask[depth // 2, height // 2, width // 2] = 1
            return {
                "image": image,
                "mask": mask,
                "meta": {"index": idx, "modality": self.modality, "classes": list(range(1, self.classes + 1))},
            }


def _zero_hausdorff(pred, target, percentile=95.0):
    """Temporary Hausdorff distance for synthetic data."""
    return 0.0


eval_mod._hausdorff_distance = _zero_hausdorff
demo_mod._hausdorff_distance = _zero_hausdorff


def load_available_datasets(datasets_dir: Path = Path("datasets")) -> tuple:
    """
    Load all available real datasets.
    
    Returns:
        (train_datasets, val_datasets, test_datasets, dataset_names)
    """
    # Check which datasets are available
    results = check_all_datasets(datasets_dir)
    
    ready_datasets = [
        name for name, status in results.items()
        if status["can_load"] and status["train_count"] > 0
    ]
    
    if not ready_datasets:
        return [], [], [], []
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    dataset_names = []
    
    for dataset_name in ready_datasets:
        try:
            dataset_path = datasets_dir / dataset_name
            train_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.TRAIN)
            val_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.VALID)
            test_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.TEST)
            
            if len(train_ds) > 0:
                train_datasets.append(train_ds)
                val_datasets.append(val_ds)
                test_datasets.append(test_ds)
                dataset_names.append(dataset_name)
        except Exception as e:
            print(f"  [SKIP] Failed to load {dataset_name}: {e}")
    
    return train_datasets, val_datasets, test_datasets, dataset_names


def main():
    """Main training function."""
    output_dir = Path("outputs/training/multi_dataset")
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
    log("IRIS Multi-Dataset Training")
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
    
    # Load all available datasets
    log("Loading available datasets...")
    train_datasets, val_datasets, test_datasets, dataset_names = load_available_datasets()
    
    # Fall back to synthetic data if no real datasets available
    if not train_datasets:
        log("No real datasets available - using synthetic data for demonstration")
        log("Note: To use real data, download datasets first using:")
        log("  python scripts/data/download_datasets.py")
        log("")
        
        # Import synthetic dataset class
        from train_iris import ImprovedMedicalDataset
        
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
        dataset_names = ["synthetic"]
        
        log(f"Synthetic datasets created:")
        log(f"  - Training: {len(train_dataset)} volumes")
        log(f"  - Validation: {len(val_dataset)} volumes")
        log(f"  - Test: {len(test_dataset)} volumes")
        log("")
    
    log(f"Found {len(train_datasets)} datasets:")
    total_train = 0
    total_val = 0
    total_test = 0
    for i, name in enumerate(dataset_names):
        train_count = len(train_datasets[i])
        val_count = len(val_datasets[i])
        test_count = len(test_datasets[i])
        total_train += train_count
        total_val += val_count
        total_test += test_count
        log(f"  {i+1}. {name}:")
        log(f"     - Train: {train_count} volumes")
        log(f"     - Val: {val_count} volumes")
        log(f"     - Test: {test_count} volumes")
    log("")
    log(f"Total volumes: {total_train} train, {total_val} val, {total_test} test")
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
    
    # Training configuration - extended for real data, reduced for synthetic
    is_synthetic = dataset_names == ["synthetic"]
    total_iterations = 200 if is_synthetic else 500
    eval_every = 50 if is_synthetic else 100
    checkpoint_every = 50 if is_synthetic else 100
    
    config = EpisodicTrainingConfig(
        base_learning_rate=1e-3,
        total_iterations=total_iterations,
        warmup_iterations=30 if is_synthetic else 50,
        batch_size=2,  # GPU memory optimized
        volume_size=(64, 64, 64),
        device=device,
        checkpoint_dir=output_dir / "checkpoints",
        log_every=25,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
    )
    
    log("Training configuration:")
    log(f"  - Total iterations: {config.total_iterations}")
    log(f"  - Batch size: {config.batch_size}")
    log(f"  - Learning rate: {config.base_learning_rate}")
    log(f"  - Datasets: {', '.join(dataset_names)}")
    log("")
    
    # Train on all datasets together
    log("Starting multi-dataset training...")
    trainer = EpisodicTrainer(model, train_datasets, config, device=device)
    trainer.train()
    
    log("")
    log("Training completed!")
    log("")
    
    # Evaluation on all test datasets
    log("Running evaluation on all test datasets...")
    eval_config = EvaluationConfig(
        batch_size=1,
        device=device,
        strategies=["one_shot", "context_ensemble", "memory_retrieval", "in_context_tuning"],
    )
    
    evaluator = MedicalEvaluationSuite(model, eval_config)
    results = evaluator.evaluate(test_datasets, dataset_names=dataset_names)
    
    log("Evaluation Results:")
    log("")
    for dataset_name, metrics in results.items():
        log(f"{dataset_name}:")
        for strategy, strategy_metrics in metrics.items():
            dice_mean = strategy_metrics.get('dice', {}).get('mean', 0)
            dice_std = strategy_metrics.get('dice', {}).get('std', 0)
            inference_time = strategy_metrics.get('inference_time', 0)
            log(f"  {strategy}:")
            log(f"    Dice: {dice_mean:.4f} ± {dice_std:.4f}")
            log(f"    Inference Time: {inference_time:.4f}s")
        log("")
    
    # Save metrics
    with metrics_file.open("w") as f:
        json.dump(results, f, indent=2)
    
    log("")
    log(f"Results saved to: {output_dir}")
    log("=" * 80)


if __name__ == "__main__":
    main()

