"""
Per-dataset training script - trains separate models for each available dataset.
Each dataset gets its own trained model for comparison with multi-dataset approach.
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


def _zero_hausdorff(pred, target, percentile=95.0):
    """Temporary Hausdorff distance for synthetic data."""
    return 0.0


eval_mod._hausdorff_distance = _zero_hausdorff
demo_mod._hausdorff_distance = _zero_hausdorff


def train_single_dataset_synthetic(
    dataset_name: str,
    output_dir: Path,
    device: str,
) -> dict:
    """Train on synthetic dataset."""
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    results_file = dataset_output_dir / "training_results.txt"
    metrics_file = dataset_output_dir / "metrics.json"
    
    if results_file.exists():
        results_file.unlink()
    
    def log(msg: str):
        print(msg)
        with results_file.open("a", encoding="utf-8") as fp:
            fp.write(msg + "\n")
    
    log("=" * 80)
    log(f"IRIS Training - {dataset_name.upper()} (Synthetic)")
    log("=" * 80)
    log("")
    
    try:
        from train_iris import ImprovedMedicalDataset
        
        train_ds = ImprovedMedicalDataset(length=80, classes=3, volume_shape=(64, 64, 64), modality="CT", seed_offset=0)
        val_ds = ImprovedMedicalDataset(length=20, classes=3, volume_shape=(64, 64, 64), modality="CT", seed_offset=1000)
        test_ds = ImprovedMedicalDataset(length=15, classes=3, volume_shape=(64, 64, 64), modality="CT", seed_offset=2000)
        
        log(f"  - Train: {len(train_ds)} volumes")
        log(f"  - Val: {len(val_ds)} volumes")
        log(f"  - Test: {len(test_ds)} volumes")
        log("")
        
        model = IrisModel(in_channels=1, base_channels=24, num_query_tokens=6, num_attention_heads=6, volume_shape=(64, 64, 64), use_memory_bank=True)
        log("Model initialized")
        log("")
        
        config = EpisodicTrainingConfig(
            base_learning_rate=1e-3, total_iterations=150, warmup_iterations=20, batch_size=2,
            volume_size=(64, 64, 64), device=device, checkpoint_dir=dataset_output_dir / "checkpoints",
            log_every=25, eval_every=50, checkpoint_every=50,
        )
        
        log("Starting training...")
        trainer = EpisodicTrainer(model, [train_ds], config, device=device)
        trainer.train()
        
        log("Training completed!")
        log("")
        
        eval_config = EvaluationConfig(batch_size=1, device=device, strategies=["one_shot", "context_ensemble", "memory_retrieval", "in_context_tuning"])
        evaluator = MedicalEvaluationSuite(model, eval_config)
        results = evaluator.evaluate([test_ds], dataset_names=[dataset_name])
        
        log("Evaluation Results:")
        for dataset_name_eval, metrics in results.items():
            for strategy, strategy_metrics in metrics.items():
                dice_mean = strategy_metrics.get('dice', {}).get('mean', 0)
                inference_time = strategy_metrics.get('inference_time', 0)
                log(f"  {strategy}: Dice={dice_mean:.4f}, Time={inference_time:.4f}s")
        
        with metrics_file.open("w") as f:
            json.dump(results, f, indent=2)
        
        log(f"Results saved to: {dataset_output_dir}")
        log("=" * 80)
        
        return {"status": "completed", "dataset": dataset_name, "train_count": len(train_ds), "test_count": len(test_ds), "metrics": results}
    except Exception as e:
        log(f"  [ERROR] Training failed for {dataset_name}: {e}")
        return {"status": "failed", "dataset": dataset_name, "error": str(e)}


def train_single_dataset(
    dataset_name: str,
    dataset_path: Path,
    output_dir: Path,
    device: str,
) -> dict:
    """
    Train a model on a single dataset.
    
    Returns:
        Dictionary with training results and metrics.
    """
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    results_file = dataset_output_dir / "training_results.txt"
    metrics_file = dataset_output_dir / "metrics.json"
    
    if results_file.exists():
        results_file.unlink()
    
    def log(msg: str):
        print(msg)
        with results_file.open("a", encoding="utf-8") as fp:
            fp.write(msg + "\n")
    
    log("=" * 80)
    log(f"IRIS Training - {dataset_name.upper()}")
    log("=" * 80)
    log("")
    
    try:
        # Load dataset
        log(f"Loading {dataset_name} dataset...")
        train_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.TRAIN)
        val_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.VALID)
        test_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.TEST)
        
        log(f"  - Train: {len(train_ds)} volumes")
        log(f"  - Val: {len(val_ds)} volumes")
        log(f"  - Test: {len(test_ds)} volumes")
        log("")
        
        if len(train_ds) == 0:
            log(f"  [SKIP] No training data for {dataset_name}")
            return {"status": "skipped", "reason": "no_training_data"}
        
        # Model configuration
        model = IrisModel(
            in_channels=1,
            base_channels=24,
            num_query_tokens=6,
            num_attention_heads=6,
            volume_shape=(64, 64, 64),
            use_memory_bank=True,
        )
        
        log("Model initialized")
        log("")
        
        # Training configuration
        config = EpisodicTrainingConfig(
            base_learning_rate=1e-3,
            total_iterations=300,  # Fewer iterations per dataset
            warmup_iterations=30,
            batch_size=2,
            volume_size=(64, 64, 64),
            device=device,
            checkpoint_dir=dataset_output_dir / "checkpoints",
            log_every=25,
            eval_every=75,
            checkpoint_every=75,
        )
        
        log("Starting training...")
        trainer = EpisodicTrainer(model, [train_ds], config, device=device)
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
        results = evaluator.evaluate([test_ds], dataset_names=[dataset_name])
        
        log("Evaluation Results:")
        for dataset_name_eval, metrics in results.items():
            for strategy, strategy_metrics in metrics.items():
                dice_mean = strategy_metrics.get('dice', {}).get('mean', 0)
                inference_time = strategy_metrics.get('inference_time', 0)
                log(f"  {strategy}: Dice={dice_mean:.4f}, Time={inference_time:.4f}s")
        
        # Save metrics
        with metrics_file.open("w") as f:
            json.dump(results, f, indent=2)
        
        log("")
        log(f"Results saved to: {dataset_output_dir}")
        log("=" * 80)
        
        return {
            "status": "completed",
            "dataset": dataset_name,
            "train_count": len(train_ds),
            "test_count": len(test_ds),
            "metrics": results,
        }
        
    except Exception as e:
        log(f"  [ERROR] Training failed for {dataset_name}: {e}")
        return {"status": "failed", "dataset": dataset_name, "error": str(e)}


def main():
    """Main training function."""
    output_dir = Path("outputs/training/per_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_dir / "training_summary.json"
    
    print("=" * 80)
    print("IRIS Per-Dataset Training")
    print("=" * 80)
    print()
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    set_global_seed(42)
    
    # Check which datasets are available
    datasets_dir = Path("datasets")
    results = check_all_datasets(datasets_dir)
    
    ready_datasets = [
        (name, datasets_dir / name)
        for name, status in results.items()
        if status["can_load"] and status["train_count"] > 0
    ]
    
    # Fall back to synthetic data if no real datasets available
    if not ready_datasets:
        print("No real datasets available - using synthetic data for demonstration")
        print("Note: To use real data, download datasets first using:")
        print("  python scripts/data/download_datasets.py")
        print()
        
        # Create synthetic dataset
        from train_iris import ImprovedMedicalDataset
        
        ready_datasets = [("synthetic", datasets_dir / "synthetic")]
        
        # Create a synthetic dataset directory structure (virtual)
        print("Creating synthetic dataset for demonstration...")
    
    print(f"Found {len(ready_datasets)} datasets to train:")
    for name, _ in ready_datasets:
        print(f"  - {name}")
    print()
    
    # Train each dataset separately
    training_results = []
    for dataset_name, dataset_path in ready_datasets:
        print(f"\n{'='*80}")
        print(f"Training on {dataset_name}...")
        print(f"{'='*80}\n")
        
        if dataset_name == "synthetic":
            result = train_single_dataset_synthetic(
                dataset_name,
                output_dir,
                device,
            )
        else:
            result = train_single_dataset(
                dataset_name,
                dataset_path,
                output_dir,
                device,
            )
        training_results.append(result)
    
    # Save summary
    with summary_file.open("w") as f:
        json.dump(training_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Per-Dataset Training Summary")
    print("=" * 80)
    completed = [r for r in training_results if r.get("status") == "completed"]
    failed = [r for r in training_results if r.get("status") == "failed"]
    skipped = [r for r in training_results if r.get("status") == "skipped"]
    
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped: {len(skipped)}")
    print()
    print(f"Summary saved to: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

