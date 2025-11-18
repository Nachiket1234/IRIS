"""
Master visualization script - orchestrates visualization for both multi-dataset and per-dataset models.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.visualize_multi_dataset import visualize_from_checkpoint, find_latest_checkpoint


def main():
    """Main orchestration function."""
    print("=" * 80)
    print("IRIS Master Visualization Orchestrator")
    print("=" * 80)
    print()
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {__import__('torch').cuda.get_device_name(0)}")
    print()
    
    # Check for multi-dataset checkpoint
    multi_checkpoint_dirs = [
        Path("outputs/training/multi_dataset/checkpoints"),
    ]
    multi_checkpoint = find_latest_checkpoint(multi_checkpoint_dirs)
    
    # Check for per-dataset checkpoints
    per_dataset_base = Path("outputs/training/per_dataset")
    per_dataset_checkpoints = {}
    if per_dataset_base.exists():
        for dataset_dir in per_dataset_base.iterdir():
            if dataset_dir.is_dir():
                ckpt_dir = dataset_dir / "checkpoints"
                ckpt = find_latest_checkpoint([ckpt_dir])
                if ckpt:
                    per_dataset_checkpoints[dataset_dir.name] = ckpt
    
    if not multi_checkpoint and not per_dataset_checkpoints:
        print("[ERROR] No checkpoints found!")
        print("Please run training first:")
        print("  python scripts/run_all_training.py")
        return
    
    visualization_results = []
    
    # Visualize multi-dataset model
    if multi_checkpoint:
        print("\n" + "=" * 80)
        print("Visualizing Multi-Dataset Model")
        print("=" * 80)
        print(f"Checkpoint: {multi_checkpoint}")
        print()
        
        try:
            result = visualize_from_checkpoint(
                multi_checkpoint,
                Path("outputs/visualization/multi_dataset"),
                device,
                num_cases_per_dataset=8,
            )
            visualization_results.append({
                "mode": "multi_dataset",
                "checkpoint": str(multi_checkpoint),
                "status": "success",
                "result": result,
            })
        except Exception as e:
            print(f"[ERROR] Multi-dataset visualization failed: {e}")
            visualization_results.append({
                "mode": "multi_dataset",
                "checkpoint": str(multi_checkpoint),
                "status": "failed",
                "error": str(e),
            })
    else:
        print("[SKIP] No multi-dataset checkpoint found")
    
    # Visualize per-dataset models
    if per_dataset_checkpoints:
        print("\n" + "=" * 80)
        print("Visualizing Per-Dataset Models")
        print("=" * 80)
        print()
        
        for dataset_name, checkpoint_path in per_dataset_checkpoints.items():
            print(f"\n{'='*80}")
            print(f"Visualizing {dataset_name} (per-dataset model)")
            print(f"{'='*80}")
            print(f"Checkpoint: {checkpoint_path}")
            print()
            
            try:
                result = visualize_from_checkpoint(
                    checkpoint_path,
                    Path("outputs/visualization/per_dataset") / dataset_name,
                    device,
                    num_cases_per_dataset=8,
                    dataset_filter=[dataset_name],
                )
                visualization_results.append({
                    "mode": "per_dataset",
                    "dataset": dataset_name,
                    "checkpoint": str(checkpoint_path),
                    "status": "success",
                    "result": result,
                })
            except Exception as e:
                print(f"[ERROR] Visualization failed for {dataset_name}: {e}")
                visualization_results.append({
                    "mode": "per_dataset",
                    "dataset": dataset_name,
                    "checkpoint": str(checkpoint_path),
                    "status": "failed",
                    "error": str(e),
                })
    else:
        print("[SKIP] No per-dataset checkpoints found")
    
    # Save summary
    output_dir = Path("outputs/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "visualization_orchestration_summary.json"
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "results": visualization_results,
    }
    
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Visualization Orchestration Summary")
    print("=" * 80)
    successful = [r for r in visualization_results if r.get("status") == "success"]
    failed = [r for r in visualization_results if r.get("status") == "failed"]
    print(f"Successful visualizations: {len(successful)}")
    print(f"Failed visualizations: {len(failed)}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

