"""
Master training script - orchestrates both multi-dataset and per-dataset training.
"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Import dataset checker
sys.path.insert(0, str(Path(__file__).parent))
from data.check_datasets import check_all_datasets


def run_training_script(script_path: Path, description: str) -> dict:
    """Run a training script and return results."""
    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        
        if result.returncode == 0:
            print(result.stdout)
            return {
                "status": "success",
                "description": description,
                "script": str(script_path),
            }
        else:
            print(result.stdout)
            print(result.stderr)
            return {
                "status": "failed",
                "description": description,
                "script": str(script_path),
                "error": result.stderr,
            }
    except Exception as e:
        return {
            "status": "error",
            "description": description,
            "script": str(script_path),
            "error": str(e),
        }


def main():
    """Main orchestration function."""
    print("=" * 80)
    print("IRIS Master Training Orchestrator")
    print("=" * 80)
    print()
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check dataset availability first
    print("Checking dataset availability...")
    datasets_dir = Path("datasets")
    results = check_all_datasets(datasets_dir)
    
    ready_datasets = [
        name for name, status in results.items()
        if status["can_load"] and status["train_count"] > 0
    ]
    
    if not ready_datasets:
        print("\n[WARNING] No real datasets available!")
        print("Training will use synthetic data.")
        print("To use real datasets, download them first:")
        print("  python scripts/data/download_datasets.py")
        print()
        response = input("Continue with synthetic data? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    else:
        print(f"\n[OK] Found {len(ready_datasets)} ready datasets: {', '.join(ready_datasets)}")
        print()
    
    # Prepare scripts
    scripts_dir = Path(__file__).parent / "training"
    multi_dataset_script = scripts_dir / "train_multi_dataset.py"
    per_dataset_script = scripts_dir / "train_per_dataset.py"
    
    training_results = []
    
    # Run multi-dataset training
    if multi_dataset_script.exists():
        result = run_training_script(
            multi_dataset_script,
            "Multi-Dataset Training (one model on all datasets)"
        )
        training_results.append(result)
    else:
        print(f"[ERROR] Script not found: {multi_dataset_script}")
    
    # Run per-dataset training
    if per_dataset_script.exists():
        result = run_training_script(
            per_dataset_script,
            "Per-Dataset Training (separate models for each dataset)"
        )
        training_results.append(result)
    else:
        print(f"[ERROR] Script not found: {per_dataset_script}")
    
    # Save summary
    output_dir = Path("outputs/training")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "training_orchestration_summary.json"
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "ready_datasets": ready_datasets,
        "results": training_results,
    }
    
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training Orchestration Summary")
    print("=" * 80)
    print(f"Ready datasets: {len(ready_datasets)}")
    print(f"Completed training runs: {sum(1 for r in training_results if r['status'] == 'success')}")
    print(f"Failed training runs: {sum(1 for r in training_results if r['status'] == 'failed')}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

