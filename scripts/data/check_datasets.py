"""
Dataset discovery script - checks which datasets are available and properly formatted.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from iris.data import build_dataset, DatasetSplit


def check_dataset(dataset_name: str, dataset_path: Path) -> Dict[str, any]:
    """
    Check if a dataset is available and properly formatted.
    
    Returns a dictionary with status information.
    """
    result = {
        "name": dataset_name,
        "path": str(dataset_path),
        "exists": False,
        "has_files": False,
        "can_load": False,
        "train_count": 0,
        "val_count": 0,
        "test_count": 0,
        "error": None,
    }
    
    if not dataset_path.exists():
        result["error"] = "Directory does not exist"
        return result
    
    result["exists"] = True
    
    # Check for NIfTI files
    nifti_files = list(dataset_path.rglob("*.nii*"))
    if len(nifti_files) > 0:
        result["has_files"] = True
    
    # Try to load the dataset
    try:
        train_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.TRAIN)
        val_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.VALID)
        test_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.TEST)
        
        result["can_load"] = True
        result["train_count"] = len(train_ds)
        result["val_count"] = len(val_ds)
        result["test_count"] = len(test_ds)
    except Exception as e:
        result["error"] = str(e)
    
    return result


def check_all_datasets(datasets_dir: Path = Path("datasets")) -> Dict[str, Dict[str, any]]:
    """
    Check all known datasets.
    
    Returns a dictionary mapping dataset names to their status.
    """
    # Known datasets that have loaders
    known_datasets = {
        "acdc": datasets_dir / "acdc",
        "amos": datasets_dir / "amos",
        "chest_xray_masks": datasets_dir / "chest_xray_masks",
        "msd_pancreas": datasets_dir / "msd_pancreas",
        "segthor": datasets_dir / "segthor",
        "isic": datasets_dir / "isic",
    }
    
    results = {}
    for dataset_name, dataset_path in known_datasets.items():
        print(f"Checking {dataset_name}...")
        results[dataset_name] = check_dataset(dataset_name, dataset_path)
    
    return results


def print_summary(results: Dict[str, Dict[str, any]]) -> None:
    """Print a formatted summary of dataset availability."""
    print("\n" + "=" * 80)
    print("Dataset Availability Summary")
    print("=" * 80)
    print()
    
    ready_datasets = []
    partial_datasets = []
    missing_datasets = []
    
    for name, status in results.items():
        if status["can_load"] and status["train_count"] > 0:
            ready_datasets.append((name, status))
        elif status["exists"] and status["has_files"]:
            partial_datasets.append((name, status))
        else:
            missing_datasets.append((name, status))
    
    if ready_datasets:
        print("[OK] READY DATASETS (can be used for training):")
        for name, status in ready_datasets:
            print(f"  {name}:")
            print(f"    - Train: {status['train_count']} volumes")
            print(f"    - Val: {status['val_count']} volumes")
            print(f"    - Test: {status['test_count']} volumes")
        print()
    
    if partial_datasets:
        print("[WARN] PARTIAL DATASETS (files exist but cannot load):")
        for name, status in partial_datasets:
            print(f"  {name}:")
            print(f"    - Path: {status['path']}")
            print(f"    - Error: {status['error']}")
        print()
    
    if missing_datasets:
        print("[MISSING] MISSING DATASETS (need download):")
        for name, status in missing_datasets:
            print(f"  {name}:")
            print(f"    - Path: {status['path']}")
            if status['error']:
                print(f"    - Error: {status['error']}")
        print()
    
    print("=" * 80)
    print(f"Total ready: {len(ready_datasets)}")
    print(f"Total partial: {len(partial_datasets)}")
    print(f"Total missing: {len(missing_datasets)}")
    print("=" * 80)


def main():
    """Main entry point."""
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print(f"Error: Datasets directory not found at {datasets_dir}")
        print("Please create the datasets directory first.")
        return
    
    results = check_all_datasets(datasets_dir)
    print_summary(results)
    
    # Save results to JSON
    output_file = Path("outputs/dataset_status.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Return ready datasets for use in other scripts
    ready = [name for name, status in results.items() 
             if status["can_load"] and status["train_count"] > 0]
    return ready


if __name__ == "__main__":
    main()

