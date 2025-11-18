"""
Dataset download helper script with Kaggle API and manual download instructions.
"""
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

import kaggle


def check_kaggle_api() -> bool:
    """Check if Kaggle API is configured."""
    try:
        api = kaggle.api
        api.authenticate()
        return True
    except Exception as e:
        print(f"Kaggle API not configured: {e}")
        print("Please set up Kaggle API credentials:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Create API token")
        print("  3. Place kaggle.json in ~/.kaggle/ (or current directory)")
        return False


def download_from_kaggle(dataset: str, output_dir: Path, unzip: bool = True) -> bool:
    """Download a dataset from Kaggle."""
    try:
        api = kaggle.api
        print(f"Downloading {dataset} from Kaggle...")
        api.dataset_download_files(dataset, path=str(output_dir), unzip=unzip)
        print(f"  [OK] Downloaded to {output_dir}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {dataset}: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """Extract a zip file."""
    try:
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"  [OK] Extracted to {output_dir}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to extract {zip_path}: {e}")
        return False


def download_acdc(output_dir: Path) -> bool:
    """Download ACDC dataset (requires manual registration)."""
    print("\n" + "=" * 80)
    print("ACDC Dataset Download")
    print("=" * 80)
    print("ACDC requires manual registration and download.")
    print("Steps:")
    print("  1. Visit: https://www.creatis.insa-lyon.fr/Challenge/acdc/")
    print("  2. Register for the challenge")
    print("  3. Download the training data")
    print("  4. Extract to: datasets/acdc/training/")
    print("  5. Expected structure:")
    print("     datasets/acdc/training/patient001/patient001_frame01.nii.gz")
    print("     datasets/acdc/training/patient001/patient001_frame01_gt.nii.gz")
    print("=" * 80)
    return False


def download_amos(output_dir: Path) -> bool:
    """Download AMOS dataset (requires manual registration)."""
    print("\n" + "=" * 80)
    print("AMOS Dataset Download")
    print("=" * 80)
    print("AMOS requires manual registration and download.")
    print("Steps:")
    print("  1. Visit: https://amos22.grand-challenge.org/")
    print("  2. Register for the challenge")
    print("  3. Download the dataset")
    print("  4. Extract to: datasets/amos/")
    print("  5. Expected structure:")
    print("     datasets/amos/imagesTr/")
    print("     datasets/amos/labelsTr/")
    print("=" * 80)
    return False


def download_msd_pancreas(output_dir: Path) -> bool:
    """Download MSD Pancreas dataset."""
    print("\n" + "=" * 80)
    print("MSD Pancreas Dataset Download")
    print("=" * 80)
    print("MSD Pancreas is part of Medical Segmentation Decathlon.")
    print("Steps:")
    print("  1. Visit: http://medicaldecathlon.com/")
    print("  2. Download Task07_Pancreas.tar")
    print("  3. Extract to: datasets/msd_pancreas/")
    print("  4. Expected structure:")
    print("     datasets/msd_pancreas/imagesTr/")
    print("     datasets/msd_pancreas/labelsTr/")
    print("=" * 80)
    return False


def download_segthor(output_dir: Path) -> bool:
    """Download SegTHOR dataset."""
    print("\n" + "=" * 80)
    print("SegTHOR Dataset Download")
    print("=" * 80)
    print("SegTHOR requires manual download.")
    print("Steps:")
    print("  1. Visit: https://competitions.codalab.org/competitions/21145")
    print("  2. Register and download the dataset")
    print("  3. Extract to: datasets/segthor/")
    print("  4. Expected structure:")
    print("     datasets/segthor/images/")
    print("     datasets/segthor/labels/")
    print("=" * 80)
    return False


def try_kaggle_downloads(output_dir: Path) -> None:
    """Try to download available datasets from Kaggle."""
    if not check_kaggle_api():
        return
    
    # Known Kaggle datasets (if available)
    kaggle_datasets = {
        # Add Kaggle dataset identifiers here if found
    }
    
    for dataset_name, kaggle_id in kaggle_datasets.items():
        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        download_from_kaggle(kaggle_id, dataset_dir)


def main():
    """Main entry point."""
    print("=" * 80)
    print("IRIS Dataset Download Helper")
    print("=" * 80)
    print()
    
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Check what's needed
    from check_datasets import check_all_datasets
    results = check_all_datasets(datasets_dir)
    
    missing = [name for name, status in results.items() 
               if not status["can_load"]]
    
    if not missing:
        print("All datasets are ready!")
        return
    
    print(f"Missing datasets: {', '.join(missing)}")
    print()
    
    # Try Kaggle downloads first
    print("Attempting Kaggle downloads...")
    try_kaggle_downloads(datasets_dir)
    print()
    
    # Provide manual download instructions
    download_functions = {
        "acdc": download_acdc,
        "amos": download_amos,
        "msd_pancreas": download_msd_pancreas,
        "segthor": download_segthor,
    }
    
    for dataset_name in missing:
        if dataset_name in download_functions:
            download_functions[dataset_name](datasets_dir / dataset_name)
    
    # Try to extract any zip files found
    print("\nChecking for zip files to extract...")
    for zip_file in datasets_dir.rglob("*.zip"):
        dataset_name = zip_file.parent.name
        print(f"Found zip: {zip_file}")
        if extract_zip(zip_file, zip_file.parent):
            # Try to remove zip after extraction
            try:
                zip_file.unlink()
            except:
                pass
    
    print("\n" + "=" * 80)
    print("After downloading datasets, run:")
    print("  python scripts/data/check_datasets.py")
    print("to verify they are properly formatted.")
    print("=" * 80)


if __name__ == "__main__":
    main()

