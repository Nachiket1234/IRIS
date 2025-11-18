"""
Script to download and prepare real medical datasets for IRIS training.
This script attempts to download datasets from publicly available sources.
"""
import os
import subprocess
from pathlib import Path

def download_acdc_official():
    """
    ACDC dataset is available from:
    https://www.creatis.insa-lyon.fr/Challenge/acdc/
    
    Users need to register and download manually.
    """
    print("ACDC Dataset Download Instructions:")
    print("1. Visit: https://www.creatis.insa-lyon.fr/Challenge/acdc/")
    print("2. Register for the challenge")
    print("3. Download the training data")
    print("4. Extract to: datasets/acdc/training/")
    print("   Expected structure:")
    print("     datasets/acdc/training/patient001/patient001_frame01.nii.gz")
    print("     datasets/acdc/training/patient001/patient001_frame01_gt.nii.gz")
    print()

def setup_dataset_structure():
    """Create directory structure for datasets."""
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    for dataset_name in ["acdc", "amos", "msd_pancreas", "segthor"]:
        dataset_path = datasets_dir / dataset_name
        dataset_path.mkdir(exist_ok=True)
        print(f"Created directory: {dataset_path}")

if __name__ == "__main__":
    print("=" * 80)
    print("Real Medical Dataset Setup")
    print("=" * 80)
    print()
    
    setup_dataset_structure()
    print()
    download_acdc_official()
    
    print("=" * 80)
    print("Note: Most medical imaging datasets require registration and manual download")
    print("due to data privacy and licensing requirements.")
    print("=" * 80)



