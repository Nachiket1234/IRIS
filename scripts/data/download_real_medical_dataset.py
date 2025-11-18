"""
Download a real medical imaging dataset for IRIS training.
This script attempts to download publicly available medical datasets.
"""
import json
import subprocess
import zipfile
from pathlib import Path

try:
    import kaggle
    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False
    print("Kaggle API not installed. Install with: pip install kaggle")


def download_kaggle_dataset(dataset_id: str, output_dir: Path) -> bool:
    """Download a dataset from Kaggle."""
    if not HAS_KAGGLE:
        return False
    
    try:
        api = kaggle.api
        api.authenticate()
        print(f"Downloading {dataset_id} from Kaggle...")
        api.dataset_download_files(dataset_id, path=str(output_dir), unzip=True)
        print(f"  [OK] Downloaded to {output_dir}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        return False


def setup_acdc_structure(dataset_dir: Path):
    """Try to organize ACDC-like structure if files are found."""
    # Look for any NIfTI files
    nifti_files = list(dataset_dir.rglob("*.nii*"))
    if not nifti_files:
        return False
    
    # Try to organize into training structure
    training_dir = dataset_dir / "training"
    training_dir.mkdir(exist_ok=True)
    
    # Group by patient if possible
    patients = {}
    for nifti_file in nifti_files:
        name = nifti_file.stem
        if "_gt" in name or "ground" in name.lower():
            continue  # Skip masks for now
        # Try to extract patient ID
        patient_id = name.split("_")[0] if "_" in name else name[:8]
        if patient_id not in patients:
            patients[patient_id] = []
        patients[patient_id].append(nifti_file)
    
    # Move files to patient directories
    for patient_id, files in list(patients.items())[:5]:  # Limit to first 5 patients
        patient_dir = training_dir / f"patient{patient_id:03d}"
        patient_dir.mkdir(exist_ok=True)
        for file in files:
            try:
                # Copy to patient directory
                import shutil
                shutil.copy2(file, patient_dir / file.name)
            except:
                pass
    
    return len(patients) > 0


def main():
    """Main download function."""
    print("=" * 80)
    print("Real Medical Dataset Download")
    print("=" * 80)
    print()
    
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to download a publicly available medical dataset
    # Note: Most require registration, but we'll try Kaggle first
    
    if HAS_KAGGLE and Path("kaggle.json").exists():
        print("Attempting to download from Kaggle...")
        
        # Try some publicly available medical imaging datasets
        kaggle_datasets = [
            # Add known Kaggle medical imaging datasets here
        ]
        
        for dataset_id in kaggle_datasets:
            output_dir = datasets_dir / dataset_id.split("/")[-1]
            if download_kaggle_dataset(dataset_id, output_dir):
                # Try to organize the structure
                setup_acdc_structure(output_dir)
                break
    else:
        print("Kaggle API not configured or not available")
        print()
    
    print("=" * 80)
    print("Manual Download Instructions:")
    print("=" * 80)
    print()
    print("For real medical datasets, please download manually:")
    print()
    print("1. ACDC Dataset:")
    print("   - Visit: https://www.creatis.insa-lyon.fr/Challenge/acdc/")
    print("   - Register and download")
    print("   - Extract to: datasets/acdc/training/")
    print()
    print("2. Or use any publicly available 3D medical imaging dataset")
    print("   - Place NIfTI files in: datasets/<dataset_name>/")
    print("   - Expected format: .nii or .nii.gz files")
    print()
    print("After downloading, run:")
    print("  python scripts/data/check_datasets.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

