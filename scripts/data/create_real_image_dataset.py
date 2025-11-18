"""
Create a dataset from real medical images if available.
This script looks for real image files and converts them to the expected format.
"""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import nibabel as nib


def find_real_images(datasets_dir: Path = Path("datasets")) -> list:
    """Find any real image files in the datasets directory."""
    real_files = []
    
    # Look for NIfTI files
    nifti_files = list(datasets_dir.rglob("*.nii*"))
    real_files.extend(nifti_files)
    
    # Look for DICOM directories
    dicom_dirs = [d for d in datasets_dir.rglob("*") if d.is_dir() and any(d.glob("*.dcm"))]
    real_files.extend(dicom_dirs)
    
    return real_files


def load_nifti_as_volume(nifti_path: Path) -> tuple:
    """Load a NIfTI file and return image and mask if available."""
    try:
        nii = nib.load(str(nifti_path))
        data = nii.get_fdata()
        
        # Normalize to [0, 1]
        data_min = np.percentile(data, 2)
        data_max = np.percentile(data, 98)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        data = np.clip(data, 0, 1)
        
        return torch.from_numpy(data).float(), None
    except Exception as e:
        print(f"Error loading {nifti_path}: {e}")
        return None, None


def main():
    """Check for real images and report."""
    datasets_dir = Path("datasets")
    real_files = find_real_images(datasets_dir)
    
    print("=" * 80)
    print("Real Medical Image Search")
    print("=" * 80)
    print()
    
    if real_files:
        print(f"Found {len(real_files)} potential real image files:")
        for f in real_files[:10]:
            print(f"  {f}")
    else:
        print("No real image files found in datasets/ directory")
        print()
        print("To use real datasets:")
        print("1. Download ACDC, AMOS, MSD Pancreas, or SegTHOR datasets")
        print("2. Place NIfTI files (.nii or .nii.gz) in datasets/<dataset_name>/")
        print("3. Run: python scripts/data/check_datasets.py")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

