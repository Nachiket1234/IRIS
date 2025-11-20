"""
Download and setup ACDC (Automatic Cardiac Diagnosis Challenge) dataset.
This script downloads the ACDC dataset from official sources and organizes it.
"""
import urllib.request
import zipfile
from pathlib import Path
import shutil

def download_acdc():
    """Download ACDC dataset using direct links."""
    dataset_root = Path("datasets/acdc")
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ACDC Dataset Download Instructions")
    print("=" * 60)
    print()
    print("The ACDC dataset requires manual download due to registration requirements.")
    print()
    print("Please follow these steps:")
    print()
    print("1. Visit: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html")
    print("2. Register for the challenge (free)")
    print("3. Download 'training.zip' and 'testing.zip'")
    print("4. Place the downloaded files in: datasets/acdc/")
    print("5. Run this script again to extract the files")
    print()
    print("=" * 60)
    
    # Check if files exist
    training_zip = dataset_root / "training.zip"
    testing_zip = dataset_root / "testing.zip"
    
    if training_zip.exists() or testing_zip.exists():
        print("\nFound dataset archives. Extracting...")
        
        if training_zip.exists():
            print(f"Extracting {training_zip}...")
            with zipfile.ZipFile(training_zip, 'r') as zip_ref:
                zip_ref.extractall(dataset_root)
            print("✓ Training set extracted")
        
        if testing_zip.exists():
            print(f"Extracting {testing_zip}...")
            with zipfile.ZipFile(testing_zip, 'r') as zip_ref:
                zip_ref.extractall(dataset_root)
            print("✓ Testing set extracted")
        
        print("\n✓ ACDC dataset ready!")
        print(f"Location: {dataset_root.absolute()}")
        
        # Verify structure
        training_dir = dataset_root / "training"
        if training_dir.exists():
            patient_dirs = list(training_dir.glob("patient*"))
            print(f"\nFound {len(patient_dirs)} patient directories")
        
        return True
    else:
        print("\nDataset files not found. Please download them manually first.")
        return False

if __name__ == "__main__":
    download_acdc()
