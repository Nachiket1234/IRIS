# IRIS Project - Session Summary

## What Was Accomplished

### 1. File Organization ✅
- **Created organized directory structure**:
  - `scripts/training/` - All training scripts
  - `scripts/visualization/` - All visualization scripts
  - `outputs/training/` - Training outputs (checkpoints, logs, metrics)
  - `outputs/visualization/` - Visualization outputs

- **Consolidated scripts**:
  - `scripts/training/train_iris.py` - Main training script (replaces multiple training scripts)
  - `scripts/visualization/visualize_iris.py` - Main visualization script (replaces multiple visualization scripts)

- **Moved legacy files**:
  - Old training scripts moved to `scripts/training/`
  - Old visualization scripts moved to `scripts/visualization/`

### 2. Fixed Memory Bank Issue ✅
- **Problem**: `ValueError: Number of class IDs per sample must match embedding count` during memory bank updates
- **Solution**: Modified `src/iris/training/pipeline.py` to update memory bank per sample instead of per batch, handling variable class counts correctly
- **Location**: `src/iris/training/pipeline.py` lines 386-397

### 3. Created Main Training Script ✅
- **File**: `scripts/training/train_iris.py`
- **Features**:
  - Automatically detects real medical datasets (ACDC, AMOS, MSD Pancreas, SegTHOR)
  - Falls back to high-quality synthetic data if real datasets unavailable
  - GPU support (automatically detects CUDA)
  - Saves outputs to organized `outputs/training/` directory
  - Comprehensive logging and metrics

### 4. Created Main Visualization Script ✅
- **File**: `scripts/visualization/visualize_iris.py`
- **Features**:
  - Automatically finds latest checkpoint
  - Generates organized output folders:
    ```
    outputs/visualization/
    ├── case_01/
    │   ├── 01_input.png
    │   ├── 02_one_shot/
    │   ├── 02_context_ensemble/
    │   ├── 02_memory_retrieval/
    │   └── 02_in_context_tuning/
    ```
  - Creates high-quality visualizations for all 4 inference strategies
  - Saves support/reference and output prediction images

### 5. Documentation ✅
- **Created**:
  - `scripts/README.md` - Guide for using training/visualization scripts
  - `PROJECT_ORGANIZATION.md` - Complete project structure documentation
  - `SESSION_SUMMARY.md` - This file

### 6. Training Status
- **Current**: Training running in background with improved synthetic data
- **Reason for synthetic data**: Real datasets (ACDC zip) are corrupted/not properly downloaded
- **Training location**: `outputs/training/`
- **Checkpoints**: Will be saved to `outputs/training/checkpoints/`

## Current State

### Training
- ✅ Training script running: `scripts/training/train_iris.py`
- ✅ Using GPU: NVIDIA GeForce GTX 1650
- ✅ Using improved synthetic dataset (100 train, 30 val, 20 test)
- ⏳ Training in progress (150 iterations)

### Real Data Setup
- ⚠️ ACDC dataset zip file is corrupted
- ⚠️ Other datasets (AMOS, MSD Pancreas, SegTHOR) not downloaded
- ✅ Scripts ready to use real data when available
- 📝 Instructions in `docs/run_real_datasets.md`

### Next Steps

1. **Wait for training to complete** (check `outputs/training/training_results.txt`)
2. **Run visualization**:
   ```bash
   python scripts/visualization/visualize_iris.py
   ```
3. **For real data**:
   - Download ACDC dataset properly (see `docs/run_real_datasets.md`)
   - Place in `datasets/acdc/training/` and `datasets/acdc/testing/`
   - Run `scripts/training/train_iris.py` - it will auto-detect

## File Locations

### Main Scripts (Use These)
- Training: `scripts/training/train_iris.py`
- Visualization: `scripts/visualization/visualize_iris.py`

### Outputs
- Training: `outputs/training/`
  - Checkpoints: `outputs/training/checkpoints/`
  - Log: `outputs/training/training_results.txt`
  - Metrics: `outputs/training/metrics.json`
- Visualization: `outputs/visualization/`

### Documentation
- Project structure: `PROJECT_ORGANIZATION.md`
- Scripts guide: `scripts/README.md`
- Real dataset setup: `docs/run_real_datasets.md`
- Model architecture: `docs/model_architecture.md`

## Key Improvements

1. **Organization**: All files properly organized in logical directories
2. **Consolidation**: Single main script for training and visualization
3. **Auto-detection**: Scripts automatically find checkpoints and datasets
4. **GPU Support**: Properly configured for NVIDIA GTX 1650
5. **Error Handling**: Fixed memory bank update issue
6. **Documentation**: Comprehensive guides for all aspects

## Commands to Run

### Training
```powershell
cd "C:\Users\nachi\Downloads\IRIS\IRIS V2"
$env:PYTHONPATH="${PWD}\src"
python scripts/training/train_iris.py
```

### Visualization (after training)
```powershell
cd "C:\Users\nachi\Downloads\IRIS\IRIS V2"
$env:PYTHONPATH="${PWD}\src"
python scripts/visualization/visualize_iris.py
```

## Notes

- Real datasets require manual download due to licensing/privacy
- ACDC dataset zip file appears corrupted - needs re-download
- Training currently using high-quality synthetic data
- All outputs organized in `outputs/` directory
- Legacy outputs kept in `demo_outputs/` for reference

