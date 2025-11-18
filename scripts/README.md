# IRIS Scripts Directory

This directory contains organized training and visualization scripts for the IRIS model.

## Structure

```
scripts/
├── training/
│   ├── train_iris.py          # Main training script (USE THIS)
│   ├── train_real_medical_gpu.py
│   ├── train_improved_medical.py
│   └── train_realistic_medical.py
└── visualization/
    ├── visualize_iris.py       # Main visualization script (USE THIS)
    ├── visualize_improved_inference.py
    └── visualize_inference.py
```

## Quick Start

### Training

```bash
# Set PYTHONPATH
$env:PYTHONPATH="${PWD}\src"

# Run training (automatically uses real data if available, otherwise synthetic)
python scripts/training/train_iris.py
```

### Visualization

```bash
# After training completes, generate visualizations
python scripts/visualization/visualize_iris.py
```

## Main Scripts

### `scripts/training/train_iris.py`

**Purpose**: Main consolidated training script that:
- Automatically detects and loads real medical datasets (ACDC, AMOS, MSD Pancreas, SegTHOR)
- Falls back to high-quality synthetic data if real datasets are not available
- Supports GPU training (automatically detects CUDA)
- Saves checkpoints and metrics to `outputs/training/`

**Usage**:
```bash
python scripts/training/train_iris.py
```

**Output**: 
- Checkpoints: `outputs/training/checkpoints/`
- Training log: `outputs/training/training_results.txt`
- Metrics: `outputs/training/metrics.json`

### `scripts/visualization/visualize_iris.py`

**Purpose**: Main visualization script that:
- Automatically finds the latest checkpoint
- Generates organized output folders for each test case
- Creates visualizations for all 4 inference strategies
- Saves high-quality images

**Usage**:
```bash
python scripts/visualization/visualize_iris.py
```

**Output Structure**:
```
outputs/visualization/
├── case_01/
│   ├── 01_input.png
│   ├── 02_one_shot/
│   │   ├── support_reference.png
│   │   └── output_prediction.png
│   ├── 02_context_ensemble/
│   ├── 02_memory_retrieval/
│   └── 02_in_context_tuning/
├── case_02/
└── ...
```

## Legacy Scripts

The other scripts in these directories are legacy versions kept for reference. Use `train_iris.py` and `visualize_iris.py` for all new work.

## Real Dataset Setup

To use real medical datasets:

1. Download datasets to `datasets/` directory:
   - ACDC: `datasets/acdc/training/` and `datasets/acdc/testing/`
   - AMOS: `datasets/amos/imagesTr/` and `datasets/amos/labelsTr/`
   - MSD Pancreas: `datasets/msd_pancreas/`
   - SegTHOR: `datasets/segthor/images/` and `datasets/segthor/labels/`

2. See `docs/run_real_datasets.md` for detailed instructions

3. Run `train_iris.py` - it will automatically detect and use real datasets if available

