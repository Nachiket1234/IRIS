# Multi-Dataset Training and Visualization Guide

This guide explains how to train IRIS on multiple real medical datasets and generate organized visualizations.

## Quick Start

### 1. Check Available Datasets

```powershell
cd "C:\Users\nachi\Downloads\IRIS\IRIS V2"
$env:PYTHONPATH="${PWD}\src"
python scripts/data/check_datasets.py
```

This will show which datasets are ready, partially available, or need download.

### 2. Download Missing Datasets (if needed)

```powershell
python scripts/data/download_datasets.py
```

This provides instructions for downloading datasets. Most medical datasets require manual registration and download.

### 3. Run All Training

```powershell
python scripts/run_all_training.py
```

This will:
- Train one model on all datasets together (multi-dataset training)
- Train separate models for each dataset (per-dataset training)
- Save checkpoints to `outputs/training/`

### 4. Run All Visualization

```powershell
python scripts/run_all_visualization.py
```

This will:
- Visualize results from multi-dataset model
- Visualize results from each per-dataset model
- Generate organized output folders with 5-10 cases per dataset
- Save to `outputs/visualization/`

## Individual Scripts

### Training Scripts

#### Multi-Dataset Training
```powershell
python scripts/training/train_multi_dataset.py
```
Trains one model on all available datasets together using episodic training.

#### Per-Dataset Training
```powershell
python scripts/training/train_per_dataset.py
```
Trains separate models for each dataset.

### Visualization Script

#### Multi-Dataset Visualization
```powershell
python scripts/visualization/visualize_multi_dataset.py --mode multi --num-cases 8
```

Options:
- `--checkpoint PATH`: Specify checkpoint file (auto-detects if not provided)
- `--output PATH`: Output directory (default: `outputs/visualization`)
- `--num-cases N`: Number of cases per dataset (default: 8, max 10)
- `--datasets DATASET1 DATASET2`: Specific datasets to visualize
- `--mode multi|per_dataset`: Visualization mode

## Output Structure

### Training Outputs

```
outputs/training/
├── multi_dataset/
│   ├── checkpoints/
│   │   └── iris_iter_XXXXX.pt
│   ├── training_results.txt
│   └── metrics.json
└── per_dataset/
    ├── acdc/
    │   ├── checkpoints/
    │   ├── training_results.txt
    │   └── metrics.json
    ├── amos/
    └── ...
```

### Visualization Outputs

```
outputs/visualization/
├── multi_dataset/
│   ├── acdc/
│   │   ├── case_01/
│   │   │   ├── 01_input.png
│   │   │   ├── 02_one_shot/
│   │   │   │   ├── support_reference.png
│   │   │   │   └── output_prediction.png
│   │   │   ├── 02_context_ensemble/
│   │   │   ├── 02_memory_retrieval/
│   │   │   └── 02_in_context_tuning/
│   │   ├── case_02/
│   │   └── summary.json
│   ├── amos/
│   └── visualization_summary.json
└── per_dataset/
    ├── acdc/
    └── ...
```

## Supported Datasets

Currently supported datasets (with loaders):
- **ACDC**: Cardiac MRI segmentation
- **AMOS**: Abdominal multi-organ segmentation
- **MSD Pancreas**: Pancreas segmentation
- **SegTHOR**: Thoracic organ segmentation

Note: ISI skin, lung pneumonia, and COVID datasets are typically 2D (X-ray/skin images) and not compatible with IRIS which is designed for 3D volumes.

## GPU Requirements

- Minimum: 4GB GPU memory (NVIDIA GTX 1650 tested)
- Batch size: 2 (optimized for 4GB GPU)
- Volume size: 64×64×64 (memory efficient)

## Troubleshooting

### No datasets found
- Run `python scripts/data/check_datasets.py` to see what's available
- Download datasets using `python scripts/data/download_datasets.py`
- Ensure datasets are in correct directory structure

### Out of memory errors
- Reduce batch size in training config
- Reduce volume size (currently 64×64×64)
- Reduce number of base channels

### Checkpoint not found
- Ensure training completed successfully
- Check `outputs/training/` for checkpoint files
- Use `--checkpoint PATH` to specify checkpoint manually

## Next Steps

After training and visualization:
1. Review metrics in `outputs/training/*/metrics.json`
2. Compare multi-dataset vs per-dataset performance
3. Examine visualizations in `outputs/visualization/`
4. Use best model for inference on new data

