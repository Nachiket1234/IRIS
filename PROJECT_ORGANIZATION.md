# IRIS Project Organization

This document describes the organization of the IRIS project files and directories.

## Directory Structure

```
IRIS V2/
├── src/                          # Source code (IRIS package)
│   └── iris/
│       ├── data/                 # Data loading and preprocessing
│       ├── model/                 # IRIS model architecture
│       └── training/             # Training, evaluation, visualization
│
├── scripts/                      # Executable scripts
│   ├── training/                 # Training scripts
│   │   └── train_iris.py         # Main training script
│   └── visualization/            # Visualization scripts
│       └── visualize_iris.py     # Main visualization script
│
├── tests/                        # Unit tests
│   ├── test_medical_datasets.py
│   ├── test_iris_model.py
│   ├── test_model_core.py
│   ├── test_memory_bank_and_tuning.py
│   └── test_training_pipeline.py
│
├── datasets/                     # Medical datasets
│   ├── acdc/                     # ACDC cardiac dataset
│   ├── amos/                     # AMOS dataset
│   ├── msd_pancreas/             # MSD Pancreas dataset
│   ├── segthor/                  # SegTHOR dataset
│   └── brain_ct/                 # Brain CT dataset (if downloaded)
│
├── outputs/                      # Training and visualization outputs
│   ├── training/                 # Training outputs
│   │   ├── checkpoints/          # Model checkpoints
│   │   ├── training_results.txt  # Training log
│   │   └── metrics.json          # Evaluation metrics
│   └── visualization/            # Visualization outputs
│       └── case_XX/               # Per-case visualizations
│
├── demo_outputs/                 # Legacy demo outputs (kept for reference)
│   ├── improved_medical_training/
│   ├── real_medical_gpu_training/
│   └── realistic_medical_training/
│
├── docs/                         # Documentation
│   ├── workflow_overview.md      # Workflow explanation
│   ├── run_real_datasets.md      # Real dataset setup guide
│   └── model_architecture.md     # Detailed model documentation
│
├── README.md                     # Project overview
├── PROJECT_ORGANIZATION.md       # This file
└── download_real_dataset.py      # Dataset download helper
```

## Key Files

### Source Code (`src/iris/`)

- **`data/`**: Data loading, preprocessing, augmentations, and dataset loaders
  - `base.py`: Base `MedicalDataset` class
  - `io.py`: NIfTI/MHD file reading
  - `preprocessing.py`: Intensity normalization, resizing
  - `augmentations.py`: 3D medical image augmentations
  - `datasets/`: Specific dataset loaders (ACDC, AMOS, etc.)

- **`model/`**: IRIS model architecture
  - `core.py`: Main `IrisModel` class
  - `encoder.py`: 3D UNet encoder
  - `task_encoding.py`: Foreground and contextual encoding
  - `decoder.py`: Mask decoding module
  - `memory.py`: Class-specific memory bank
  - `tuning.py`: In-context tuning mechanism

- **`training/`**: Training, evaluation, and visualization
  - `pipeline.py`: Episodic training loop
  - `evaluation.py`: Medical evaluation suite
  - `demo.py`: Clinical demonstration runner
  - `visualization.py`: Visualization utilities
  - `lamb.py`: Lamb optimizer implementation

### Scripts (`scripts/`)

- **`training/train_iris.py`**: Main training script
  - Automatically detects real datasets
  - Falls back to synthetic data
  - GPU support
  - Saves to `outputs/training/`

- **`visualization/visualize_iris.py`**: Main visualization script
  - Finds latest checkpoint automatically
  - Generates organized output folders
  - Creates visualizations for all inference strategies
  - Saves to `outputs/visualization/`

### Tests (`tests/`)

- Unit tests for all major components
- Run with: `python -m pytest tests/`

### Datasets (`datasets/`)

- Place downloaded medical datasets here
- Each dataset has its own subdirectory
- See `docs/run_real_datasets.md` for setup instructions

### Outputs (`outputs/`)

- **`training/`**: Training outputs
  - `checkpoints/`: Model checkpoints (`.pt` files)
  - `training_results.txt`: Training log
  - `metrics.json`: Evaluation metrics

- **`visualization/`**: Visualization outputs
  - `case_XX/`: Per-case folders
    - `01_input.png`: Input image
    - `02_one_shot/`: One-shot inference results
    - `02_context_ensemble/`: Context ensemble results
    - `02_memory_retrieval/`: Memory retrieval results
    - `02_in_context_tuning/`: In-context tuning results

## File Organization Principles

1. **Source code** in `src/iris/` - organized by functionality
2. **Scripts** in `scripts/` - organized by purpose (training/visualization)
3. **Outputs** in `outputs/` - organized by type (training/visualization)
4. **Tests** in `tests/` - mirror source structure
5. **Documentation** in `docs/` - comprehensive guides
6. **Datasets** in `datasets/` - one subdirectory per dataset

## Legacy Files

Some files from earlier development are kept for reference:
- `demo_outputs/`: Previous training outputs
- Old training/visualization scripts in `scripts/` subdirectories

The main scripts to use are:
- `scripts/training/train_iris.py`
- `scripts/visualization/visualize_iris.py`

## Quick Start

1. **Training**:
   ```bash
   $env:PYTHONPATH="${PWD}\src"
   python scripts/training/train_iris.py
   ```

2. **Visualization**:
   ```bash
   python scripts/visualization/visualize_iris.py
   ```

3. **Using Real Data**:
   - Download datasets to `datasets/`
   - Run `train_iris.py` - it will auto-detect real datasets

## Maintenance

- Keep `src/` clean and organized
- Use `scripts/` for all executable scripts
- Save outputs to `outputs/` (not root directory)
- Document new features in `docs/`
- Add tests for new functionality in `tests/`

