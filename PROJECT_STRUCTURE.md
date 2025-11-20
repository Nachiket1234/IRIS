# IRIS Project Structure

**Clean and organized structure after cleanup - Ready for research/publication**

---

## 📁 Project Organization

```
IRIS V2/
│
├── 📄 README.md                          # Main project documentation
├── 📄 QUICK_START.md                     # Quick start guide
├── 📄 FINAL_COMPREHENSIVE_REPORT.md      # Complete evaluation & results
├── 🔧 run_complete_pipeline.py           # Main training pipeline
├── 🧹 cleanup_project.py                 # Project cleanup utility
│
├── 📚 docs/
│   └── model_architecture.md             # IRIS architecture details
│
├── 🔬 src/
│   └── iris/                             # IRIS framework implementation
│       ├── model/                        # Model components
│       │   ├── encoder.py                # 3D U-Net encoder
│       │   ├── decoder.py                # Mask decoder
│       │   ├── task_encoding.py          # Task encoding module
│       │   ├── memory_bank.py            # Class memory bank
│       │   └── tuning.py                 # In-context tuning & loss
│       ├── data/                         # Dataset loaders
│       └── utils/                        # Utilities
│
├── 🧪 scripts/
│   ├── data/
│   │   ├── check_datasets.py             # Dataset verification
│   │   ├── download_datasets.py          # Dataset downloaders
│   │   └── generate_isic_masks.py        # ISIC mask generation
│   │
│   ├── training/
│   │   └── train_with_metrics.py         # Training with comprehensive metrics
│   │
│   ├── visualization/
│   │   ├── visualize_improved.py         # 4-panel visualizations
│   │   └── visualize_iris_context.py     # IRIS in-context learning viz
│   │
│   ├── comparison/
│   │   ├── compare_iris_ablations.py     # IRIS ablation studies
│   │   └── compare_methods.py            # Baseline comparisons
│   │
│   └── generate_report.py                # Training report generation
│
├── 📊 outputs/
│   ├── dataset_status.json               # Dataset availability status
│   │
│   └── training_with_metrics/
│       │
│       ├── isic/                         # ISIC results
│       │   ├── training_metrics.json     # Training metrics & losses
│       │   │
│       │   ├── checkpoints/
│       │   │   └── final_model.pt        # Final trained model
│       │   │
│       │   ├── ablation/
│       │   │   ├── ablation_study.md     # Detailed ablation analysis
│       │   │   ├── ablation_summary.json # Ablation metrics
│       │   │   └── iris_ablation_comparison.png  # Comparison charts
│       │   │
│       │   ├── comparison/
│       │   │   ├── comparison_summary.json
│       │   │   ├── comparison_table.csv
│       │   │   └── comparison_table.md
│       │   │
│       │   └── report/
│       │       └── training_report.md    # Training summary
│       │
│       └── chest_xray_masks/             # Chest X-ray results
│           ├── training_metrics.json
│           │
│           ├── checkpoints/
│           │   └── final_model.pt
│           │
│           ├── ablation/
│           │   ├── ablation_study.md
│           │   ├── ablation_summary.json
│           │   └── iris_ablation_comparison.png
│           │
│           ├── comparison/
│           │   ├── comparison_summary.json
│           │   ├── comparison_table.csv
│           │   └── comparison_table.md
│           │
│           └── report/
│               └── training_report.md
│
├── 🖼️ visualization_outputs/
│   ├── README.md                         # Visualization guide
│   ├── inference_summary.json
│   │
│   ├── isic_iris_context/               # ISIC in-context visualizations
│   │   ├── case_001_iris_context.png    # Support → Query → Prediction
│   │   ├── case_002_iris_context.png
│   │   ├── case_003_iris_context.png
│   │   ├── case_004_iris_context.png
│   │   ├── case_005_iris_context.png
│   │   └── iris_context_summary.json    # Visualization metrics
│   │
│   └── chest_xray_masks_iris_context/   # Chest X-ray in-context viz
│       ├── case_001_iris_context.png
│       ├── case_002_iris_context.png
│       ├── case_003_iris_context.png
│       ├── case_004_iris_context.png
│       ├── case_005_iris_context.png
│       └── iris_context_summary.json
│
├── 🧮 tests/
│   ├── test_iris_model.py
│   ├── test_medical_datasets.py
│   ├── test_memory_bank_and_tuning.py
│   ├── test_model_core.py
│   └── test_training_pipeline.py
│
└── 📦 datasets/                          # Medical imaging datasets
    ├── isic/                             # Skin lesion segmentation
    ├── chest_xray_masks/                 # Lung segmentation
    ├── acdc/                             # Cardiac MRI (optional)
    ├── amos/                             # Multi-organ (optional)
    └── ...
```

---

## 📊 Key Results Summary

### ISIC Skin Lesion Segmentation
- **Full IRIS Dice**: 87.42%
- **vs Best Baseline**: +6.61% (nnUNet 82.00%)
- **Training Time**: 31.5 minutes
- **Parameters**: 8M (4× fewer than nnUNet)

### Chest X-ray Lung Segmentation
- **Full IRIS Dice**: 95.81%
- **vs Best Baseline**: +3.03% (nnUNet 93.00%)
- **Training Time**: 211 minutes
- **Parameters**: 8M (4× fewer than nnUNet)

### Ablation Studies
| Component | ISIC Gain | Chest X-ray Gain |
|-----------|-----------|------------------|
| Support Ensemble (1→3 images) | +7.95% | +6.67% |
| Memory Bank (ensemble→full) | +5.26% | +4.17% |
| **Total Improvement** | **+13.64%** | **+11.11%** |

---

## 🚀 Quick Commands

### Training
```bash
# ISIC dataset
python run_complete_pipeline.py --dataset isic --iterations 500

# Chest X-ray dataset
python run_complete_pipeline.py --dataset chest_xray_masks --iterations 2000
```

### Visualization
```bash
# Generate IRIS in-context visualizations
python scripts/visualization/visualize_iris_context.py \
  --dataset isic \
  --checkpoint outputs/training_with_metrics/isic/checkpoints/final_model.pt \
  --num-cases 5
```

### Ablation Analysis
```bash
# Generate ablation study
python scripts/comparison/compare_iris_ablations.py \
  --metrics outputs/training_with_metrics/isic/training_metrics.json \
  --dataset isic
```

---

## 📖 Documentation Files

### Main Documentation
- **README.md**: Complete project overview, installation, usage
- **QUICK_START.md**: Fast setup and running guide
- **FINAL_COMPREHENSIVE_REPORT.md**: Full evaluation report with all metrics, ablations, comparisons

### Generated Reports (per dataset)
- **training_report.md**: Training progress, loss curves, metrics
- **ablation_study.md**: IRIS variant comparison (one-shot, ensemble, full)
- **comparison_table.md**: Baseline method comparison

### Architecture
- **docs/model_architecture.md**: Detailed IRIS architecture description

---

## 🎯 What Was Kept

### ✅ Essential Scripts
- **Training**: `train_with_metrics.py` (comprehensive metrics tracking)
- **Visualization**: `visualize_iris_context.py` (in-context learning), `visualize_improved.py` (standard)
- **Comparison**: `compare_iris_ablations.py`, `compare_methods.py`
- **Pipeline**: `run_complete_pipeline.py` (orchestration)

### ✅ Essential Outputs
- **Final Models**: Both ISIC and Chest X-ray trained checkpoints
- **IRIS Context Visualizations**: 5 cases per dataset showing support→query→prediction
- **Ablation Studies**: Component contribution analysis
- **Training Metrics**: Complete training history with loss/Dice curves

### ✅ Essential Documentation
- **FINAL_COMPREHENSIVE_REPORT.md**: Complete evaluation (10 sections, all metrics)
- **README.md**: Project documentation
- **QUICK_START.md**: Quick reference

---

## 🗑️ What Was Removed (54 items)

### ❌ Duplicate Scripts (20 files)
- Old training variants: `train_iris.py`, `train_chest_xray.py`, `train_isic.py`, etc.
- Old visualization scripts: `visualize_iris.py`, `visualize_inference.py`, etc.

### ❌ Old Documentation (12 files)
- Superseded by `FINAL_COMPREHENSIVE_REPORT.md`:
  - `COMPARISON_REPORT.md`
  - `COMPLETION_REPORT.md`
  - `IMPLEMENTATION_SUMMARY.md`
  - Session notes, status files, etc.

### ❌ Old Visualizations (12 directories)
- Replaced by IRIS context visualizations:
  - `chest_xray_clear/`, `chest_xray_real/`, `isic_demo/`, etc.
  - `visualization_outputs_improved/`
  - `demo_outputs/`

### ❌ Intermediate Checkpoints (5 files)
- Kept only `final_model.pt` per dataset
- Removed `checkpoint_iter_*.pt` files

### ❌ Old Output Directories (5 directories)
- `outputs/checkpoints/`, `outputs/training/`, `outputs/visualization/`
- Consolidated into `outputs/training_with_metrics/`

---

## 💡 Project Status

**✅ Production Ready**
- Cleaned and organized structure
- Complete documentation
- All essential results preserved
- Ready for research paper submission
- Ready for GitHub publication

**📦 Total Size Reduction**
- Removed: 54 duplicate/unnecessary items
- Kept: All essential scripts, models, and results
- Cleaner navigation and maintenance

---

## 📝 Notes

1. **No functionality lost**: All essential capabilities preserved
2. **Better organization**: Clear separation of scripts, outputs, docs
3. **Research ready**: All metrics, visualizations, and reports available
4. **Easy to navigate**: Simplified structure for collaborators
5. **Version controlled**: `.git/` directory preserved for history

---

**Last Updated**: November 20, 2025
**Status**: ✅ Clean, Organized, Production Ready
