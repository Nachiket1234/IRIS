# 🚀 IRIS Quick Start Guide

## Run Everything in One Command

```powershell
cd "c:\Users\nachi\Downloads\IRIS V2 - Copy"
$env:PYTHONPATH = "${PWD}\src"
python run_complete_pipeline.py
```

That's it! The script will:
1. ✅ Train the model (2000 iterations with validation)
2. ✅ Generate 10 clear visualizations showing input/GT/predictions
3. ✅ Create comprehensive training report with plots

---

## What You'll Get

### 📊 Training Metrics
**File:** `outputs/training_with_metrics/chest_xray_masks/training_metrics.json`

Contains:
- Training & validation loss (every iteration)
- Dice scores
- Learning rates
- GPU memory usage
- Timestamps

### 🎨 Clear Visualizations
**Folder:** `visualization_outputs/chest_xray_masks_clear/`

Each image shows 4 panels:
```
┌──────────────┬──────────────┐
│ Input Image  │ Ground Truth │
│              │ (Green)      │
├──────────────┼──────────────┤
│  Prediction  │  Comparison  │
│  (Red)       │ (Yellow=Good)│
└──────────────┴──────────────┘
```

**Color Code:**
- 🟢 **Green** = Ground Truth only
- 🔴 **Red** = Prediction only
- 🟡 **Yellow** = Correct overlap

### 📝 Training Report
**File:** `outputs/training_with_metrics/chest_xray_masks/report/training_report.md`

Includes:
- Executive summary
- Loss curves (PNG plots)
- Dice progression
- Performance analysis
- Next steps

---

## Individual Steps (Optional)

If you want to run steps separately:

### Step 1: Training Only
```powershell
python scripts/training/train_with_metrics.py `
  --dataset chest_xray_masks `
  --iterations 2000 `
  --eval-every 200 `
  --max-samples 50
```

### Step 2: Visualization Only
```powershell
python scripts/visualization/visualize_improved.py `
  --dataset chest_xray_masks `
  --checkpoint outputs/training_with_metrics/chest_xray_masks/checkpoints/final_model.pt `
  --num-cases 10
```

### Step 3: Report Generation Only
```powershell
python scripts/generate_report.py `
  --metrics outputs/training_with_metrics/chest_xray_masks/training_metrics.json
```

---

## For ACDC Dataset

### 1. Download ACDC
```powershell
python scripts/download_acdc.py
```

Follow the instructions to manually download from:
https://www.creatis.insa-lyon.fr/Challenge/acdc/

### 2. Train on ACDC
Edit `run_complete_pipeline.py`:
```python
dataset = "acdc"  # Change from "chest_xray_masks"
```

Then run:
```powershell
python run_complete_pipeline.py
```

---

## Troubleshooting

### "CUDA out of memory"
```powershell
# Reduce sample size
python scripts/training/train_with_metrics.py --max-samples 30
```

### "Dataset not found"
Check that these exist:
- `datasets/chest_xray_masks/Lung Segmentation/CXR_png/`
- `datasets/chest_xray_masks/Lung Segmentation/masks/`

### Visualizations still unclear
Make sure you're using `scripts/visualization/visualize_improved.py`, not the old visualization script.

---

## Expected Output Locations

```
outputs/training_with_metrics/chest_xray_masks/
├── checkpoints/
│   ├── checkpoint_iter_000500.pt
│   ├── checkpoint_iter_001000.pt
│   ├── checkpoint_iter_001500.pt
│   ├── checkpoint_iter_002000.pt
│   └── final_model.pt
├── training_metrics.json
└── report/
    ├── training_report.md
    ├── loss_curves.png
    └── learning_rate.png

visualization_outputs/chest_xray_masks_clear/
├── case_001_comparison.png
├── case_002_comparison.png
├── ...
├── case_010_comparison.png
└── visualization_summary.json
```

---

## Training Time Estimate

With GTX 1650 (4GB):
- **50 samples, 2000 iterations:** ~20-30 minutes
- **Visualization generation:** ~2-3 minutes
- **Report generation:** <1 minute

**Total:** ~25-35 minutes

---

## Next Steps After Completion

1. **Review visualizations** in `visualization_outputs/chest_xray_masks_clear/`
2. **Read training report** at `outputs/.../report/training_report.md`
3. **Check metrics** in `training_metrics.json`
4. **Train on ACDC dataset** for comparison
5. **Tune hyperparameters** if needed

---

## For Your Report/Publication

You now have everything needed:
- ✅ Quantitative metrics (JSON + tables)
- ✅ Training curves (PNG plots)
- ✅ Visual results (high-res comparisons)
- ✅ Written analysis (Markdown report)

All ready to include in papers, presentations, or documentation!

---

**Questions?** See `README_TRAINING.md` for detailed documentation.
