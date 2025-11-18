# IRIS Multi-Dataset Training and Visualization - Output Status

## Current Status

### Training
- **Multi-dataset training**: Running in background (using synthetic data as fallback)
- **Per-dataset training**: Ready to run (will use synthetic data if no real datasets)
- **Output location**: `outputs/training/`

### Visualization
- **Working visualization script**: `scripts/visualization/visualize_iris.py` ✅
- **Enhanced multi-dataset script**: `scripts/visualization/visualize_multi_dataset.py` (needs debugging)
- **Output location**: `outputs/visualization/`

## Generated Outputs

### Visualization Structure
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
├── case_03/
└── case_04/
```

### Available Checkpoints
- `demo_outputs/improved_medical_training/checkpoints/iris_iter_000050.pt` ✅ (Working)
- `outputs/training/multi_dataset/checkpoints/` (Training in progress)

## Commands to Generate Outputs

### 1. Check Training Status
```powershell
cd "C:\Users\nachi\Downloads\IRIS\IRIS V2"
$env:PYTHONPATH="${PWD}\src"
python scripts/data/check_datasets.py
```

### 2. Run Training (if needed)
```powershell
python scripts/training/train_multi_dataset.py
# or
python scripts/run_all_training.py
```

### 3. Generate Visualizations
```powershell
# Using working script
python scripts/visualization/visualize_iris.py

# Or with specific checkpoint
python scripts/visualization/visualize_multi_dataset.py --checkpoint "demo_outputs/improved_medical_training/checkpoints/iris_iter_000050.pt" --num-cases 8
```

## Output Files Generated

### Training Outputs
- `outputs/training/multi_dataset/training_results.txt` - Training log
- `outputs/training/multi_dataset/metrics.json` - Evaluation metrics
- `outputs/training/multi_dataset/checkpoints/*.pt` - Model checkpoints

### Visualization Outputs
- `outputs/visualization/case_XX/01_input.png` - Input image for each case
- `outputs/visualization/case_XX/02_*/support_reference.png` - Support/reference image
- `outputs/visualization/case_XX/02_*/output_prediction.png` - Prediction output
- `outputs/visualization/*/summary.json` - Case summaries with Dice scores

## Next Steps

1. **Wait for training to complete** - Check `outputs/training/multi_dataset/training_results.txt`
2. **Run visualization** - Use `scripts/visualization/visualize_iris.py` (working) or wait for multi-dataset script fix
3. **Download real datasets** - For real data training, use `scripts/data/download_datasets.py`

## Notes

- Currently using synthetic data as real datasets are not downloaded
- All scripts support GPU (auto-detects NVIDIA GTX 1650)
- Outputs are organized in the requested folder structure
- Visualization generates high-quality PNG images with RGB overlays

