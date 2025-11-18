# Instructions for Using Real Medical Datasets

## Current Status

The visualization scripts have been updated to:
- ✅ Only show `support_reference.png` for in-context tuning (not all strategies)
- ✅ Track and save loss information for each case and strategy
- ✅ Much improved image quality (512x512 resolution, sharpening, 4x upscaling)
- ✅ **Require real datasets** - synthetic data is no longer used

## Downloading Real Datasets

### Option 1: ACDC Dataset (Recommended)
1. Visit: https://www.creatis.insa-lyon.fr/Challenge/acdc/
2. Register for the challenge
3. Download the training data
4. Extract to: `datasets/acdc/training/`
5. Expected structure:
   ```
   datasets/acdc/training/
   ├── patient001/
   │   ├── patient001_frame01.nii.gz
   │   ├── patient001_frame01_gt.nii.gz
   │   └── ...
   └── patient002/
       └── ...
   ```

### Option 2: AMOS Dataset
1. Visit: https://amos22.grand-challenge.org/
2. Register and download
3. Extract to: `datasets/amos/`
4. Expected structure:
   ```
   datasets/amos/
   ├── imagesTr/
   └── labelsTr/
   ```

### Option 3: MSD Pancreas
1. Visit: http://medicaldecathlon.com/
2. Download Task07_Pancreas.tar
3. Extract to: `datasets/msd_pancreas/`

### Option 4: SegTHOR
1. Visit: https://competitions.codalab.org/competitions/21145
2. Register and download
3. Extract to: `datasets/segthor/`

## After Downloading

1. **Verify dataset**:
   ```powershell
   $env:PYTHONPATH="${PWD}\src"
   python scripts/data/check_datasets.py
   ```

2. **Train on real data**:
   ```powershell
   python scripts/training/train_multi_dataset.py
   # or
   python scripts/run_all_training.py
   ```

3. **Generate visualizations**:
   ```powershell
   python scripts/visualization/visualize_multi_dataset.py --mode multi --num-cases 8
   ```

## Output Structure

After running visualization with real data:
```
outputs/visualization/multi_dataset/
├── acdc/
│   ├── case_01/
│   │   ├── 01_input.png (512x512, high quality)
│   │   ├── 02_one_shot/
│   │   │   └── output_prediction.png (no support_reference)
│   │   ├── 02_context_ensemble/
│   │   │   └── output_prediction.png
│   │   ├── 02_memory_retrieval/
│   │   │   └── output_prediction.png
│   │   └── 02_in_context_tuning/
│   │       ├── support_reference.png (ONLY here)
│   │       └── output_prediction.png
│   └── summary.json (with loss information)
└── ...
```

## Loss Information

Each case's `summary.json` now includes:
- `dice`: Dice score per strategy
- `loss`: Loss value per strategy (Dice + Cross-Entropy)
- `total_loss`: Sum of losses across all strategies
- `avg_dice`: Average Dice score

## Image Quality Improvements

- **Resolution**: 512x512 pixels (up from 64x64)
- **Upscaling**: 4x with bicubic interpolation
- **Sharpening**: Applied for better clarity
- **Contrast**: Percentile-based normalization (2-98 percentile)
- **DPI**: 300 DPI for print quality
- **Format**: PNG quality=100

## Note

The scripts will **not run with synthetic data**. You must download and place real medical datasets in the `datasets/` directory before running training or visualization.

