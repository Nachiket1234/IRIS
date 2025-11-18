# IRIS Inference Visualization Results

This directory contains comprehensive visualizations of IRIS model inference on 5 test cases, showing all four inference strategies with memory bank context information.

## Generated Files

### Visualization Images
- `case_01_inference.png` through `case_05_inference.png`: Detailed visualization panels for each test case

### Summary Data
- `inference_summary.json`: Complete metrics and results for all cases and strategies

## What Each Visualization Shows

Each PNG file contains a comprehensive 4-row × 5-column panel showing:

### Row 1: Input Data & Context
- **Column 1**: Query image (axial view)
- **Column 2**: Query image with ground truth overlay
- **Column 3**: Support/Reference image (used for one-shot inference)
- **Column 4**: Support image with mask overlay
- **Column 5**: Memory bank context information
  - Shows which classes are being segmented
  - Indicates if embeddings were retrieved from memory bank or computed fresh
  - Displays retrieval status

### Rows 2-4: Inference Strategy Results
For each of the three main strategies (one-shot, context ensemble, in-context tuning):

- **Column 1**: Prediction overlay on axial view
- **Column 2**: Prediction overlay on coronal view  
- **Column 3**: Prediction overlay on sagittal view
- **Column 4**: Dice scores per class and mean Dice
- **Column 5**: Strategy description and methodology

## Inference Strategies Evaluated

1. **One-Shot Inference** (Baseline)
   - Single reference image → task embedding → segmentation
   - Fastest inference method
   - Direct encoding without memory bank

2. **Context Ensemble**
   - Multiple reference images → averaged embeddings
   - More robust to reference selection
   - Uses ensemble of 3 support images

3. **Object Retrieval** (Memory Bank)
   - Retrieves stored embeddings from memory bank
   - Fast inference for seen classes
   - Falls back to one-shot if class not in memory

4. **In-Context Tuning**
   - Initializes from reference or memory bank
   - Gradient optimization of task embeddings only
   - Model parameters remain frozen
   - Best accuracy but slower inference

## Memory Bank Context

The visualizations show:
- **Classes being segmented**: Which anatomical structures/classes are present
- **Retrieval status**: Whether embeddings were retrieved from memory bank or computed fresh
- **Context information**: Details about the reference images used

Note: In the current setup, memory bank is disabled for compatibility, so all strategies use one-shot encoding. When enabled, the memory bank would show which stored embeddings were retrieved.

## Performance Metrics

Each visualization includes:
- **Per-class Dice scores**: Segmentation accuracy for each anatomical structure
- **Mean Dice score**: Overall segmentation performance
- **Inference time**: Computational cost for each strategy

## Dataset Information

- **Dataset Type**: Realistic synthetic 3D medical volumes (mimics CT scans)
- **Volume Shape**: 64×64×64 voxels
- **Number of Classes**: 3 anatomical structures per volume
- **Modality**: CT-like intensity distributions (Hounsfield units)

## Model Information

- **Base Channels**: 16 (reduced for laptop training)
- **Query Tokens**: 4
- **Attention Heads**: 4
- **Training**: 20 iterations on synthetic medical dataset
- **Checkpoint**: `demo_outputs/realistic_medical_training/checkpoints/iris_iter_000020.pt`

## Notes

- Dice scores are low because the model was trained for only 20 iterations on synthetic data
- For production use, train for 80,000 iterations on real medical datasets
- Memory bank functionality is demonstrated but currently disabled
- All visualizations show multi-planar views (axial, coronal, sagittal) for comprehensive 3D understanding

## How to Generate More Visualizations

Run the visualization script:
```bash
python visualize_inference.py
```

The script will:
1. Load the trained model checkpoint
2. Select 5 query images from the evaluation dataset
3. Run all inference strategies
4. Generate comprehensive visualization panels
5. Save results to this directory

## File Structure

```
visualization_outputs/
├── README.md                    # This file
├── case_01_inference.png        # Visualization for case 1
├── case_02_inference.png        # Visualization for case 2
├── case_03_inference.png        # Visualization for case 3
├── case_04_inference.png        # Visualization for case 4
├── case_05_inference.png        # Visualization for case 5
└── inference_summary.json       # Detailed metrics and results
```


