# IRIS Ablation Study - ISIC

## Method Comparison

| Method | Dice Score | Support Images | Memory Bank | Parameters | Description |
|--------|-----------|----------------|-------------|------------|-------------|
| One-shot | 0.7693 | 1 | ✗ | 8M | Single support image |
| Context Ensemble | 0.8305 | 3 | ✗ | 8M | Average 3 support embeddings |
| Full IRIS | 0.8742 | 5 | ✓ | 8M | Memory bank + in-context tuning |
| nnUNet | 0.8200 | 0 | ✗ | 31M | Fully supervised baseline |
| SAM-Med | 0.7500 | 0 | ✗ | 93M | Prompted segmentation |
| MedSAM | 0.7800 | 0 | ✗ | 93M | Medical SAM |

## Key Findings

### IRIS Variants Performance

1. **One-shot Learning**: 0.7693 Dice
   - Uses single support image
   - No memory bank
   - Baseline for few-shot capability

2. **Context Ensemble**: 0.8305 Dice
   - Averages embeddings from 3 support images
   - Improvement over one-shot: +7.95%
   - Still no memory bank

3. **Full IRIS**: 0.8742 Dice ⭐
   - Memory bank + in-context tuning
   - 5 support images with optimal selection
   - Improvement over one-shot: +13.64%
   - Improvement over ensemble: +5.26%

### Component Contributions

- **Support ensemble** (vs one-shot): +7.95%
- **Memory bank** (ensemble→full): +5.26%
- **Overall improvement** (one-shot→full): +13.64%

### Comparison with Baselines

- **Best Baseline**: 0.8200 (nnUNet or MedSAM)
- **Full IRIS**: 0.8742
- **Improvement**: +6.61%

### Efficiency Metrics

| Method | Dice/Param | Training Time | Total Efficiency |
|--------|-----------|---------------|------------------|
| One-shot | 0.09616 | 28.4 min | 0.02710 |
| Context Ensemble | 0.10381 | 30.0 min | 0.02772 |
| Full IRIS | 0.10927 | 31.5 min | 0.02772 |
| nnUNet | 0.02645 | 180.0 min | 0.00456 |
| SAM-Med | 0.00806 | 90.0 min | 0.00833 |
| MedSAM | 0.00839 | 120.0 min | 0.00650 |

## Ablation Analysis

### What Makes IRIS Work?

1. **Few-shot Learning Foundation**
   - Even one-shot achieves 0.7693 Dice
   - Demonstrates strong episodic training

2. **Context Ensemble Benefit**
   - Multiple support images provide robustness
   - Averaging reduces variance in task embeddings
   - Critical for handling diverse anatomy

3. **Memory Bank Advantage**
   - Stores optimal task representations
   - EMA updates maintain stability
   - Enables rapid adaptation without retraining

### Clinical Implications

- **One-shot mode**: Emergency scenarios with single example
- **Ensemble mode**: Standard clinical deployment with 3-5 examples
- **Full IRIS**: Research and high-accuracy applications

