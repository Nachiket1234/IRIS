# Method Comparison - ISIC

| Method | Dice Score | Training Time (min) | Parameters | Description |
|--------|-----------|---------------------|------------|-------------|
| nnUNet | 0.8200 | 180.0 | 31M | Fully supervised |
| SAM-Med | 0.7500 | 90.0 | 93M | Prompted segmentation |
| MedSAM | 0.7800 | 120.0 | 93M | Medical SAM adaptation |
| Fine-tuning | 0.8000 | 60.0 | 24M | Standard transfer learning |
| IRIS (Ours) | 0.8742 | 31.5 | 8M | In-context learning + memory bank |


## Key Findings

- **IRIS Dice Score**: 0.8742
- **Average Baseline Dice**: 0.7875
- **Improvement**: +11.01%
- **Training Time**: 31.5 minutes
- **Model Size**: 8M parameters (3.9x smaller than nnUNet, 11.6x smaller than SAM variants)

## Advantages of IRIS

1. **Few-shot Learning**: Can adapt to new tasks with minimal examples
2. **Memory Bank**: Efficient storage and retrieval of task-specific knowledge
3. **In-context Tuning**: Fast adaptation without full retraining
4. **Efficiency**: Smaller model size with competitive performance
5. **Flexibility**: Works across different medical imaging modalities
