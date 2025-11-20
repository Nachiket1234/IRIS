# IRIS Web Application

Interactive web interface for the IRIS (In-context Retrieval for Image Segmentation) medical image segmentation system.

## Features

### 🎯 Interactive Inference
- Upload medical images for real-time segmentation
- Choose between three IRIS variants:
  - **One-Shot**: Single support image (~2s inference)
  - **Ensemble**: Multiple support images averaged (~3s)
  - **Full IRIS + Tuning**: Memory bank + in-context tuning (~15s)
- Real-time metrics display (inference time, predicted volume)
- Visual overlay of predictions

### 📊 Training Analytics
- Interactive training curves (loss and Dice over iterations)
- Validation Dice progression with early stopping indicators
- Comprehensive metrics tables
- Performance comparison across variants

### 🔀 Variant Comparisons
- Browse pre-generated comparison visualizations
- Per-case performance heatmaps
- Dice score distribution across test cases

### 📈 Performance Dashboard
- Cross-dataset performance comparison
- Training time analysis
- Dataset-specific radar charts
- Memory bank contribution analysis

### 🖼️ IRIS Context Gallery
- Browse in-context learning visualizations
- Support images with ground truth overlays
- Step-by-step prediction workflow

## Installation

### 1. Install Dependencies

```powershell
cd web_app
pip install -r requirements.txt
```

### 2. Required Packages

- torch >= 2.0.0
- torchvision >= 0.15.0
- gradio >= 4.0.0
- plotly >= 5.14.0
- matplotlib >= 3.7.0
- numpy >= 1.24.0
- pillow >= 9.5.0
- pandas >= 2.0.0

## Usage

### Launch the Application

```powershell
python app.py
```

The application will start at `http://localhost:7860`

### Command Line Options

You can modify the launch parameters in `app.py`:

```python
demo.launch(
    share=True,           # Set to True to create public link
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True
)
```

## Available Datasets

The application automatically detects available trained models from:

- **Kvasir**: Colonoscopy polyp segmentation
- **DRIVE**: Retinal vessel segmentation
- **Brain Tumor**: MRI tumor segmentation
- **ISIC**: Skin lesion segmentation
- **Chest X-Ray**: Lung segmentation

## Directory Structure

```
web_app/
├── app.py                  # Main Gradio application
├── inference.py            # IRIS model inference wrapper
├── metrics_analyzer.py     # Metrics loading and analysis
├── chart_utils.py          # Plotly/Matplotlib charting
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Model Requirements

The application expects trained models at:
```
outputs/training_with_metrics/{dataset}/checkpoints/final_model.pt
```

And visualization outputs at:
```
visualization_outputs/{dataset}_variants_comparison/
visualization_outputs/{dataset}_iris_context/
```

## Performance Notes

### Inference Speed
- **One-Shot**: ~2 seconds (1 support image, no memory bank)
- **Ensemble**: ~3 seconds (3 support images averaged)
- **Full IRIS + Tuning**: ~15 seconds (5 support images + 20 tuning steps)

### GPU vs CPU
- **GPU**: Real-time inference (<5s for ensemble)
- **CPU**: Slower but functional (10-30s for ensemble)

The application automatically detects CUDA availability and uses GPU when available.

## Metrics Displayed

### Training Metrics
- Total training time (minutes)
- Total iterations (default: 1000)
- Final training loss
- Best validation Dice score

### Variant Performance
- One-shot Dice score
- Ensemble Dice score
- Full IRIS Dice score
- Ensemble vs one-shot improvement (%)
- Memory bank contribution (%)

### Real-Time Inference
- Inference time (seconds)
- Predicted volume (voxels)
- Dice score (if ground truth available)

## Customization

### Add New Dataset

1. Add configuration to `inference.py`:
```python
DATASET_CONFIGS = {
    'your_dataset': {
        'root': 'datasets/your_dataset_path',
        'volume_shape': (16, 256, 256),
        'modality': 'CT',  # or 'MRI', 'Fundus', etc.
        'display_name': 'Your Dataset Name'
    }
}
```

2. Ensure trained model exists at:
```
outputs/training_with_metrics/your_dataset/checkpoints/final_model.pt
```

### Modify Chart Styles

Edit `chart_utils.py` to customize Plotly charts:
- Color schemes
- Chart types
- Layout options

## Troubleshooting

### Model Loading Errors
- Ensure checkpoint files exist in correct locations
- Check dataset name matches folder name exactly
- Verify model was trained with same volume_shape

### Visualization Not Found
- Run visualization scripts to generate missing images
- Check `visualization_outputs/` directory structure

### CUDA Out of Memory
- Reduce batch size (currently 1 for inference)
- Use CPU mode
- Close other GPU applications

### Gradio Version Issues
- Ensure gradio >= 4.0.0
- Update with: `pip install --upgrade gradio`

## Example Usage

### Upload Your Own Image

1. Go to **Interactive Inference** tab
2. Upload a medical image (grayscale or RGB)
3. Select model/dataset matching your image type
4. Choose variant (start with "Ensemble" for good balance)
5. Click "Run Segmentation"

### Compare Datasets

1. Go to **Performance Dashboard** tab
2. View cross-dataset comparison chart
3. Select specific dataset for detailed radar chart
4. Compare training times and Dice scores

### Explore Visualizations

1. Go to **Variant Comparisons** or **IRIS Context Gallery**
2. Select dataset from dropdown
3. Use slider to browse through test cases
4. Observe performance differences across variants

## Citation

If you use IRIS in your research, please cite:

```bibtex
@article{iris2024,
  title={IRIS: In-context Retrieval for Image Segmentation},
  author={...},
  journal={...},
  year={2024}
}
```

## License

See main repository LICENSE file.
