"""IRIS Medical Image Segmentation Web Application."""

# Import pandas first to prevent circular import issues with plotly
try:
    import pandas as pd
except ImportError:
    pd = None

import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from inference import IRISInference
from metrics_analyzer import MetricsAnalyzer
from chart_utils import (
    create_training_curve_plotly,
    create_variant_comparison_chart,
    create_improvement_chart,
    create_dataset_comparison_chart,
    create_training_time_chart,
    create_radar_chart,
    create_per_case_heatmap,
    create_metrics_table_html
)

# Initialize analyzers
base_dir = Path(__file__).parent.parent
metrics_analyzer = MetricsAnalyzer(base_dir)

# Get available datasets
available_datasets = metrics_analyzer.get_available_datasets()
print(f"Available datasets: {available_datasets}")

# Initialize models dictionary (lazy loading)
models = {}

def get_model(dataset: str) -> IRISInference:
    """Get or create model instance for dataset."""
    if dataset not in models:
        checkpoint_path = f"outputs/training_with_metrics/{dataset}/checkpoints/final_model.pt"
        try:
            models[dataset] = IRISInference(checkpoint_path, dataset, base_dir)
        except Exception as e:
            raise ValueError(f"Failed to load model for {dataset}: {e}")
    return models[dataset]


def segment_image(image, dataset_choice, variant, num_support, tuning_steps, color_scheme):
    """
    Main inference function for Gradio interface.
    
    Args:
        image: PIL Image uploaded by user
        dataset_choice: Which model to use
        variant: "One-Shot", "Ensemble", or "Full IRIS + Tuning"
        num_support: Number of support images (1-5)
        tuning_steps: Number of tuning iterations (0-50)
        color_scheme: Color scheme for segmentation overlay
    
    Returns:
        (result_image, metrics_html)
    """
    if image is None:
        return None, "<p style='color: red;'>Please upload an image first.</p>"
    
    try:
        # Get model
        inference = get_model(dataset_choice)
        
        # Preprocess query image
        query_tensor = inference.preprocess_image(image)
        
        # Get support images
        support_imgs, support_masks = inference.get_support_images(num_support)
        
        # Run inference based on variant
        if variant == "One-Shot":
            prediction, inf_time = inference.predict_oneshot(
                query_tensor,
                support_imgs[0:1],
                support_masks[0:1]
            )
        elif variant == "Ensemble":
            prediction, inf_time = inference.predict_ensemble(
                query_tensor,
                support_imgs,
                support_masks,
                num_support=num_support
            )
        else:  # Full IRIS + Tuning
            # Get initial prediction
            initial_pred, _ = inference.predict_ensemble(
                query_tensor,
                support_imgs,
                support_masks,
                num_support=num_support
            )
            
            # Tune with initial prediction as target
            prediction, inf_time = inference.predict_with_tuning(
                query_tensor,
                initial_pred,
                support_imgs,
                support_masks,
                tuning_steps=tuning_steps,
                num_support=num_support
            )
        
        # Visualize with color scheme
        result_image = inference.visualize_prediction(query_tensor, prediction, color_scheme=color_scheme)
        
        # Create metrics HTML
        pred_volume = prediction.sum().item()
        metrics_html = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
            <h3 style="color: #2c3e50; margin-top: 0;">Inference Results</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Variant:</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #ddd;">{variant}</td>
                </tr>
                <tr style="background-color: #ffffff;">
                    <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Support Images:</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #ddd;">{num_support}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Tuning Steps:</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #ddd;">{tuning_steps if variant == "Full IRIS + Tuning" else "N/A"}</td>
                </tr>
                <tr style="background-color: #ffffff;">
                    <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Inference Time:</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #ddd; color: #3498db;"><strong>{inf_time:.3f}s</strong></td>
                </tr>
                <tr>
                    <td style="padding: 10px;"><strong>Predicted Volume:</strong></td>
                    <td style="padding: 10px; color: #2ecc71;"><strong>{pred_volume:.0f} voxels</strong></td>
                </tr>
            </table>
        </div>
        """
        
        return result_image, metrics_html
    
    except Exception as e:
        error_html = f"<p style='color: red;'>Error: {str(e)}</p>"
        return None, error_html


def load_visualization(dataset, vis_type, case_num):
    """Load pre-generated visualization."""
    vis_dir = base_dir / "visualization_outputs" / f"{dataset}_{vis_type}"
    image_path = vis_dir / f"case_{case_num:03d}_{vis_type}.png"
    
    if image_path.exists():
        return Image.open(image_path)
    return None


def update_training_curves(dataset):
    """Update training curves for selected dataset."""
    curves = metrics_analyzer.extract_training_curves(dataset)
    if curves is None:
        return None
    
    iterations, losses, dice_scores = curves
    
    if dice_scores:
        fig = create_training_curve_plotly(iterations, losses, dice_scores, 
                                          title=f"Training Progress - {dataset.upper()}")
    else:
        fig = create_training_curve_plotly(iterations, losses, 
                                          title=f"Training Loss - {dataset.upper()}")
    
    return fig


def update_variant_charts(dataset):
    """Update variant comparison charts."""
    stats = metrics_analyzer.get_summary_stats(dataset)
    
    if not stats.get('variant_comparison_available'):
        return None, None, create_metrics_table_html(stats)
    
    # Variant comparison bar chart
    variant_fig = create_variant_comparison_chart(
        stats.get('avg_oneshot_dice', 0),
        stats.get('avg_ensemble_dice', 0),
        stats.get('avg_full_dice', 0),
        dataset.upper()
    )
    
    # Improvement chart
    improvement_fig = create_improvement_chart(
        stats.get('ensemble_improvement', 0),
        stats.get('memory_bank_contribution', 0),
        dataset.upper()
    )
    
    # Metrics table
    table_html = create_metrics_table_html(stats)
    
    return variant_fig, improvement_fig, table_html


def update_performance_dashboard():
    """Update cross-dataset performance dashboard."""
    comparison = metrics_analyzer.compare_datasets(available_datasets)
    
    if not comparison['datasets']:
        return None, None, None
    
    # Dataset comparison chart
    comparison_fig = create_dataset_comparison_chart(comparison)
    
    # Training time chart
    time_fig = create_training_time_chart(comparison)
    
    # Radar chart for first dataset
    if comparison['datasets']:
        first_dataset = comparison['datasets'][0]
        stats = metrics_analyzer.get_summary_stats(first_dataset)
        radar_fig = create_radar_chart(stats, first_dataset.upper())
    else:
        radar_fig = None
    
    return comparison_fig, time_fig, radar_fig


def update_radar_for_dataset(dataset):
    """Update radar chart for specific dataset."""
    stats = metrics_analyzer.get_summary_stats(dataset)
    return create_radar_chart(stats, dataset.upper())


def update_heatmap(dataset):
    """Update per-case heatmap."""
    cases = metrics_analyzer.get_per_case_results(dataset)
    if not cases:
        return None
    
    return create_per_case_heatmap(cases, dataset.upper())


# Create Gradio interface
with gr.Blocks(title="IRIS Medical Image Segmentation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔬 IRIS: In-context Retrieval for Image Segmentation
    
    **Interactive web interface for medical image segmentation using IRIS model**
    
    Upload medical images and perform segmentation with different IRIS variants, or explore pre-generated visualizations and training analytics.
    """)
    
    with gr.Tabs():
        # ========== TAB 1: Training Analytics ==========
        with gr.Tab("📊 Training Analytics"):
            gr.Markdown("### Explore training progress and metrics across datasets")
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Upload Medical Image")
                    
                    dataset_select = gr.Dropdown(
                        choices=available_datasets,
                        value=available_datasets[0] if available_datasets else None,
                        label="Select Model/Dataset"
                    )
                    
                    variant_select = gr.Dropdown(
                        choices=["One-Shot", "Ensemble", "Full IRIS + Tuning"],
                        value="Ensemble",
                        label="IRIS Variant"
                    )
                    
                    with gr.Row():
                        num_support_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Number of Support Images"
                        )
                        tuning_steps_slider = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=20,
                            step=5,
                            label="Tuning Steps (Full IRIS only)"
                        )
                    
                    submit_btn = gr.Button("🚀 Run Segmentation", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    **Variant Descriptions:**
                    - **One-Shot**: Uses single support image (fastest, ~2s)
                    - **Ensemble**: Averages multiple support embeddings (~3s)
                    - **Full IRIS + Tuning**: Uses memory bank + in-context tuning (~15s)
                    """)
                
                with gr.Column(scale=1):
                    output_image = gr.Image(type="pil", label="Segmentation Result")
                    metrics_display = gr.HTML(label="Metrics")
            
            submit_btn.click(
                fn=segment_image,
                inputs=[input_image, dataset_select, variant_select, 
                       num_support_slider, tuning_steps_slider],
                outputs=[output_image, metrics_display]
            )
        
        # ========== TAB 2: Variant Comparisons ==========
        with gr.Tab("🔀 Variant Comparisons"):
            gr.Markdown("### Browse pre-generated variant comparison visualizations")
            
            with gr.Row():
                dataset_train = gr.Dropdown(
                    choices=available_datasets,
                    value=available_datasets[0] if available_datasets else None,
                    label="Select Dataset"
                )
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            training_curve_plot = gr.Plot(label="Training Curves")
            
            gr.Markdown("### Training Summary")
            with gr.Row():
                with gr.Column():
                    variant_chart = gr.Plot(label="Variant Performance Comparison")
                with gr.Column():
                    improvement_chart = gr.Plot(label="Performance Improvements")
            
            metrics_table = gr.HTML(label="Detailed Metrics")
            
            # Update on dataset change
            dataset_train.change(
                fn=update_training_curves,
                inputs=[dataset_train],
                outputs=[training_curve_plot]
            )
            
            dataset_train.change(
                fn=update_variant_charts,
                inputs=[dataset_train],
                outputs=[variant_chart, improvement_chart, metrics_table]
            )
            
            refresh_btn.click(
                fn=update_training_curves,
                inputs=[dataset_train],
                outputs=[training_curve_plot]
            )
            
            refresh_btn.click(
                fn=update_variant_charts,
                inputs=[dataset_train],
                outputs=[variant_chart, improvement_chart, metrics_table]
            )
            
            # Load initial data
            demo.load(
                fn=update_training_curves,
                inputs=[dataset_train],
                outputs=[training_curve_plot]
            )
            
            demo.load(
                fn=update_variant_charts,
                inputs=[dataset_train],
                outputs=[variant_chart, improvement_chart, metrics_table]
            )
        
        # ========== TAB 3: Performance Dashboard ==========
        with gr.Tab("📈 Performance Dashboard"):
            gr.Markdown("### Cross-dataset performance comparison")
            
            with gr.Row():
                dataset_comp = gr.Dropdown(
                    choices=available_datasets,
                    value=available_datasets[0] if available_datasets else None,
                    label="Dataset"
                )
                case_comp = gr.Slider(1, 5, value=1, step=1, label="Case Number")
            
            comparison_image = gr.Image(type="pil", label="Variant Comparison Visualization")
            
            gr.Markdown("### Per-Case Performance Heatmap")
            heatmap_plot = gr.Plot(label="Dice Scores Across Cases")
            
            def update_comparison(dataset, case):
                img = load_visualization(dataset, "variants_comparison", case)
                heatmap = update_heatmap(dataset)
                return img, heatmap
            
            dataset_comp.change(update_comparison, [dataset_comp, case_comp], 
                              [comparison_image, heatmap_plot])
            case_comp.change(update_comparison, [dataset_comp, case_comp], 
                           [comparison_image, heatmap_plot])
            
            # Load initial
            demo.load(update_comparison, [dataset_comp, case_comp], 
                     [comparison_image, heatmap_plot])
        
        # ========== TAB 4: IRIS Context Gallery ==========
        with gr.Tab("🖼️ IRIS Context Gallery"):
            gr.Markdown("### Browse IRIS in-context learning visualizations")
            
            refresh_dashboard_btn = gr.Button("🔄 Refresh Dashboard", size="sm")
            
            with gr.Row():
                cross_dataset_plot = gr.Plot(label="Cross-Dataset Dice Comparison")
                training_time_plot = gr.Plot(label="Training Time Comparison")
            
            gr.Markdown("### Dataset-Specific Radar Chart")
            
            with gr.Row():
                dataset_radar = gr.Dropdown(
                    choices=available_datasets,
                    value=available_datasets[0] if available_datasets else None,
                    label="Select Dataset for Radar Chart"
                )
            
            radar_plot = gr.Plot(label="Performance Radar")
            
            # Update functions
            def update_all_dashboard():
                comp_fig, time_fig, radar_fig = update_performance_dashboard()
                return comp_fig, time_fig, radar_fig
            
            refresh_dashboard_btn.click(
                fn=update_all_dashboard,
                outputs=[cross_dataset_plot, training_time_plot, radar_plot]
            )
            
            dataset_radar.change(
                fn=update_radar_for_dataset,
                inputs=[dataset_radar],
                outputs=[radar_plot]
            )
            
            # Load initial
            demo.load(
                fn=update_all_dashboard,
                outputs=[cross_dataset_plot, training_time_plot, radar_plot]
            )
        
        # ========== TAB 5: Interactive Inference ==========
        with gr.Tab("🎯 Interactive Inference (Custom Images)"):
            gr.Markdown("### Upload your own medical images for real-time segmentation")
            
            gr.Markdown("""
            These visualizations show the complete IRIS workflow:
            - **Top row**: Support images with ground truth masks (cyan overlay)
            - **Bottom row**: Query image → Prediction → Ground Truth → Comparison
            """)
            
            with gr.Row():
                dataset_ctx = gr.Dropdown(
                    choices=available_datasets,
                    value=available_datasets[0] if available_datasets else None,
                    label="Dataset"
                )
                case_ctx = gr.Slider(1, 5, value=1, step=1, label="Case Number")
            
            context_image = gr.Image(type="pil", label="IRIS Context Visualization")
            
            def update_context(dataset, case):
                return load_visualization(dataset, "iris_context", case)
            
            dataset_ctx.change(update_context, [dataset_ctx, case_ctx], context_image)
            case_ctx.change(update_context, [dataset_ctx, case_ctx], context_image)
            
            # Load initial
            demo.load(update_context, [dataset_ctx, case_ctx], context_image)
    
    gr.Markdown("""
    ---
    ## About IRIS
    
    **IRIS** (Imaging Retrieval via In-context Segmentation) is a medical image segmentation model that uses in-context learning with support images to adapt to new segmentation tasks without fine-tuning.
    
    ### Features:
    - ✅ **One-Shot Learning**: Segment with a single reference image
    - ✅ **Context Ensemble**: Improved accuracy with multiple support images
    - ✅ **Memory Bank**: Retrieval-augmented predictions using training exemplars
    - ✅ **In-Context Tuning**: Fine-tune task embeddings on-the-fly for optimal performance
    
    ### Available Datasets:
    """)
    
    for dataset in available_datasets:
        config = IRISInference.DATASET_CONFIGS.get(dataset, {})
        gr.Markdown(f"- **{config.get('display_name', dataset)}**")
    
    gr.Markdown("""
    ---
    **Developed with ❤️ for medical image analysis research**
    """)

# Launch app
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Launching IRIS Web Application")
    print("="*80)
    print(f"Available datasets: {', '.join(available_datasets)}")
    print(f"Base directory: {base_dir}")
    print("="*80 + "\n")
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        prevent_thread_lock=False,
        quiet=False,
        show_api=False,
        ssr_mode=False,
        allowed_paths=None
    )
