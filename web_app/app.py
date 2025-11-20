"""IRIS Medical Image Segmentation Web Application - Reorganized."""

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
                    <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Color Scheme:</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #ddd;">{color_scheme}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Inference Time:</strong></td>
                    <td style="padding: 10px; border-bottom: 1px solid #ddd; color: #3498db;"><strong>{inf_time:.3f}s</strong></td>
                </tr>
                <tr style="background-color: #ffffff;">
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


def update_training_curves(dataset):
    """Load and display training curves."""
    try:
        metrics = metrics_analyzer.load_training_metrics(dataset)
        if not metrics:
            return "<p style='color: red;'>No training data available</p>"
        
        total_iters = metrics.get('total_iterations', 0)
        final_loss = metrics.get('final_train_loss', 0)
        best_dice = metrics.get('best_val_dice', 0)
        time_mins = metrics.get('total_training_time_seconds', 0) / 60
        
        html = f"""
        <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3 style="color: #2c3e50; margin-top: 0;">{dataset.upper()} Training Progress</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div style="background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db;">
                    <p style="margin: 0; color: #7f8c8d; font-size: 14px;">Total Iterations</p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: bold; color: #2c3e50;">{total_iters}</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #e74c3c;">
                    <p style="margin: 0; color: #7f8c8d; font-size: 14px;">Final Training Loss</p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: bold; color: #2c3e50;">{final_loss:.4f}</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #2ecc71;">
                    <p style="margin: 0; color: #7f8c8d; font-size: 14px;">Best Validation Dice</p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: bold; color: #2ecc71;">{best_dice*100:.2f}%</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #f39c12;">
                    <p style="margin: 0; color: #7f8c8d; font-size: 14px;">Training Time</p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: bold; color: #2c3e50;">{time_mins:.1f} min</p>
                </div>
            </div>
        </div>
        """
        return html
    except Exception as e:
        print(f"Error loading training curves: {e}")
        import traceback
        traceback.print_exc()
        return f"<p style='color: red;'>Error: {e}</p>"


def update_variant_charts(dataset):
    """Load and display variant comparison charts."""
    try:
        stats = metrics_analyzer.get_summary_stats(dataset)
        variant_comp = metrics_analyzer.load_variant_comparison(dataset)
        
        if not variant_comp:
            return "<p>No data</p>", "<p>No data</p>", "<p>No variant comparison data available</p>"
        
        # Extract scores
        avg_dice = variant_comp.get('average_dice', {})
        oneshot = avg_dice.get('oneshot', 0) * 100
        ensemble = avg_dice.get('ensemble', 0) * 100
        full = avg_dice.get('full', 0) * 100
        
        # Extract improvements
        improvements = variant_comp.get('improvements', {})
        ensemble_vs_oneshot = improvements.get('ensemble_vs_oneshot', 0)
        memory_bank = improvements.get('memory_bank_contribution', 0)
        
        # Variant HTML
        variant_html = f"""
        <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3 style="color: #2c3e50; margin-top: 0;">Variant Performance - {dataset.upper()}</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                <div style="background: #3498db; padding: 20px; border-radius: 5px; text-align: center; color: white;">
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">One-Shot</p>
                    <p style="margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{oneshot:.2f}%</p>
                </div>
                <div style="background: #e74c3c; padding: 20px; border-radius: 5px; text-align: center; color: white;">
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">Ensemble (3 images)</p>
                    <p style="margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{ensemble:.2f}%</p>
                </div>
                <div style="background: #2ecc71; padding: 20px; border-radius: 5px; text-align: center; color: white;">
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">Full IRIS + Memory Bank</p>
                    <p style="margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{full:.2f}%</p>
                </div>
            </div>
        </div>
        """
        
        # Improvements HTML
        improvement_html = f"""
        <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3 style="color: #2c3e50; margin-top: 0;">Performance Improvements</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div style="background: #9b59b6; padding: 20px; border-radius: 5px; text-align: center; color: white;">
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">Ensemble vs One-Shot</p>
                    <p style="margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">+{ensemble_vs_oneshot:.2f}%</p>
                </div>
                <div style="background: #f39c12; padding: 20px; border-radius: 5px; text-align: center; color: white;">
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">Memory Bank Contribution</p>
                    <p style="margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{memory_bank:+.2f}%</p>
                </div>
            </div>
        </div>
        """
        
        # Metrics table
        table_html = create_metrics_table_html(stats)
        
        return variant_html, improvement_html, table_html
    except Exception as e:
        print(f"Error loading variant charts: {e}")
        import traceback
        traceback.print_exc()
        return f"<p style='color: red;'>Error: {e}</p>", "", ""


def update_performance_dashboard():
    """Create cross-dataset performance dashboard."""
    try:
        comparison_data = metrics_analyzer.compare_datasets(available_datasets)
        
        if not comparison_data or not comparison_data.get('datasets'):
            return "<p>No data</p>", "<p>No data</p>", "<p>No data</p>"
        
        datasets = comparison_data.get('datasets', [])
        oneshot = comparison_data.get('oneshot_dice', [])
        ensemble = comparison_data.get('ensemble_dice', [])
        full = comparison_data.get('full_dice', [])
        times = comparison_data.get('training_times', [])
        
        # Cross-dataset comparison HTML
        rows = ""
        for i, ds in enumerate(datasets):
            rows += f"""
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">{ds.upper()}</td>
                <td style="padding: 10px; border: 1px solid #ddd; background: #3498db; color: white; text-align: center;">{oneshot[i]:.2f}%</td>
                <td style="padding: 10px; border: 1px solid #ddd; background: #e74c3c; color: white; text-align: center;">{ensemble[i]:.2f}%</td>
                <td style="padding: 10px; border: 1px solid #ddd; background: #2ecc71; color: white; text-align: center;">{full[i]:.2f}%</td>
            </tr>
            """
        
        comparison_html = f"""
        <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3 style="color: #2c3e50; margin-top: 0;">Cross-Dataset Dice Comparison</h3>
            <table style="width: 100%; border-collapse: collapse; background: white;">
                <tr style="background: #34495e; color: white;">
                    <th style="padding: 12px; border: 1px solid #ddd;">Dataset</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">One-Shot</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">Ensemble</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">Full IRIS</th>
                </tr>
                {rows}
            </table>
        </div>
        """
        
        # Training times HTML
        time_rows = ""
        for i, ds in enumerate(datasets):
            if i < len(times):
                time_rows += f"""
                <div style="background: #16a085; color: white; padding: 15px; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                    <p style="margin: 0; font-size: 14px; opacity: 0.9;">{ds.upper()}</p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: bold;">{times[i]:.1f} min</p>
                </div>
                """
        
        time_html = f"""
        <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3 style="color: #2c3e50; margin-top: 0;">Training Time Comparison</h3>
            {time_rows}
        </div>
        """
        
        # Metrics for first dataset
        if datasets:
            stats = metrics_analyzer.get_summary_stats(datasets[0])
            metrics_html = f"""
            <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
                <h3 style="color: #2c3e50; margin-top: 0;">Performance Metrics - {datasets[0].upper()}</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                    <div style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
                        <p style="margin: 0; color: #7f8c8d; font-size: 12px;">Validation Dice</p>
                        <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold; color: #2c3e50;">{stats.get('best_val_dice', 0)*100:.2f}%</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
                        <p style="margin: 0; color: #7f8c8d; font-size: 12px;">One-Shot</p>
                        <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold; color: #3498db;">{stats.get('avg_oneshot_dice', 0)*100:.2f}%</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
                        <p style="margin: 0; color: #7f8c8d; font-size: 12px;">Ensemble</p>
                        <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold; color: #e74c3c;">{stats.get('avg_ensemble_dice', 0)*100:.2f}%</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
                        <p style="margin: 0; color: #7f8c8d; font-size: 12px;">Full IRIS</p>
                        <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold; color: #2ecc71;">{stats.get('avg_full_dice', 0)*100:.2f}%</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
                        <p style="margin: 0; color: #7f8c8d; font-size: 12px;">Context Learning</p>
                        <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold; color: #9b59b6;">{stats.get('avg_context_dice', 0)*100:.2f}%</p>
                    </div>
                </div>
            </div>
            """
        else:
            metrics_html = "<p>No data</p>"
        
        return comparison_html, time_html, metrics_html
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return f"<p style='color: red;'>Error: {e}</p>", "", ""


def update_radar_for_dataset(dataset):
    """Update metrics for specific dataset."""
    try:
        stats = metrics_analyzer.get_summary_stats(dataset)
        if not stats:
            return "<p>No data available</p>"
        
        html = f"""
        <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3 style="color: #2c3e50; margin-top: 0;">Performance Metrics - {dataset.upper()}</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                <div style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
                    <p style="margin: 0; color: #7f8c8d; font-size: 12px;">Validation Dice</p>
                    <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold; color: #2c3e50;">{stats.get('best_val_dice', 0)*100:.2f}%</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
                    <p style="margin: 0; color: #7f8c8d; font-size: 12px;">One-Shot</p>
                    <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold; color: #3498db;">{stats.get('avg_oneshot_dice', 0)*100:.2f}%</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
                    <p style="margin: 0; color: #7f8c8d; font-size: 12px;">Ensemble</p>
                    <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold; color: #e74c3c;">{stats.get('avg_ensemble_dice', 0)*100:.2f}%</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
                    <p style="margin: 0; color: #7f8c8d; font-size: 12px;">Full IRIS</p>
                    <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold; color: #2ecc71;">{stats.get('avg_full_dice', 0)*100:.2f}%</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
                    <p style="margin: 0; color: #7f8c8d; font-size: 12px;">Context Learning</p>
                    <p style="margin: 5px 0 0 0; font-size: 20px; font-weight: bold; color: #9b59b6;">{stats.get('avg_context_dice', 0)*100:.2f}%</p>
                </div>
            </div>
        </div>
        """
        return html
    except Exception as e:
        print(f"Error creating metrics: {e}")
        import traceback
        traceback.print_exc()
        return f"<p style='color: red;'>Error: {e}</p>"


def update_heatmap(dataset):
    """Create per-case performance table."""
    try:
        cases = metrics_analyzer.get_per_case_results(dataset)
        if not cases:
            return "<p>No per-case data available</p>"
        
        rows = ""
        for case in cases:
            case_num = case.get('case', 0)
            oneshot = case.get('oneshot_dice', 0) * 100
            ensemble = case.get('ensemble_dice', 0) * 100
            full = case.get('full_dice', 0) * 100
            
            rows += f"""
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Case {case_num}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center; background: rgba(52, 152, 219, 0.3);">{oneshot:.2f}%</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center; background: rgba(231, 76, 60, 0.3);">{ensemble:.2f}%</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center; background: rgba(46, 204, 113, 0.3);">{full:.2f}%</td>
            </tr>
            """
        
        html = f"""
        <div style="padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3 style="color: #2c3e50; margin-top: 0;">Per-Case Dice Scores - {dataset.upper()}</h3>
            <table style="width: 100%; border-collapse: collapse; background: white;">
                <tr style="background: #34495e; color: white;">
                    <th style="padding: 12px; border: 1px solid #ddd;">Case</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">One-Shot</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">Ensemble</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">Full IRIS</th>
                </tr>
                {rows}
            </table>
        </div>
        """
        return html
    except Exception as e:
        print(f"Error creating table: {e}")
        import traceback
        traceback.print_exc()
        return f"<p style='color: red;'>Error: {e}</p>"


def load_visualization(dataset, viz_type, case_num):
    """Load pre-generated visualization images."""
    try:
        # Try multiple possible paths
        paths = [
            base_dir / f"visualization_outputs/{dataset}_{viz_type}/case_{case_num:02d}.png",
            base_dir / f"outputs/visualization/{dataset}/{viz_type}_case_{case_num:02d}.png",
            base_dir / f"outputs/visualization/{dataset}/case_{case_num:02d}_{viz_type}.png",
        ]
        
        for path in paths:
            if path.exists():
                return Image.open(path)
        
        # Return placeholder if not found
        return None
    except Exception as e:
        print(f"Error loading visualization: {e}")
        return None


# Create Gradio interface
with gr.Blocks(title="IRIS Medical Image Segmentation") as demo:
    gr.Markdown("""
    # 🔬 IRIS: In-context Retrieval for Image Segmentation
    
    **Interactive web interface for medical image segmentation using IRIS model**
    
    Explore dataset analytics and training metrics, or upload custom medical images for real-time segmentation.
    """)
    
    with gr.Tabs():
        # ========== TAB 1: Training Analytics ==========
        with gr.Tab("📊 Training Analytics"):
            gr.Markdown("### Explore training progress and metrics across datasets")
            
            with gr.Row():
                dataset_train = gr.Dropdown(
                    choices=available_datasets,
                    value=available_datasets[0] if available_datasets else None,
                    label="Select Dataset"
                )
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            training_curve_plot = gr.HTML(label="Training Progress")
            
            gr.Markdown("### Training Summary")
            with gr.Row():
                with gr.Column():
                    variant_chart = gr.HTML(label="Variant Performance Comparison")
                with gr.Column():
                    improvement_chart = gr.HTML(label="Performance Improvements")
            
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
            
            # Load initial data when app starts
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
        
        # ========== TAB 2: Variant Comparisons ==========
        with gr.Tab("🔀 Variant Comparisons"):
            gr.Markdown("### Browse pre-generated variant comparison visualizations")
            
            with gr.Row():
                dataset_comp = gr.Dropdown(
                    choices=available_datasets,
                    value=available_datasets[0] if available_datasets else None,
                    label="Dataset"
                )
                case_comp = gr.Slider(1, 5, value=1, step=1, label="Case Number")
            
            comparison_image = gr.Image(type="pil", label="Variant Comparison Visualization")
            
            gr.Markdown("### Per-Case Performance Heatmap")
            heatmap_plot = gr.HTML(label="Dice Scores Across Cases")
            
            def update_comparison(dataset, case):
                img = load_visualization(dataset, "variants_comparison", case)
                heatmap = update_heatmap(dataset)
                return img, heatmap
            
            dataset_comp.change(update_comparison, [dataset_comp, case_comp], 
                              [comparison_image, heatmap_plot])
            case_comp.change(update_comparison, [dataset_comp, case_comp], 
                           [comparison_image, heatmap_plot])
            
            # Load initial data
            demo.load(update_comparison, [dataset_comp, case_comp], 
                     [comparison_image, heatmap_plot])
        
        # ========== TAB 3: Performance Dashboard ==========
        with gr.Tab("📈 Performance Dashboard"):
            gr.Markdown("### Cross-dataset performance comparison")
            
            refresh_dashboard_btn = gr.Button("🔄 Refresh Dashboard", size="sm")
            
            with gr.Row():
                cross_dataset_plot = gr.HTML(label="Cross-Dataset Dice Comparison")
                training_time_plot = gr.HTML(label="Training Time Comparison")
            
            gr.Markdown("### Dataset-Specific Radar Chart")
            
            with gr.Row():
                dataset_radar = gr.Dropdown(
                    choices=available_datasets,
                    value=available_datasets[0] if available_datasets else None,
                    label="Select Dataset for Radar Chart"
                )
            
            radar_plot = gr.HTML(label="Performance Metrics")
            
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
            
            # Load initial dashboard
            demo.load(
                fn=update_all_dashboard,
                outputs=[cross_dataset_plot, training_time_plot, radar_plot]
            )
        
        # ========== TAB 4: IRIS Context Gallery ==========
        with gr.Tab("🖼️ IRIS Context Gallery"):
            gr.Markdown("### Browse IRIS in-context learning visualizations")
            
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
            
            # Load initial context visualization
            demo.load(update_context, [dataset_ctx, case_ctx], context_image)
        
        # ========== TAB 5: Interactive Inference (Custom Images) ==========
        with gr.Tab("🎯 Interactive Inference (Custom Images)"):
            gr.Markdown("### Upload your own medical images for real-time segmentation")
            gr.Markdown("This tab allows you to test the IRIS model on your custom images, separate from the pre-computed dataset analytics above.")
            
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
                    
                    color_select = gr.Dropdown(
                        choices=["Green-Gold", "Blue-Cyan", "Red-Orange", "Purple-Pink", "Rainbow"],
                        value="Green-Gold",
                        label="Segmentation Color Scheme"
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
                    
                    **Color Schemes:**
                    - **Green-Gold**: Bright green to golden yellow
                    - **Blue-Cyan**: Ocean blue to cyan
                    - **Red-Orange**: Red to orange (classic)
                    - **Purple-Pink**: Purple to hot pink
                    - **Rainbow**: Multi-color gradient
                    """)
                
                with gr.Column(scale=1):
                    output_image = gr.Image(type="pil", label="Segmentation Result")
                    metrics_display = gr.HTML(label="Metrics")
            
            submit_btn.click(
                fn=segment_image,
                inputs=[input_image, dataset_select, variant_select, 
                       num_support_slider, tuning_steps_slider, color_select],
                outputs=[output_image, metrics_display]
            )
    
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
    print("Launching IRIS Web Application")
    print("="*80)
    print(f"Available datasets: {', '.join(available_datasets)}")
    print(f"Base directory: {base_dir}")
    print("="*80 + "\n")
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True
    )
