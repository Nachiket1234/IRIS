"""Chart generation utilities for IRIS web application."""

# Import pandas first to prevent circular import with plotly
try:
    import pandas as pd
except ImportError:
    pd = None

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import io
from PIL import Image


def create_training_curve_plotly(iterations: List[int], losses: List[float], 
                                  dice_scores: Optional[List[float]] = None,
                                  title: str = "Training Progress") -> go.Figure:
    """
    Create interactive training curve with Plotly.
    
    Args:
        iterations: List of iteration numbers
        losses: List of loss values
        dice_scores: Optional list of Dice scores
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add loss curve
    fig.add_trace(
        go.Scatter(x=iterations, y=losses, name="Training Loss",
                  line=dict(color='#1f77b4', width=2),
                  mode='lines'),
        secondary_y=False,
    )
    
    # Add Dice curve if available
    if dice_scores:
        fig.add_trace(
            go.Scatter(x=iterations, y=dice_scores, name="Validation Dice",
                      line=dict(color='#ff7f0e', width=2),
                      mode='lines'),
            secondary_y=True,
        )
    
    # Update axes
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    if dice_scores:
        fig.update_yaxes(title_text="Dice Score", secondary_y=True)
    
    fig.update_layout(
        title=title,
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(x=0.7, y=1.0)
    )
    
    return fig


def create_variant_comparison_chart(oneshot: float, ensemble: float, full: float,
                                     dataset: str = "Dataset") -> go.Figure:
    """
    Create bar chart comparing IRIS variants.
    
    Args:
        oneshot: One-shot Dice score
        ensemble: Ensemble Dice score
        full: Full IRIS Dice score
        dataset: Dataset name
        
    Returns:
        Plotly figure
    """
    variants = ['One-Shot', 'Ensemble (3 images)', 'Full IRIS + Memory Bank']
    scores = [oneshot * 100, ensemble * 100, full * 100]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    fig = go.Figure(data=[
        go.Bar(
            x=variants,
            y=scores,
            marker_color=colors,
            text=[f'{s:.2f}%' for s in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"IRIS Variant Performance - {dataset}",
        yaxis_title="Dice Score (%)",
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    fig.update_yaxes(range=[0, 100])
    
    return fig


def create_improvement_chart(ensemble_vs_oneshot: float, memory_bank_contrib: float,
                             dataset: str = "Dataset") -> go.Figure:
    """
    Create chart showing percentage improvements.
    
    Args:
        ensemble_vs_oneshot: Percentage improvement of ensemble over one-shot
        memory_bank_contrib: Percentage contribution of memory bank
        dataset: Dataset name
        
    Returns:
        Plotly figure
    """
    improvements = ['Ensemble vs One-Shot', 'Memory Bank Contribution']
    values = [ensemble_vs_oneshot, memory_bank_contrib]
    colors = ['#9b59b6', '#f39c12']
    
    fig = go.Figure(data=[
        go.Bar(
            x=improvements,
            y=values,
            marker_color=colors,
            text=[f'+{v:.2f}%' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"Performance Improvements - {dataset}",
        yaxis_title="Improvement (%)",
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def create_dataset_comparison_chart(comparison_data: Dict) -> go.Figure:
    """
    Create grouped bar chart comparing multiple datasets.
    
    Args:
        comparison_data: Dictionary with comparison data
        
    Returns:
        Plotly figure
    """
    datasets = comparison_data['datasets']
    
    fig = go.Figure(data=[
        go.Bar(name='One-Shot', x=datasets, y=comparison_data['oneshot_dice'],
               marker_color='#3498db'),
        go.Bar(name='Ensemble', x=datasets, y=comparison_data['ensemble_dice'],
               marker_color='#e74c3c'),
        go.Bar(name='Full IRIS', x=datasets, y=comparison_data['full_dice'],
               marker_color='#2ecc71'),
    ])
    
    fig.update_layout(
        title="Cross-Dataset Performance Comparison",
        yaxis_title="Dice Score (%)",
        barmode='group',
        template='plotly_white',
        height=500,
        legend=dict(x=0.7, y=1.0)
    )
    
    fig.update_yaxes(range=[0, 100])
    
    return fig


def create_training_time_chart(comparison_data: Dict) -> go.Figure:
    """
    Create bar chart showing training times.
    
    Args:
        comparison_data: Dictionary with comparison data
        
    Returns:
        Plotly figure
    """
    datasets = comparison_data['datasets']
    times = comparison_data['training_times']
    
    fig = go.Figure(data=[
        go.Bar(
            x=datasets,
            y=times,
            marker_color='#16a085',
            text=[f'{t:.1f}m' for t in times],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Training Time Comparison",
        yaxis_title="Time (minutes)",
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def create_radar_chart(stats: Dict, dataset: str) -> go.Figure:
    """
    Create radar chart for dataset performance metrics.
    
    Args:
        stats: Statistics dictionary
        dataset: Dataset name
        
    Returns:
        Plotly figure
    """
    categories = ['Validation Dice', 'One-Shot', 'Ensemble', 'Full IRIS', 'Context Learning']
    
    values = [
        stats.get('best_val_dice', 0) * 100,
        stats.get('avg_oneshot_dice', 0) * 100,
        stats.get('avg_ensemble_dice', 0) * 100,
        stats.get('avg_full_dice', 0) * 100,
        stats.get('avg_context_dice', 0) * 100,
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.3)',
        line=dict(color='#2ecc71', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title=f"Performance Radar - {dataset}",
        template='plotly_white',
        height=500
    )
    
    return fig


def create_per_case_heatmap(cases: List[Dict], dataset: str) -> go.Figure:
    """
    Create heatmap showing per-case performance across variants.
    
    Args:
        cases: List of per-case results
        dataset: Dataset name
        
    Returns:
        Plotly figure
    """
    case_ids = [f"Case {c['case']}" for c in cases]
    
    # Create matrix: rows = cases, columns = variants
    matrix = []
    for case in cases:
        matrix.append([
            case.get('oneshot_dice', 0) * 100,
            case.get('ensemble_dice', 0) * 100,
            case.get('full_dice', 0) * 100,
        ])
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=['One-Shot', 'Ensemble', 'Full IRIS'],
        y=case_ids,
        colorscale='RdYlGn',
        text=[[f'{v:.1f}%' for v in row] for row in matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Dice (%)"),
    ))
    
    fig.update_layout(
        title=f"Per-Case Performance Heatmap - {dataset}",
        template='plotly_white',
        height=400,
        xaxis_title="Variant",
        yaxis_title="Test Case"
    )
    
    return fig


def create_metrics_table_html(stats: Dict) -> str:
    """
    Create HTML table with metrics.
    
    Args:
        stats: Statistics dictionary
        
    Returns:
        HTML string
    """
    html = """
    <div style="font-family: Arial, sans-serif; padding: 20px;">
        <h3>Training Metrics</h3>
        <table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
            <tr style="background-color: #f2f2f2;">
                <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Metric</th>
                <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Value</th>
            </tr>
            <tr>
                <td style="padding: 12px; border: 1px solid #ddd;">Total Iterations</td>
                <td style="padding: 12px; border: 1px solid #ddd;">{iterations}</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 12px; border: 1px solid #ddd;">Training Time</td>
                <td style="padding: 12px; border: 1px solid #ddd;">{time:.1f} minutes</td>
            </tr>
            <tr>
                <td style="padding: 12px; border: 1px solid #ddd;">Final Training Loss</td>
                <td style="padding: 12px; border: 1px solid #ddd;">{train_loss:.4f}</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 12px; border: 1px solid #ddd;">Best Validation Dice</td>
                <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold; color: #2ecc71;">{val_dice:.2f}%</td>
            </tr>
        </table>
        
        <h3>Variant Performance</h3>
        <table style="border-collapse: collapse; width: 100%;">
            <tr style="background-color: #f2f2f2;">
                <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Variant</th>
                <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Dice Score</th>
                <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Improvement</th>
            </tr>
            <tr>
                <td style="padding: 12px; border: 1px solid #ddd;">One-Shot</td>
                <td style="padding: 12px; border: 1px solid #ddd;">{oneshot:.2f}%</td>
                <td style="padding: 12px; border: 1px solid #ddd;">-</td>
            </tr>
            <tr style="background-color: #f9f9f9;">
                <td style="padding: 12px; border: 1px solid #ddd;">Ensemble</td>
                <td style="padding: 12px; border: 1px solid #ddd;">{ensemble:.2f}%</td>
                <td style="padding: 12px; border: 1px solid #ddd; color: #3498db;">+{ensemble_imp:.2f}%</td>
            </tr>
            <tr>
                <td style="padding: 12px; border: 1px solid #ddd;">Full IRIS + Memory Bank</td>
                <td style="padding: 12px; border: 1px solid #ddd; font-weight: bold;">{full:.2f}%</td>
                <td style="padding: 12px; border: 1px solid #ddd; color: #2ecc71; font-weight: bold;">+{memory:.2f}%</td>
            </tr>
        </table>
    </div>
    """.format(
        iterations=stats.get('total_iterations', 0),
        time=stats.get('total_training_time', 0) / 60,
        train_loss=stats.get('final_train_loss', 0),
        val_dice=stats.get('best_val_dice', 0) * 100,
        oneshot=stats.get('avg_oneshot_dice', 0) * 100,
        ensemble=stats.get('avg_ensemble_dice', 0) * 100,
        full=stats.get('avg_full_dice', 0) * 100,
        ensemble_imp=stats.get('ensemble_improvement', 0),
        memory=stats.get('memory_bank_contribution', 0)
    )
    
    return html
