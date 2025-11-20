"""
Generate comprehensive training and evaluation report for publications/presentations.
Creates markdown report with all metrics, plots, and analysis.
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_training_metrics(metrics_path: Path) -> Dict:
    """Load training metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_loss_curves(metrics: Dict, output_dir: Path):
    """Generate loss curves plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    train_loss = metrics["metrics"]["training_loss"]
    iterations = [m["iteration"] for m in train_loss]
    losses = [m["value"] for m in train_loss]
    ax1.plot(iterations, losses, 'b-', linewidth=2, label='Training Loss')
    
    if metrics["metrics"]["validation_loss"]:
        val_loss = metrics["metrics"]["validation_loss"]
        val_iters = [m["iteration"] for m in val_loss]
        val_losses = [m["value"] for m in val_loss]
        ax1.plot(val_iters, val_losses, 'r-', linewidth=2, marker='o', label='Validation Loss')
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dice scores
    train_dice = metrics["metrics"]["training_dice"]
    dice_iters = [m["iteration"] for m in train_dice]
    dice_values = [m["value"] for m in train_dice]
    ax2.plot(dice_iters, dice_values, 'b-', linewidth=2, label='Training Dice')
    
    if metrics["metrics"]["validation_dice"]:
        val_dice = metrics["metrics"]["validation_dice"]
        val_dice_iters = [m["iteration"] for m in val_dice]
        val_dice_values = [m["value"] for m in val_dice]
        ax2.plot(val_dice_iters, val_dice_values, 'r-', linewidth=2, marker='o', label='Validation Dice')
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title('Dice Score Progress', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plot_path = output_dir / "loss_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Loss curves saved: {plot_path}")
    return plot_path


def plot_learning_rate(metrics: Dict, output_dir: Path):
    """Generate learning rate schedule plot."""
    lr_data = metrics["metrics"]["learning_rates"]
    iterations = [m["iteration"] for m in lr_data]
    lrs = [m["value"] for m in lr_data]
    
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, lrs, 'g-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plot_path = output_dir / "learning_rate.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Learning rate plot saved: {plot_path}")
    return plot_path


def generate_markdown_report(metrics: Dict, plots: Dict[str, Path], output_path: Path):
    """Generate comprehensive markdown report."""
    
    report = f"""# IRIS Model Training Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Executive Summary

This report presents the training results of the IRIS (Imaging Retrieval via In-context Segmentation) model on medical imaging data.

### Key Metrics

| Metric | Value |
|--------|-------|
| Total Training Time | {metrics['total_training_time_seconds']/60:.1f} minutes |
| Total Iterations | {metrics['total_iterations']} |
| Final Training Loss | {f"{metrics['final_train_loss']:.4f}" if metrics['final_train_loss'] is not None else 'N/A'} |
| Final Validation Loss | {f"{metrics['final_val_loss']:.4f}" if metrics['final_val_loss'] is not None else 'N/A'} |
| Best Validation Dice | {f"{metrics['best_val_dice']:.4f}" if metrics['best_val_dice'] is not None else 'N/A'} |

---

## Training Progress

### Loss Curves

![Loss Curves]({plots['loss_curves'].name})

The figure above shows the training and validation loss over time. 
"""

    # Add loss analysis
    if metrics["metrics"]["training_loss"]:
        train_losses = [m["value"] for m in metrics["metrics"]["training_loss"]]
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        reduction = ((initial_loss - final_loss) / initial_loss) * 100
        
        report += f"""
**Loss Analysis:**
- Initial Training Loss: {initial_loss:.4f}
- Final Training Loss: {final_loss:.4f}
- Loss Reduction: {reduction:.1f}%
"""

    # Add Dice analysis
    if metrics["metrics"]["validation_dice"]:
        val_dice = [m["value"] for m in metrics["metrics"]["validation_dice"]]
        best_dice = max(val_dice)
        final_dice = val_dice[-1]
        
        report += f"""
### Dice Score Performance

**Validation Dice Scores:**
- Best Dice Score: {best_dice:.4f}
- Final Dice Score: {final_dice:.4f}
- Improvement: {(final_dice - val_dice[0])*100:.1f}%
"""

    # Learning rate schedule
    report += f"""
---

## Training Configuration

### Learning Rate Schedule

![Learning Rate]({plots['learning_rate'].name})

"""

    # Memory usage if available
    if metrics["metrics"]["memory_usage"]:
        max_memory = max([m["memory_mb"] for m in metrics["metrics"]["memory_usage"]])
        report += f"""
### Resource Usage

- Peak GPU Memory: {max_memory:.1f} MB
"""

    # Detailed metrics table
    report += """
---

## Detailed Training Log

### Training Loss Progress

| Iteration | Training Loss | Training Dice | Validation Loss | Validation Dice |
|-----------|---------------|---------------|-----------------|-----------------|
"""

    # Add key checkpoint metrics
    val_dict = {m["iteration"]: (m["value"], None) for m in metrics["metrics"]["validation_loss"]}
    for val_dice in metrics["metrics"]["validation_dice"]:
        if val_dice["iteration"] in val_dict:
            loss_val, _ = val_dict[val_dice["iteration"]]
            val_dict[val_dice["iteration"]] = (loss_val, val_dice["value"])
    
    train_dict = {}
    for i, tloss in enumerate(metrics["metrics"]["training_loss"]):
        if i < len(metrics["metrics"]["training_dice"]):
            train_dict[tloss["iteration"]] = (tloss["value"], metrics["metrics"]["training_dice"][i]["value"])
    
    # Combine and sample
    all_iters = sorted(set(list(val_dict.keys()) + [m["iteration"] for m in metrics["metrics"]["training_loss"][::100]]))
    
    for iter_num in all_iters[-20:]:  # Last 20 entries
        train_loss, train_dice = train_dict.get(iter_num, (None, None))
        val_loss, val_dice = val_dict.get(iter_num, (None, None))
        
        row = f"| {iter_num} "
        row += f"| {train_loss:.4f} " if train_loss is not None else "| - "
        row += f"| {train_dice:.4f} " if train_dice is not None else "| - "
        row += f"| {val_loss:.4f} " if val_loss is not None else "| - "
        row += f"| {val_dice:.4f} " if val_dice is not None else "| - "
        row += "|"
        
        report += row + "\n"

    report += """
---

## Conclusions

"""

    if metrics['best_val_dice'] and metrics['best_val_dice'] > 0.7:
        report += "✓ **Model achieved good performance** (Dice > 0.7)\n\n"
    elif metrics['best_val_dice'] and metrics['best_val_dice'] > 0.5:
        report += "⚠ **Model shows moderate performance** (0.5 < Dice < 0.7)\n\n"
    else:
        report += "⚠ **Model requires further tuning** (Dice < 0.5)\n\n"

    if metrics['final_train_loss']:
        if metrics['final_val_loss'] and metrics['final_val_loss'] > metrics['final_train_loss'] * 1.5:
            report += "⚠ **Potential overfitting detected** - validation loss significantly higher than training loss\n\n"
        else:
            report += "✓ **No significant overfitting** - training and validation losses are comparable\n\n"

    report += """
### Next Steps

1. **Visualization Analysis:** Review output visualizations to understand model predictions qualitatively
2. **Hyperparameter Tuning:** Consider adjusting learning rate, batch size, or model capacity if needed
3. **Extended Training:** Train for more iterations if loss is still decreasing
4. **Dataset Expansion:** Include more training data if available
5. **Cross-validation:** Test on other medical imaging datasets for generalization

---

*Report generated by IRIS Training Pipeline*
"""

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n  ✓ Report saved: {output_path}")


def generate_full_report(metrics_path: Path, output_dir: Path = None):
    """Generate complete training report with plots and analysis."""
    
    if output_dir is None:
        output_dir = metrics_path.parent / "report"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING TRAINING REPORT")
    print("="*60 + "\n")
    
    # Load metrics
    print("Loading metrics...")
    metrics = load_training_metrics(metrics_path)
    print(f"  ✓ Loaded {metrics['total_iterations']} iterations of data\n")
    
    # Generate plots
    print("Generating plots...")
    plots = {}
    plots['loss_curves'] = plot_loss_curves(metrics, output_dir)
    plots['learning_rate'] = plot_learning_rate(metrics, output_dir)
    
    # Generate markdown report
    print("\nGenerating report...")
    report_path = output_dir / "training_report.md"
    generate_markdown_report(metrics, plots, report_path)
    
    print("\n" + "="*60)
    print("REPORT GENERATION COMPLETE")
    print("="*60)
    print(f"\nReport location: {report_path}")
    print(f"Plots location: {output_dir}")
    
    return report_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training report")
    parser.add_argument("--metrics", type=Path, required=True, help="Path to training_metrics.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for report")
    
    args = parser.parse_args()
    
    generate_full_report(args.metrics, args.output_dir)
