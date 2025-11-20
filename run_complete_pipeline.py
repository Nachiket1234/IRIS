"""
Complete IRIS training and evaluation pipeline.
Runs training, visualization, and report generation in sequence.
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and report status."""
    print("\n" + "="*60)
    print(f"{description}")
    print("="*60 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False


def main():
    """Run complete pipeline."""
    
    print("\n" + "#"*60)
    print("# IRIS COMPLETE TRAINING & EVALUATION PIPELINE")
    print("#"*60)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run IRIS training pipeline')
    parser.add_argument('--dataset', type=str, default='isic', 
                        choices=['isic', 'chest_xray_masks', 'brain_tumor', 'drive', 'kvasir', 'covid_ct', 'acdc', 'amos', 'msd_pancreas', 'segthor'],
                        help='Dataset to train on')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum training samples (None for all)')
    args = parser.parse_args()
    
    # Configuration
    dataset = args.dataset
    
    # Set dataset-specific paths
    if dataset == "brain_tumor":
        dataset_root = Path("datasets/brain-tumor-dataset-includes-the-mask-and-images/data/data")
    elif dataset == "drive":
        dataset_root = Path("datasets/drive-digital-retinal-images-for-vessel-extraction")
    elif dataset == "kvasir":
        dataset_root = Path("datasets/Kvasir-SEG Data (Polyp segmentation & detection)")
    elif dataset == "covid_ct":
        dataset_root = Path("datasets/COVID-19 CT scans")
    else:
        dataset_root = Path(f"datasets/{dataset}")
    
    iterations = args.iterations
    eval_every = 200
    num_vis_cases = 10
    
    # Step 1: Train model with metrics tracking
    print("\n\n>>> STEP 1: Training Model with Metrics Tracking")
    train_cmd = [
        sys.executable,
        "scripts/training/train_with_metrics.py",
        "--dataset", dataset,
        "--dataset-root", str(dataset_root),
        "--iterations", str(iterations),
        "--eval-every", str(eval_every),
        "--lr", "0.001"
    ]
    
    if args.max_samples is not None:
        train_cmd.extend(["--max-samples", str(args.max_samples)])
    
    if not run_command(train_cmd, "Training"):
        print("\nTraining failed. Stopping pipeline.")
        return
    
    # Find the latest checkpoint
    output_dir = Path("outputs/training_with_metrics") / dataset
    checkpoint_dir = output_dir / "checkpoints"
    
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        print("\n✗ No checkpoints found. Stopping pipeline.")
        return
    
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"\n✓ Using checkpoint: {latest_checkpoint}")
    
    # Step 2: Generate clear visualizations
    print("\n\n>>> STEP 2: Generating Clear Visualizations")
    vis_cmd = [
        sys.executable,
        "scripts/visualization/visualize_improved.py",
        "--dataset", dataset,
        "--checkpoint", str(latest_checkpoint),
        "--dataset-root", str(dataset_root),
        "--num-cases", str(num_vis_cases)
    ]
    
    run_command(vis_cmd, "Visualization Generation")
    
    # Step 3: Generate comprehensive report
    print("\n\n>>> STEP 3: Generating Training Report")
    metrics_file = output_dir / "training_metrics.json"
    
    if metrics_file.exists():
        report_cmd = [
            sys.executable,
            "scripts/generate_report.py",
            "--metrics", str(metrics_file),
            "--output-dir", str(output_dir / "report")
        ]
        
        run_command(report_cmd, "Report Generation")
    else:
        print(f"\n✗ Metrics file not found: {metrics_file}")
    
    # Summary
    print("\n\n" + "#"*60)
    print("# PIPELINE COMPLETE")
    print("#"*60)
    print("\n📊 Results Summary:")
    print(f"  • Training outputs: {output_dir}")
    print(f"  • Model checkpoint: {latest_checkpoint}")
    print(f"  • Visualizations: visualization_outputs/{dataset}_clear/")
    print(f"  • Report: {output_dir}/report/training_report.md")
    
    print("\n📁 Key Files:")
    print(f"  • Metrics: {metrics_file}")
    print(f"  • Loss curves: {output_dir}/report/loss_curves.png")
    print(f"  • Training report: {output_dir}/report/training_report.md")
    
    print("\n✓ All tasks completed successfully!")
    print("\nNext steps:")
    print("  1. Review visualizations in visualization_outputs/")
    print("  2. Read the training report for detailed analysis")
    print("  3. Check model performance metrics")
    
    # Optional: ACDC dataset instructions
    print("\n" + "-"*60)
    print("To train on ACDC dataset:")
    print("  1. Run: python scripts/download_acdc.py")
    print("  2. Follow download instructions")
    print("  3. Update dataset='acdc' in this script")
    print("-"*60)


if __name__ == "__main__":
    main()
