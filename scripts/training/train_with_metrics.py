"""
Comprehensive training script with detailed metrics tracking for reporting.
Tracks training loss, validation loss, Dice scores, learning rates, and more.
"""
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iris.data import DatasetSplit, build_dataset
from iris.model import IrisModel
from iris.model.tuning import DiceCrossEntropyLoss
from iris.training import EpisodicTrainingConfig, set_global_seed
from iris.training.lamb import Lamb


class MetricsTracker:
    """Track and save comprehensive training metrics."""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            "training_loss": [],
            "validation_loss": [],
            "training_dice": [],
            "validation_dice": [],
            "learning_rates": [],
            "iteration_times": [],
            "memory_usage": [],
            "timestamps": []
        }
        
        self.per_class_metrics = defaultdict(lambda: {"dice": [], "loss": []})
        self.start_time = time.time()
        self.best_val_dice = 0.0
        self.patience_counter = 0
    
    def log_iteration(self, iteration: int, train_loss: float, train_dice: float, lr: float):
        """Log training iteration metrics."""
        self.metrics["training_loss"].append({"iteration": iteration, "value": train_loss})
        self.metrics["training_dice"].append({"iteration": iteration, "value": train_dice})
        self.metrics["learning_rates"].append({"iteration": iteration, "value": lr})
        self.metrics["timestamps"].append({"iteration": iteration, "time": datetime.now().isoformat()})
        
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.metrics["memory_usage"].append({"iteration": iteration, "memory_mb": memory_mb})
    
    def log_validation(self, iteration: int, val_loss: float, val_dice: float, per_class_dice: Dict = None):
        """Log validation metrics."""
        self.metrics["validation_loss"].append({"iteration": iteration, "value": val_loss})
        self.metrics["validation_dice"].append({"iteration": iteration, "value": val_dice})
        
        if per_class_dice:
            for class_id, dice in per_class_dice.items():
                self.per_class_metrics[class_id]["dice"].append({"iteration": iteration, "value": dice})
    
    def check_early_stopping(self, val_dice: float, patience: int = 5) -> bool:
        """Check if training should stop early. Returns True if should stop."""
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                return True
            return False
    
    def save(self, filename="training_metrics.json"):
        """Save all metrics to JSON file."""
        output_path = self.save_dir / filename
        
        summary = {
            "total_training_time_seconds": time.time() - self.start_time,
            "total_iterations": len(self.metrics["training_loss"]),
            "final_train_loss": self.metrics["training_loss"][-1]["value"] if self.metrics["training_loss"] else None,
            "final_val_loss": self.metrics["validation_loss"][-1]["value"] if self.metrics["validation_loss"] else None,
            "best_val_dice": max([m["value"] for m in self.metrics["validation_dice"]]) if self.metrics["validation_dice"] else None,
            "metrics": self.metrics,
            "per_class_metrics": dict(self.per_class_metrics)
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Metrics saved to {output_path}")
        return summary
    
    def print_summary(self):
        """Print training summary to console."""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        
        if self.metrics["training_loss"]:
            train_losses = [m["value"] for m in self.metrics["training_loss"]]
            print(f"Training Loss:   {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
        
        if self.metrics["validation_loss"]:
            val_losses = [m["value"] for m in self.metrics["validation_loss"]]
            print(f"Validation Loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
        
        if self.metrics["training_dice"]:
            train_dice = [m["value"] for m in self.metrics["training_dice"]]
            print(f"Training Dice:   {train_dice[0]:.4f} → {train_dice[-1]:.4f}")
        
        if self.metrics["validation_dice"]:
            val_dice = [m["value"] for m in self.metrics["validation_dice"]]
            best_dice = max(val_dice)
            print(f"Validation Dice: {val_dice[0]:.4f} → {val_dice[-1]:.4f} (best: {best_dice:.4f})")
        
        elapsed = time.time() - self.start_time
        print(f"\nTotal Time: {elapsed/60:.1f} minutes")
        print("=" * 60)


def train_with_detailed_metrics(
    dataset_name: str = "chest_xray_masks",
    dataset_root: Path = None,
    iterations: int = 2000,
    eval_every: int = 200,
    output_dir: Path = None,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    max_samples: int = None,
    early_stopping_patience: int = 5
):
    """Train IRIS model with comprehensive metrics tracking."""
    
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path("outputs") / "training_with_metrics" / dataset_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize metrics tracker
    tracker = MetricsTracker(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Training on {dataset_name}")
    print(f"{'='*60}\n")
    
    # Load dataset
    try:
        # Dataset-specific configurations
        if dataset_name == "chest_xray_masks":
            train_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/chest_xray_masks/Lung Segmentation"),
                split=DatasetSplit.TRAIN,
                images_folder="CXR_png",
                masks_folder="masks",
                depth_slices=16,
                target_resolution=128,
                train_ratio=0.9
            )
            val_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/chest_xray_masks/Lung Segmentation"),
                split=DatasetSplit.VALID,
                images_folder="CXR_png",
                masks_folder="masks",
                depth_slices=16,
                target_resolution=128,
                train_ratio=0.9
            )
            volume_shape = (16, 128, 128)
        elif dataset_name == "isic":
            train_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/isic"),
                split=DatasetSplit.TRAIN,
                depth_slices=16,
                target_resolution=256,
                train_ratio=0.9
            )
            val_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/isic"),
                split=DatasetSplit.VALID,
                depth_slices=16,
                target_resolution=256,
                train_ratio=0.9
            )
            volume_shape = (16, 256, 256)
        elif dataset_name == "brain_tumor":
            train_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/brain-tumor-dataset-includes-the-mask-and-images/data/data"),
                split=DatasetSplit.TRAIN,
                images_folder="images",
                masks_folder="masks",
                depth_slices=16,
                target_resolution=256,
                train_ratio=0.9
            )
            val_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/brain-tumor-dataset-includes-the-mask-and-images/data/data"),
                split=DatasetSplit.VALID,
                images_folder="images",
                masks_folder="masks",
                depth_slices=16,
                target_resolution=256,
                train_ratio=0.9
            )
            volume_shape = (16, 256, 256)
        elif dataset_name == "drive":
            train_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/drive-digital-retinal-images-for-vessel-extraction"),
                split=DatasetSplit.TRAIN,
                depth_slices=16,
                target_resolution=256,
                train_ratio=0.9
            )
            val_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/drive-digital-retinal-images-for-vessel-extraction"),
                split=DatasetSplit.VALID,
                depth_slices=16,
                target_resolution=256,
                train_ratio=0.9
            )
            volume_shape = (16, 256, 256)
        elif dataset_name == "kvasir":
            train_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/Kvasir-SEG Data (Polyp segmentation & detection)"),
                split=DatasetSplit.TRAIN,
                depth_slices=16,
                target_resolution=256,
                train_ratio=0.9
            )
            val_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/Kvasir-SEG Data (Polyp segmentation & detection)"),
                split=DatasetSplit.VALID,
                depth_slices=16,
                target_resolution=256,
                train_ratio=0.9
            )
            volume_shape = (16, 256, 256)
        elif dataset_name == "covid_ct":
            train_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/COVID-19 CT scans"),
                split=DatasetSplit.TRAIN,
                mask_type="infection",
                target_resolution=128,
                train_ratio=0.9
            )
            val_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/COVID-19 CT scans"),
                split=DatasetSplit.VALID,
                mask_type="infection",
                target_resolution=128,
                train_ratio=0.9
            )
            volume_shape = (128, 128, 128)
        elif dataset_name == "acdc":
            train_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/acdc"),
                split=DatasetSplit.TRAIN
            )
            val_dataset = build_dataset(
                dataset_name,
                root=dataset_root or Path("datasets/acdc"),
                split=DatasetSplit.VALID
            )
            volume_shape = (128, 128, 128)
        else:
            train_dataset = build_dataset(dataset_name, root=dataset_root, split=DatasetSplit.TRAIN)
            val_dataset = build_dataset(dataset_name, root=dataset_root, split=DatasetSplit.VALID)
            volume_shape = (128, 128, 128)
        
        # Limit dataset size if requested
        if max_samples:
            train_size = min(max_samples, len(train_dataset))
            val_size = min(max_samples // 3, len(val_dataset))
            train_dataset = Subset(train_dataset, list(range(train_size)))
            val_dataset = Subset(val_dataset, list(range(val_size)))
        
        print(f"✓ Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples\n")
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None
    
    # Create model with volume_shape set during dataset loading
    model = IrisModel(
        in_channels=1,
        base_channels=16,
        num_query_tokens=8,
        volume_shape=volume_shape,
        use_memory_bank=True
    ).to(device)
    
    # Setup optimizer
    optimizer = Lamb(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        eps=1e-6
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    # Loss function
    criterion = DiceCrossEntropyLoss()
    
    # Training loop
    model.train()
    global_step = 0
    
    print("Starting training...\n")
    
    for iteration in range(iterations):
        # Sample random training example
        idx = torch.randint(0, len(train_dataset), (1,)).item()
        sample = train_dataset[idx]
        
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        
        # Binarize masks - convert any non-zero class to 1 (for multi-class datasets like ISIC)
        mask = (mask > 0).float()
        
        # Split into support and query
        support_img, query_img = image, image
        support_mask, query_mask = mask, mask
        
        # Ensure masks have correct shape: (B, K, D, H, W)
        if support_mask.dim() == 4:
            support_mask = support_mask.unsqueeze(1)  # Add class dimension
        if query_mask.dim() == 4:
            query_mask = query_mask.unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Encode support
        support_out = model.encode_support(support_img, support_mask)
        task_embeddings = support_out["task_embeddings"]
        
        # Decode query
        outputs = model(query_img, task_embeddings)
        logits = outputs["logits"]
        
        # Compute loss
        loss = criterion(logits, query_mask)
        
        # Compute Dice score
        with torch.no_grad():
            pred = torch.sigmoid(logits) > 0.5
            intersection = (pred * query_mask).sum()
            union = pred.sum() + query_mask.sum()
            if union > 0:
                dice = (2.0 * intersection / (union + 1e-6)).item()
                dice = max(0.0, min(1.0, dice))  # Clamp to [0, 1]
            else:
                dice = 0.0
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        tracker.log_iteration(iteration + 1, loss.item(), dice, current_lr)
        
        # Print progress
        if (iteration + 1) % 50 == 0:
            print(f"Iter {iteration+1}/{iterations} | Loss: {loss.item():.4f} | Dice: {dice:.4f} | LR: {current_lr:.6f}")
        
        # Validation
        if (iteration + 1) % eval_every == 0:
            model.eval()
            val_losses = []
            val_dices = []
            
            with torch.no_grad():
                for val_idx in range(min(10, len(val_dataset))):
                    val_sample = val_dataset[val_idx]
                    val_img = val_sample["image"].unsqueeze(0).to(device)
                    val_mask = val_sample["mask"].unsqueeze(0).to(device)
                    
                    # Binarize validation masks too
                    val_mask = (val_mask > 0).float()
                    
                    # Ensure mask has correct shape
                    if val_mask.dim() == 4:
                        val_mask = val_mask.unsqueeze(1)
                    
                    val_support_out = model.encode_support(val_img, val_mask)
                    val_outputs = model(val_img, val_support_out["task_embeddings"])
                    val_logits = val_outputs["logits"]
                    
                    val_loss = criterion(val_logits, val_mask)
                    val_pred = torch.sigmoid(val_logits) > 0.5
                    val_inter = (val_pred * val_mask).sum()
                    val_union = val_pred.sum() + val_mask.sum()
                    if val_union > 0:
                        val_dice = (2.0 * val_inter / (val_union + 1e-6)).item()
                        val_dice = max(0.0, min(1.0, val_dice))
                    else:
                        val_dice = 0.0
                    
                    val_losses.append(val_loss.item())
                    val_dices.append(val_dice)
            
            avg_val_loss = np.mean(val_losses)
            avg_val_dice = np.mean(val_dices)
            
            tracker.log_validation(iteration + 1, avg_val_loss, avg_val_dice)
            
            print(f"\n  Validation | Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f}\n")
            
            # Check early stopping
            if early_stopping_patience > 0 and tracker.check_early_stopping(avg_val_dice, early_stopping_patience):
                print(f"\n⚠ Early stopping triggered after {tracker.patience_counter} validations without improvement")
                print(f"  Best validation Dice: {tracker.best_val_dice:.4f}\n")
                model.train()
                break
            
            model.train()
        
        # Save checkpoint
        if (iteration + 1) % 500 == 0 or (iteration + 1) == iterations:
            checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration+1:06d}.pt"
            torch.save({
                'iteration': iteration + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'dice': dice
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path.name}")
        
        # Update learning rate
        if (iteration + 1) % 500 == 0:
            scheduler.step()
    
    # Save final metrics
    summary = tracker.save()
    tracker.print_summary()
    
    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Final model saved: {final_path}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chest_xray_masks", 
                       choices=["chest_xray_masks", "acdc", "isic", "brain_tumor", "drive", "kvasir", "covid_ct"])
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=5, 
                       help="Number of validations without improvement before stopping (0 to disable)")
    
    args = parser.parse_args()
    
    train_with_detailed_metrics(
        dataset_name=args.dataset,
        dataset_root=args.dataset_root,
        iterations=args.iterations,
        eval_every=args.eval_every,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        early_stopping_patience=args.early_stopping_patience
    )
