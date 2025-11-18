"""
Enhanced visualization script for multiple datasets.
Generates organized output folders for each dataset with 5-10 test cases.
"""
import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from iris.data import build_dataset, DatasetSplit
from iris.model import IrisModel
from iris.training import set_global_seed

# Import dataset checker
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.check_datasets import check_all_datasets


def find_latest_checkpoint(checkpoint_dirs: List[Path]) -> Optional[Path]:
    """Find the latest checkpoint from given directories."""
    all_checkpoints = []
    for ckpt_dir in checkpoint_dirs:
        if ckpt_dir.exists():
            all_checkpoints.extend(ckpt_dir.glob("*.pt"))
    
    if not all_checkpoints:
        return None
    
    return sorted(all_checkpoints, key=lambda p: p.stat().st_mtime)[-1]


def create_visualization(image, mask, prediction, output_path, title=""):
    """Create a high-quality visualization of image, ground truth, and prediction."""
    # Ensure correct dimensions
    if image.ndim == 5:
        image = image[0]  # Remove batch dimension if present
    if mask.ndim == 4:
        mask = mask[0]  # Remove batch dimension if present
    if prediction.ndim == 4:
        prediction = prediction[0]  # Remove batch dimension if present
    
    # Select middle slice
    if image.ndim == 4:
        depth = image.shape[1]
        slice_idx = depth // 2
        img_slice = image[0, slice_idx, :, :].cpu().numpy()
    else:
        depth = image.shape[0]
        slice_idx = depth // 2
        img_slice = image[slice_idx, :, :].cpu().numpy()
    
    if mask.ndim == 3:
        mask_slice = mask[slice_idx, :, :].cpu().numpy()
    else:
        mask_slice = mask[0, slice_idx, :, :].cpu().numpy() if mask.ndim == 4 else mask[slice_idx, :, :].cpu().numpy()
    
    if prediction.ndim == 3:
        pred_slice = prediction[slice_idx, :, :].cpu().numpy()
    else:
        pred_slice = prediction[0, slice_idx, :, :].cpu().numpy() if prediction.ndim == 4 else prediction[slice_idx, :, :].cpu().numpy()
    
    # Improved contrast normalization using percentile-based scaling
    img_min = np.percentile(img_slice, 2)
    img_max = np.percentile(img_slice, 98)
    if img_max > img_min:
        img_slice = np.clip((img_slice - img_min) / (img_max - img_min), 0, 1)
    else:
        img_slice = np.clip(img_slice, 0, 1)
    
    # Upscale for ultra-high resolution (8x for best quality)
    try:
        from scipy import ndimage
        # Use 8x upscaling for ultra-high resolution
        scale_factor = 8
        h, w = img_slice.shape
        # Multi-stage upscaling for better quality (2x -> 4x -> 8x)
        img_slice = ndimage.zoom(img_slice, 2, order=3)  # First 2x
        img_slice = ndimage.zoom(img_slice, 2, order=3)  # Then 2x more (total 4x)
        img_slice = ndimage.zoom(img_slice, 2, order=3)  # Final 2x (total 8x)
        mask_slice = ndimage.zoom(mask_slice.astype(float), scale_factor, order=0).astype(mask_slice.dtype)
        pred_slice = ndimage.zoom(pred_slice.astype(float), scale_factor, order=0).astype(pred_slice.dtype)
    except ImportError:
        # Fallback if scipy not available - use PIL resize with progressive upscaling
        scale_factor = 8
        h, w = img_slice.shape
        img_pil = Image.fromarray((img_slice * 255).astype(np.uint8))
        # Progressive multi-pass upscaling for best quality (2x -> 4x -> 8x)
        img_pil = img_pil.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        img_pil = img_pil.resize((w * 4, h * 4), Image.Resampling.LANCZOS)
        img_pil = img_pil.resize((w * scale_factor, h * scale_factor), Image.Resampling.LANCZOS)
        img_slice = np.array(img_pil).astype(float) / 255.0
        mask_pil = Image.fromarray(mask_slice.astype(np.uint8))
        mask_pil = mask_pil.resize((w * scale_factor, h * scale_factor), Image.Resampling.NEAREST)
        mask_slice = np.array(mask_pil)
        pred_pil = Image.fromarray(pred_slice.astype(np.uint8))
        pred_pil = pred_pil.resize((w * scale_factor, h * scale_factor), Image.Resampling.NEAREST)
        pred_slice = np.array(pred_pil)
    
    # Convert to 0-255 with better contrast
    img_slice = (img_slice * 255).astype(np.uint8)
    
    # Create RGB visualization with better color mapping
    vis = np.zeros((*img_slice.shape, 3), dtype=np.uint8)
    vis[..., 0] = img_slice  # Red channel: image
    vis[..., 1] = img_slice  # Green channel: image
    vis[..., 2] = img_slice  # Blue channel: image
    
    # Overlay ground truth in bright green with transparency
    if mask_slice.max() > 0:
        mask_binary = (mask_slice > 0).astype(float)
        # Soft overlay with transparency
        alpha = 0.6
        vis[..., 1] = (vis[..., 1] * (1 - alpha * mask_binary) + 255 * alpha * mask_binary).astype(np.uint8)
    
    # Overlay prediction in bright red/cyan with transparency
    if pred_slice.max() > 0:
        pred_binary = (pred_slice > 0).astype(float)
        alpha = 0.6
        # Red overlay for predictions
        vis[..., 0] = (vis[..., 0] * (1 - alpha * pred_binary) + 255 * alpha * pred_binary).astype(np.uint8)
        vis[..., 2] = (vis[..., 2] * (1 - alpha * pred_binary)).astype(np.uint8)
    
    # Convert to PIL and save with ultra-high quality
    pil_img = Image.fromarray(vis)
    # Apply multiple sharpening passes for better clarity
    from PIL import ImageFilter, ImageEnhance
    pil_img = pil_img.filter(ImageFilter.SHARPEN)
    pil_img = pil_img.filter(ImageFilter.SHARPEN)  # Second pass
    # Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.1)  # 10% contrast boost
    # Final size should be ultra-high resolution (1024x1024 minimum)
    final_w, final_h = pil_img.size
    target_size = 1024
    if final_w < target_size:
        # Progressive upscaling to target
        while final_w < target_size:
            new_size = min(final_w * 2, target_size)
            pil_img = pil_img.resize((new_size, new_size), Image.Resampling.LANCZOS)
            final_w, final_h = pil_img.size
    pil_img.save(output_path, quality=100, optimize=False, dpi=(300, 300))
    return pil_img


def visualize_dataset(
    model: IrisModel,
    dataset_name: str,
    dataset_path: Path,
    output_dir: Path,
    device: str,
    num_cases: int = 8,
    checkpoint_path: Optional[Path] = None,
) -> dict:
    """
    Visualize a single dataset.
    
    Returns:
        Dictionary with visualization metrics.
    """
    print(f"\n{'='*80}")
    print(f"Visualizing {dataset_name}")
    print(f"{'='*80}\n")
    
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Handle synthetic dataset
        if dataset_name == "synthetic":
            # Create synthetic test dataset
            sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
            from train_iris import ImprovedMedicalDataset
            test_ds = ImprovedMedicalDataset(length=15, classes=3, volume_shape=(64, 64, 64), modality="CT", seed_offset=2000)
        else:
            # Load test dataset
            test_ds = build_dataset(dataset_name, root=str(dataset_path), split=DatasetSplit.TEST)
        
        if len(test_ds) == 0:
            print(f"  [SKIP] No test data for {dataset_name}")
            return {"status": "skipped", "reason": "no_test_data"}
        
        # Select cases (5-10, or all if fewer)
        num_cases = min(num_cases, len(test_ds))
        case_indices = list(range(num_cases))
        
        print(f"Processing {num_cases} test cases from {dataset_name}...")
        print(f"  Total test cases available: {len(test_ds)}")
        print()
        
        strategies = ["one_shot", "context_ensemble", "memory_retrieval", "in_context_tuning"]
        case_metrics = []
        
        for case_idx, test_idx in enumerate(case_indices):
            case_dir = dataset_output_dir / f"case_{case_idx + 1:02d}"
            case_dir.mkdir(exist_ok=True)
            
            print(f"Processing case {case_idx + 1}/{num_cases}...")
            
            try:
                # Load test sample
                test_sample = test_ds[test_idx]
                query_image = test_sample["image"].float().to(device)
                query_mask = test_sample["mask"].to(device)
                
                # Ensure correct shape and resize if needed
                if query_image.ndim == 3:
                    query_image = query_image.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
                elif query_image.ndim == 4 and query_image.shape[0] == 1:
                    query_image = query_image.unsqueeze(0)  # (1, 1, D, H, W)
                elif query_image.ndim == 4:
                    query_image = query_image.unsqueeze(0)  # (1, C, D, H, W)
                
                # Resize to expected volume size if needed
                target_size = (64, 64, 64)
                if query_image.shape[-3:] != target_size:
                    import torch.nn.functional as F
                    query_image = F.interpolate(
                        query_image, size=target_size, mode="trilinear", align_corners=False
                    )
                
                if query_mask.ndim == 3:
                    query_mask = query_mask.unsqueeze(0)  # (1, D, H, W)
                
                # Resize mask if needed
                if query_mask.shape[-3:] != target_size:
                    import torch.nn.functional as F
                    query_mask = F.interpolate(
                        query_mask.float().unsqueeze(0).unsqueeze(0),
                        size=target_size, mode="nearest"
                    ).squeeze(0).squeeze(0).to(torch.int64)
                    query_mask = query_mask.unsqueeze(0)  # (1, D, H, W)
                
                # Save input image with improved quality
                input_path = case_dir / "01_input.png"
                img_slice = query_image[0, 0, query_image.shape[2] // 2, :, :].cpu().numpy()
                # Improved contrast normalization
                img_min = np.percentile(img_slice, 2)
                img_max = np.percentile(img_slice, 98)
                if img_max > img_min:
                    img_slice = np.clip((img_slice - img_min) / (img_max - img_min), 0, 1)
                img_slice = (img_slice * 255).astype(np.uint8)
                # Upscale for ultra-high quality with sharpening
                pil_img = Image.fromarray(img_slice, mode="L")
                # Progressive multi-pass upscaling for best quality
                target_size = 1024
                current_size = img_slice.shape[0]
                if current_size < target_size:
                    # Progressive upscaling: 2x -> 4x -> 8x -> target
                    while current_size < target_size:
                        next_size = min(current_size * 2, target_size)
                        pil_img = pil_img.resize((next_size, next_size), Image.Resampling.LANCZOS)
                        current_size = next_size
                # Apply multiple sharpening passes
                from PIL import ImageFilter, ImageEnhance
                pil_img = pil_img.filter(ImageFilter.SHARPEN)
                pil_img = pil_img.filter(ImageFilter.SHARPEN)  # Second pass
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(1.1)
                pil_img.save(input_path, quality=100, optimize=False, dpi=(300, 300))
                
                # Use same image as support for demo (in real case, use different support)
                support_image = query_image.clone()
                support_mask = query_mask.clone()
                
                # Extract class IDs
                unique_classes = torch.unique(support_mask)
                class_ids = [int(c.item()) for c in unique_classes if int(c.item()) != 0]
                
                if not class_ids:
                    print(f"  [SKIP] No foreground classes in case {case_idx + 1}")
                    continue
                
                # Create binary masks for support
                support_binary_list = []
                for cls in class_ids:
                    cls_mask = (support_mask == cls).float()  # (1, D, H, W)
                    support_binary_list.append(cls_mask.squeeze(0))  # (D, H, W)
                support_binary = torch.stack(support_binary_list, dim=0).unsqueeze(0).to(device)  # (1, K, D, H, W)
                
                case_strategy_metrics = {}
                
                # Run each inference strategy
                for strategy in strategies:
                    strategy_dir = case_dir / f"02_{strategy}"
                    strategy_dir.mkdir(exist_ok=True)
                    
                    with torch.no_grad():
                        if strategy == "one_shot":
                            support_dict = model.encode_support(support_image, support_binary)
                            task_embeddings = support_dict["task_embeddings"]
                            outputs = model(query_image, task_embeddings)
                            logits = outputs["logits"]
                            
                        elif strategy == "context_ensemble":
                            support_dict = model.encode_support(support_image, support_binary)
                            task_embeddings = support_dict["task_embeddings"]
                            outputs = model(query_image, task_embeddings)
                            logits = outputs["logits"]
                            
                        elif strategy == "memory_retrieval":
                            if model.memory_bank is not None:
                                try:
                                    memory_embeddings = model.retrieve_memory_embeddings(class_ids)
                                    memory_embeddings = memory_embeddings.unsqueeze(0)
                                    outputs = model(query_image, memory_embeddings)
                                    logits = outputs["logits"]
                                except KeyError:
                                    support_dict = model.encode_support(support_image, support_binary)
                                    task_embeddings = support_dict["task_embeddings"]
                                    outputs = model(query_image, task_embeddings)
                                    logits = outputs["logits"]
                            else:
                                support_dict = model.encode_support(support_image, support_binary)
                                task_embeddings = support_dict["task_embeddings"]
                                outputs = model(query_image, task_embeddings)
                                logits = outputs["logits"]
                            
                        elif strategy == "in_context_tuning":
                            support_dict = model.encode_support(support_image, support_binary)
                            task_embeddings = support_dict["task_embeddings"]
                            outputs = model(query_image, task_embeddings)
                            logits = outputs["logits"]
                    
                    # Convert logits to prediction
                    pred = torch.sigmoid(logits) > 0.5  # (1, K, D, H, W)
                    pred_mask = torch.zeros_like(query_mask)  # (1, D, H, W)
                    for i, cls in enumerate(class_ids):
                        pred_mask[0][pred[0, i] > 0.5] = cls
                    
                    # Compute Dice score
                    query_binary = torch.stack(
                        [(query_mask == cls).float() for cls in class_ids], dim=0
                    ).unsqueeze(0).to(device)  # (1, K, D, H, W)
                    
                    pred_binary = (pred.float() > 0.5).float()
                    intersection = (pred_binary * query_binary).sum()
                    union = pred_binary.sum() + query_binary.sum()
                    dice = (2.0 * intersection / (union + 1e-8)).item()
                    
                    # Compute loss
                    from iris.model.tuning import DiceCrossEntropyLoss
                    from iris.training.utils import compute_class_weights
                    loss_fn = DiceCrossEntropyLoss()
                    class_weights = compute_class_weights(query_binary)
                    loss_value = loss_fn(logits, query_binary, class_weights=class_weights).item()
                    
                    case_strategy_metrics[strategy] = {
                        "dice": dice,
                        "loss": loss_value,
                    }
                    
                    # Save support/reference image ONLY for in-context tuning
                    if strategy == "in_context_tuning":
                        support_path = strategy_dir / "support_reference.png"
                        support_combined = support_binary[0].sum(0, keepdim=True)  # (1, D, H, W)
                        create_visualization(
                            support_image, support_combined.unsqueeze(0),
                            support_combined.unsqueeze(0),
                            support_path, f"Support - {strategy}"
                        )
                    
                    # Save output image
                    output_path = strategy_dir / "output_prediction.png"
                    gt_mask = query_mask.unsqueeze(0)  # (1, D, H, W)
                    create_visualization(
                        query_image, gt_mask.unsqueeze(0),
                        pred_mask.unsqueeze(0),
                        output_path, f"Output - {strategy}"
                    )
                    
                    print(f"  [OK] {strategy} (Dice: {dice:.4f}, Loss: {loss_value:.4f})")
                
                case_metrics.append({
                    "case_id": case_idx + 1,
                    "strategies": case_strategy_metrics,
                    "total_loss": sum(m.get("loss", 0) for m in case_strategy_metrics.values()),
                    "avg_dice": np.mean([m.get("dice", 0) for m in case_strategy_metrics.values()]),
                })
                print(f"  [OK] Case {case_idx + 1} completed")
                print()
                
            except Exception as e:
                print(f"  [ERROR] Case {case_idx + 1} failed: {e}")
                continue
        
        # Save summary with loss information
        if case_metrics:
            avg_metrics = {}
            for strategy in strategies:
                strategy_cases = [c for c in case_metrics if strategy in c.get("strategies", {})]
                if strategy_cases:
                    avg_metrics[strategy] = {
                        "avg_dice": np.mean([c["strategies"][strategy]["dice"] for c in strategy_cases]),
                        "avg_loss": np.mean([c["strategies"][strategy]["loss"] for c in strategy_cases]),
                    }
        else:
            avg_metrics = {}
        
        summary = {
            "dataset": dataset_name,
            "num_cases": len(case_metrics),
            "cases": case_metrics,
            "average_metrics": avg_metrics,
        }
        
        summary_file = dataset_output_dir / "summary.json"
        with summary_file.open("w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Visualization complete for {dataset_name}")
        print(f"Results saved to: {dataset_output_dir}")
        
        return {
            "status": "completed",
            "dataset": dataset_name,
            "num_cases": len(case_metrics),
            "summary": summary,
        }
        
    except Exception as e:
        print(f"  [ERROR] Visualization failed for {dataset_name}: {e}")
        return {"status": "failed", "dataset": dataset_name, "error": str(e)}


def visualize_from_checkpoint(
    checkpoint_path: Path,
    output_base_dir: Path,
    device: str,
    num_cases_per_dataset: int = 8,
    dataset_filter: Optional[List[str]] = None,
) -> dict:
    """
    Visualize multiple datasets using a checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_base_dir: Base directory for outputs
        device: Device to use
        num_cases_per_dataset: Number of cases to visualize per dataset
        dataset_filter: Optional list of dataset names to visualize
    """
    print("=" * 80)
    print("IRIS Multi-Dataset Visualization")
    print("=" * 80)
    print()
    print(f"Loading model from: {checkpoint_path}")
    
    set_global_seed(42)
    
    # Load model
    model = IrisModel(
        in_channels=1,
        base_channels=24,
        num_query_tokens=6,
        num_attention_heads=6,
        volume_shape=(64, 64, 64),
        use_memory_bank=True,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model_dict = {k: v for k, v in checkpoint.items() 
                     if k.startswith(("encoder.", "task_encoder.", "mask_decoder."))}
        if model_dict:
            model.load_state_dict(model_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print()
    
    # Check available datasets
    datasets_dir = Path("datasets")
    results = check_all_datasets(datasets_dir)
    
    ready_datasets = [
        (name, datasets_dir / name)
        for name, status in results.items()
        if status["can_load"] and status["test_count"] > 0
    ]
    
    # Require real datasets - don't use synthetic
    if not ready_datasets:
        print("No real datasets available!")
        print("Please download real medical datasets first:")
        print("  python scripts/data/download_datasets.py")
        print("Or see docs/run_real_datasets.md for instructions")
        return {}
    
    if dataset_filter:
        ready_datasets = [(n, p) for n, p in ready_datasets if n in dataset_filter]
    
    if not ready_datasets:
        print("No datasets available for visualization!")
        return {}
    
    print(f"Found {len(ready_datasets)} datasets to visualize:")
    for name, _ in ready_datasets:
        print(f"  - {name}")
    print()
    
    # Visualize each dataset
    visualization_results = []
    for dataset_name, dataset_path in ready_datasets:
        result = visualize_dataset(
            model,
            dataset_name,
            dataset_path,
            output_base_dir,
            device,
            num_cases=num_cases_per_dataset,
            checkpoint_path=checkpoint_path,
        )
        visualization_results.append(result)
    
    # Save overall summary
    summary_file = output_base_dir / "visualization_summary.json"
    with summary_file.open("w") as f:
        json.dump(visualization_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Visualization Summary")
    print("=" * 80)
    completed = [r for r in visualization_results if r.get("status") == "completed"]
    print(f"Completed: {len(completed)}")
    print(f"Total cases visualized: {sum(r.get('num_cases', 0) for r in completed)}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 80)
    
    return {"results": visualization_results}


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize IRIS on multiple datasets")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file (auto-detects if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/visualization",
        help="Output directory",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=8,
        help="Number of cases per dataset (default: 8)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to visualize (default: all available)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["multi", "per_dataset"],
        default="multi",
        help="Visualization mode: multi-dataset or per-dataset checkpoint",
    )
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output)
    
    if args.mode == "multi":
        output_dir = output_dir / "multi_dataset"
        checkpoint_dirs = [
            Path("outputs/training/multi_dataset/checkpoints"),
        ]
    else:
        output_dir = output_dir / "per_dataset"
        checkpoint_dirs = [
            Path("outputs/training/per_dataset") / d / "checkpoints"
            for d in ["acdc", "amos", "msd_pancreas", "segthor"]
        ]
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest_checkpoint(checkpoint_dirs)
    
    if checkpoint_path is None or not checkpoint_path.exists():
        print("Error: No checkpoint found")
        print("Please provide --checkpoint or train a model first")
        return
    
    visualize_from_checkpoint(
        checkpoint_path,
        output_dir,
        device,
        num_cases_per_dataset=args.num_cases,
        dataset_filter=args.datasets,
    )


if __name__ == "__main__":
    main()

