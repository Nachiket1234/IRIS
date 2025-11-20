"""
Generate visualizations comparing IRIS variants:
1. One-shot: Single support image
2. Context Ensemble: 3 support images averaged
3. Full IRIS: 5 support images + memory bank
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iris.model import IrisModel
from iris.data import build_dataset, DatasetSplit


def load_model(checkpoint_path: Path, volume_shape: tuple, device: str = 'cuda') -> IrisModel:
    """Load trained IRIS model."""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = IrisModel(
        in_channels=1,
        base_channels=16,
        num_query_tokens=8,
        num_attention_heads=8,
        volume_shape=volume_shape,
        use_memory_bank=True,
        memory_momentum=0.999
    ).to(device)
    
    # Load state dict directly (checkpoint is already state dict, not wrapped)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def get_support_images(dataset, num_support: int, exclude_idx: int = None):
    """Get random support images from training set."""
    train_indices = [i for i in range(len(dataset)) if i != exclude_idx]
    support_indices = np.random.choice(train_indices, num_support, replace=False)
    
    support_images = []
    support_masks = []
    
    for idx in support_indices:
        data = dataset[idx]
        support_images.append(data['image'])
        support_masks.append(data['mask'])
    
    return support_images, support_masks


def predict_with_variant(model, query_img, support_images, support_masks, 
                         variant: str, device: str = 'cuda'):
    """
    Make prediction using specific IRIS variant.
    
    Args:
        variant: 'oneshot', 'ensemble', or 'full'
    """
    with torch.no_grad():
        if variant == 'oneshot':
            # Use only first support image, no memory bank
            supp_img = support_images[0].unsqueeze(0).to(device)
            supp_mask = support_masks[0].unsqueeze(0).to(device)
            
            if supp_mask.dim() == 4:
                supp_mask = supp_mask.unsqueeze(1)
            
            support_out = model.encode_support(supp_img, supp_mask)
            task_embeddings = support_out["task_embeddings"]
            
        elif variant == 'ensemble':
            # Average embeddings from multiple support images, no memory bank
            all_embeddings = []
            num_support = min(3, len(support_images))
            
            for i in range(num_support):
                supp_img = support_images[i].unsqueeze(0).to(device)
                supp_mask = support_masks[i].unsqueeze(0).to(device)
                
                if supp_mask.dim() == 4:
                    supp_mask = supp_mask.unsqueeze(1)
                
                support_out = model.encode_support(supp_img, supp_mask)
                all_embeddings.append(support_out["task_embeddings"])
            
            task_embeddings = torch.stack(all_embeddings).mean(dim=0)
            
        elif variant == 'full':
            # Full IRIS: 5 support images + memory bank
            all_embeddings = []
            num_support = min(5, len(support_images))
            
            for i in range(num_support):
                supp_img = support_images[i].unsqueeze(0).to(device)
                supp_mask = support_masks[i].unsqueeze(0).to(device)
                
                if supp_mask.dim() == 4:
                    supp_mask = supp_mask.unsqueeze(1)
                
                support_out = model.encode_support(supp_img, supp_mask)
                all_embeddings.append(support_out["task_embeddings"])
            
            task_embeddings = torch.stack(all_embeddings).mean(dim=0)
        
        # Run query through model
        query_img_batch = query_img.unsqueeze(0).to(device)
        outputs = model(query_img_batch, task_embeddings)
        logits = outputs["logits"]
        pred_mask = (torch.sigmoid(logits) > 0.5).float()
        
    return pred_mask.squeeze().cpu()


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Dice coefficient."""
    # Remove batch and channel dimensions if present
    while pred.dim() > 3:
        pred = pred.squeeze(0)
    while target.dim() > 3:
        target = target.squeeze(0)
    
    # Flatten to 1D
    pred = pred.flatten()
    target = target.flatten()
    
    # Ensure same size
    if pred.shape != target.shape:
        print(f"Warning: Shape mismatch - pred: {pred.shape}, target: {target.shape}")
        return 0.0
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    
    dice = (2.0 * intersection) / (union + 1e-8)
    return float(dice.clamp(0, 1))


def create_variant_comparison(query_img, gt_mask, pred_oneshot, pred_ensemble, 
                              pred_full, dice_scores, case_num: int, output_path: Path):
    """Create side-by-side comparison of all three variants."""
    
    # Handle both 3D and 4D tensors
    if query_img.dim() == 4:
        # (C, D, H, W) - take first channel and middle slice
        query_img = query_img[0]  # Now (D, H, W)
    
    # Get middle slice
    depth = query_img.shape[0]
    mid_slice = depth // 2
    
    img_slice = query_img[mid_slice].cpu().numpy()
    gt_slice = gt_mask[mid_slice].cpu().numpy() if gt_mask.dim() == 3 else gt_mask[0, mid_slice].cpu().numpy()
    oneshot_slice = pred_oneshot[mid_slice].cpu().numpy() if pred_oneshot.dim() == 3 else pred_oneshot[0, mid_slice].cpu().numpy()
    ensemble_slice = pred_ensemble[mid_slice].cpu().numpy() if pred_ensemble.dim() == 3 else pred_ensemble[0, mid_slice].cpu().numpy()
    full_slice = pred_full[mid_slice].cpu().numpy() if pred_full.dim() == 3 else pred_full[0, mid_slice].cpu().numpy()
    
    # Normalize image
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    
    # Create figure with 5 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Input, GT, One-shot
    axes[0, 0].imshow(img_slice, cmap='gray')
    axes[0, 0].set_title('Query Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_slice, cmap='gray')
    axes[0, 1].imshow(gt_slice, cmap='Greens', alpha=0.5)
    axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_slice, cmap='gray')
    axes[0, 2].imshow(oneshot_slice, cmap='Reds', alpha=0.5)
    axes[0, 2].set_title(f'One-shot\nDice: {dice_scores["oneshot"]:.3f}', 
                        fontsize=14, fontweight='bold', color='#D32F2F')
    axes[0, 2].axis('off')
    
    # Row 2: Ensemble, Full IRIS, Comparison
    axes[1, 0].imshow(img_slice, cmap='gray')
    axes[1, 0].imshow(ensemble_slice, cmap='Blues', alpha=0.5)
    axes[1, 0].set_title(f'Context Ensemble (3 support)\nDice: {dice_scores["ensemble"]:.3f}', 
                        fontsize=14, fontweight='bold', color='#1976D2')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_slice, cmap='gray')
    axes[1, 1].imshow(full_slice, cmap='Greens', alpha=0.5)
    axes[1, 1].set_title(f'Full IRIS (5 support + memory)\nDice: {dice_scores["full"]:.3f}', 
                        fontsize=14, fontweight='bold', color='#388E3C')
    axes[1, 1].axis('off')
    
    # Comparison panel showing all three overlaid
    comparison = np.zeros((*img_slice.shape, 3))
    comparison[oneshot_slice > 0.5] = [1, 0, 0]  # Red for one-shot
    comparison[ensemble_slice > 0.5] = [0, 0, 1]  # Blue for ensemble
    comparison[full_slice > 0.5] = [0, 1, 0]  # Green for full IRIS
    
    # Where all agree = white
    all_agree = (oneshot_slice > 0.5) & (ensemble_slice > 0.5) & (full_slice > 0.5)
    comparison[all_agree] = [1, 1, 1]
    
    axes[1, 2].imshow(img_slice, cmap='gray')
    axes[1, 2].imshow(comparison, alpha=0.6)
    axes[1, 2].set_title('Overlay Comparison\n(Red=One-shot, Blue=Ensemble, Green=Full, White=All)', 
                        fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add legend with improvements
    oneshot_dice = dice_scores["oneshot"]
    ensemble_dice = dice_scores["ensemble"]
    full_dice = dice_scores["full"]
    
    ensemble_improve = ((ensemble_dice - oneshot_dice) / oneshot_dice * 100) if oneshot_dice > 0 else 0
    full_improve = ((full_dice - oneshot_dice) / oneshot_dice * 100) if oneshot_dice > 0 else 0
    
    legend_text = f"""
    IRIS Variant Performance:
    
    One-shot:           {oneshot_dice:.3f}  (baseline)
    Context Ensemble:   {ensemble_dice:.3f}  ({ensemble_improve:+.1f}%)
    Full IRIS:          {full_dice:.3f}  ({full_improve:+.1f}%)
    
    Ensemble gain:      {ensemble_improve:.1f}%
    Memory bank gain:   {(full_dice - ensemble_dice) / ensemble_dice * 100:.1f}%
    """
    
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=11, 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'IRIS Variants Comparison - Case {case_num:03d}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Visualize IRIS variants comparison")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of test cases")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = Path(f"visualization_outputs/{args.dataset}_variants_comparison")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("IRIS VARIANTS COMPARISON VISUALIZATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"Cases: {args.num_cases}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Determine volume shape
    if args.dataset == 'chest_xray_masks':
        volume_shape = (16, 128, 128)
    elif args.dataset == 'isic':
        volume_shape = (16, 256, 256)
    elif args.dataset == 'brain_tumor':
        volume_shape = (16, 256, 256)
    elif args.dataset == 'kvasir':
        volume_shape = (16, 256, 256)
    elif args.dataset == 'drive':
        volume_shape = (16, 256, 256)
    elif args.dataset == 'covid_ct':
        volume_shape = (128, 128, 128)
    else:
        volume_shape = (128, 128, 128)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, volume_shape, str(device))
    print("✓ Model loaded")
    
    # Build dataset
    print("\nBuilding dataset...")
    dataset_kwargs = {}
    if args.dataset == 'chest_xray_masks':
        dataset_kwargs['volume_shape'] = (16, 128, 128)
        dataset_kwargs['images_folder'] = "CXR_png"
        dataset_kwargs['masks_folder'] = "masks"
        dataset_kwargs['depth_slices'] = 16
        dataset_kwargs['target_resolution'] = 128
        data_root = "datasets/chest_xray_masks/Lung Segmentation"
    elif args.dataset == 'isic':
        dataset_kwargs['volume_shape'] = (16, 256, 256)
        data_root = "datasets/isic"
    elif args.dataset == 'brain_tumor':
        dataset_kwargs['volume_shape'] = (16, 256, 256)
        dataset_kwargs['images_folder'] = "images"
        dataset_kwargs['masks_folder'] = "masks"
        dataset_kwargs['depth_slices'] = 16
        dataset_kwargs['target_resolution'] = 256
        data_root = "datasets/brain-tumor-dataset-includes-the-mask-and-images/data/data"
    elif args.dataset == 'kvasir':
        dataset_kwargs['volume_shape'] = (16, 256, 256)
        dataset_kwargs['depth_slices'] = 16
        dataset_kwargs['target_resolution'] = 256
        data_root = "datasets/Kvasir-SEG Data (Polyp segmentation & detection)"
    elif args.dataset == 'drive':
        dataset_kwargs['volume_shape'] = (16, 256, 256)
        dataset_kwargs['depth_slices'] = 16
        dataset_kwargs['target_resolution'] = 256
        data_root = "datasets/drive-digital-retinal-images-for-vessel-extraction"
    elif args.dataset == 'covid_ct':
        dataset_kwargs['volume_shape'] = (128, 128, 128)
        dataset_kwargs['mask_type'] = "infection"
        dataset_kwargs['target_resolution'] = 128
        data_root = "datasets/COVID-19 CT scans"
    else:
        data_root = f"datasets/{args.dataset}"
    
    dataset = build_dataset(args.dataset, data_root, DatasetSplit.TRAIN, **dataset_kwargs)
    print(f"✓ Dataset loaded: {len(dataset)} samples (using train split for visualization)")
    
    # Generate visualizations
    print(f"\nGenerating {args.num_cases} variant comparisons...")
    
    all_results = []
    
    for i in range(args.num_cases):
        print(f"\nProcessing case {i+1}/{args.num_cases}...")
        
        # Get query image
        data = dataset[i]
        query_img = data['image']
        gt_mask = data['mask'].squeeze()
        
        # Get support images (more than needed)
        support_images, support_masks = get_support_images(dataset, num_support=5, exclude_idx=i)
        
        # Get predictions from all three variants
        pred_oneshot = predict_with_variant(model, query_img, support_images, support_masks, 
                                            'oneshot', str(device))
        pred_ensemble = predict_with_variant(model, query_img, support_images, support_masks, 
                                             'ensemble', str(device))
        pred_full = predict_with_variant(model, query_img, support_images, support_masks, 
                                         'full', str(device))
        
        # Compute Dice scores
        dice_oneshot = compute_dice(pred_oneshot, gt_mask)
        dice_ensemble = compute_dice(pred_ensemble, gt_mask)
        dice_full = compute_dice(pred_full, gt_mask)
        
        dice_scores = {
            'oneshot': dice_oneshot,
            'ensemble': dice_ensemble,
            'full': dice_full
        }
        
        print(f"  One-shot:  {dice_oneshot:.3f}")
        if dice_oneshot > 0:
            print(f"  Ensemble:  {dice_ensemble:.3f} ({((dice_ensemble-dice_oneshot)/dice_oneshot*100):+.1f}%)")
            print(f"  Full IRIS: {dice_full:.3f} ({((dice_full-dice_oneshot)/dice_oneshot*100):+.1f}%)")
        else:
            print(f"  Ensemble:  {dice_ensemble:.3f} (+{dice_ensemble:.3f})")
            print(f"  Full IRIS: {dice_full:.3f} (+{dice_full:.3f})")
        
        # Create visualization
        output_path = args.output_dir / f"case_{i+1:03d}_variants_comparison.png"
        create_variant_comparison(query_img, gt_mask, pred_oneshot, pred_ensemble, 
                                 pred_full, dice_scores, i+1, output_path)
        
        # Store results
        ensemble_improvement = ((dice_ensemble - dice_oneshot) / dice_oneshot * 100) if dice_oneshot > 0 else 0.0
        full_improvement = ((dice_full - dice_oneshot) / dice_oneshot * 100) if dice_oneshot > 0 else 0.0
        
        all_results.append({
            'case': i+1,
            'oneshot_dice': float(dice_oneshot),
            'ensemble_dice': float(dice_ensemble),
            'full_dice': float(dice_full),
            'ensemble_improvement': float(ensemble_improvement),
            'full_improvement': float(full_improvement)
        })
    
    # Compute averages
    avg_oneshot = np.mean([r['oneshot_dice'] for r in all_results])
    avg_ensemble = np.mean([r['ensemble_dice'] for r in all_results])
    avg_full = np.mean([r['full_dice'] for r in all_results])
    
    # Save summary
    summary = {
        'dataset': args.dataset,
        'num_cases': args.num_cases,
        'average_dice': {
            'oneshot': float(avg_oneshot),
            'ensemble': float(avg_ensemble),
            'full': float(avg_full)
        },
        'improvements': {
            'ensemble_vs_oneshot': float((avg_ensemble - avg_oneshot) / avg_oneshot * 100),
            'full_vs_oneshot': float((avg_full - avg_oneshot) / avg_oneshot * 100),
            'memory_bank_contribution': float((avg_full - avg_ensemble) / avg_ensemble * 100)
        },
        'cases': all_results
    }
    
    summary_path = args.output_dir / 'variants_comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nAverage Dice Scores:")
    print(f"  One-shot:          {avg_oneshot:.4f}")
    print(f"  Context Ensemble:  {avg_ensemble:.4f} ({((avg_ensemble-avg_oneshot)/avg_oneshot*100):+.2f}%)")
    print(f"  Full IRIS:         {avg_full:.4f} ({((avg_full-avg_oneshot)/avg_oneshot*100):+.2f}%)")
    
    print(f"\nComponent Contributions:")
    print(f"  Support Ensemble:  {((avg_ensemble-avg_oneshot)/avg_oneshot*100):+.2f}%")
    print(f"  Memory Bank:       {((avg_full-avg_ensemble)/avg_ensemble*100):+.2f}%")
    
    print(f"\n✓ Summary saved: {summary_path}")
    print(f"✓ All visualizations saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
