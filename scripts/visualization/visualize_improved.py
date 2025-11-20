"""
Improved visualization with clear, high-quality outputs showing distinct differences
between input, ground truth, and predictions.
"""
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iris.data import DatasetSplit, build_dataset
from iris.model import IrisModel


def normalize_image(img_array: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 range with contrast enhancement."""
    # Clip extreme values
    p2, p98 = np.percentile(img_array, [2, 98])
    img_array = np.clip(img_array, p2, p98)
    
    # Normalize to 0-1
    img_min, img_max = img_array.min(), img_array.max()
    if img_max > img_min:
        img_array = (img_array - img_min) / (img_max - img_min)
    
    # Convert to 0-255
    return (img_array * 255).astype(np.uint8)


def create_clear_comparison(
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    output_path: Path,
    case_name: str = "Case",
    dice_score: float = None
):
    """Create side-by-side comparison with clear visual differences."""
    
    # Extract middle slice
    img = image.detach().cpu().numpy().squeeze()
    gt = ground_truth.detach().cpu().numpy().squeeze()
    pred = prediction.detach().cpu().numpy().squeeze()
    
    # Handle 3D volumes
    if img.ndim == 3:
        img = img[img.shape[0] // 2]
        gt = gt[gt.shape[0] // 2]
        pred = torch.sigmoid(torch.from_numpy(pred)).numpy()
        pred = pred[pred.shape[0] // 2]
    else:
        pred = torch.sigmoid(torch.from_numpy(pred)).numpy()
    
    # Normalize image
    img_norm = normalize_image(img)
    
    # Create RGB versions
    h, w = img_norm.shape
    
    # 1. Input image (grayscale)
    input_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
    
    # 2. Ground truth overlay (green on image)
    gt_rgb = input_rgb.copy()
    gt_mask = (gt > 0.5).astype(np.uint8)
    gt_rgb[gt_mask == 1, 1] = 255  # Green channel
    gt_rgb[gt_mask == 1, 0] = gt_rgb[gt_mask == 1, 0] // 2  # Reduce red
    gt_rgb[gt_mask == 1, 2] = gt_rgb[gt_mask == 1, 2] // 2  # Reduce blue
    
    # 3. Prediction overlay (red on image)  
    pred_rgb = input_rgb.copy()
    pred_mask = (pred > 0.5).astype(np.uint8)
    pred_rgb[pred_mask == 1, 0] = 255  # Red channel
    pred_rgb[pred_mask == 1, 1] = pred_rgb[pred_mask == 1, 1] // 2  # Reduce green
    pred_rgb[pred_mask == 1, 2] = pred_rgb[pred_mask == 1, 2] // 2  # Reduce blue
    
    # 4. Comparison overlay (GT=green, Pred=red, Overlap=yellow)
    comparison_rgb = input_rgb.copy()
    overlap = (gt_mask * pred_mask).astype(bool)
    gt_only = (gt_mask & ~pred_mask).astype(bool)
    pred_only = (pred_mask & ~gt_mask).astype(bool)
    
    comparison_rgb[overlap, 0] = 255  # Yellow (overlap)
    comparison_rgb[overlap, 1] = 255
    comparison_rgb[overlap, 2] = 0
    
    comparison_rgb[gt_only, 1] = 255  # Green (GT only)
    comparison_rgb[gt_only, 0] = comparison_rgb[gt_only, 0] // 2
    comparison_rgb[gt_only, 2] = comparison_rgb[gt_only, 2] // 2
    
    comparison_rgb[pred_only, 0] = 255  # Red (Pred only)
    comparison_rgb[pred_only, 1] = comparison_rgb[pred_only, 1] // 2
    comparison_rgb[pred_only, 2] = comparison_rgb[pred_only, 2] // 2
    
    # Create 2x2 grid
    grid_h = h * 2 + 60  # Extra space for titles
    grid_w = w * 2 + 40
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    # Place images
    margin = 20
    grid[margin:margin+h, margin:margin+w] = input_rgb
    grid[margin:margin+h, margin+w+margin:margin+w+margin+w] = gt_rgb
    grid[margin+h+margin:margin+h+margin+h, margin:margin+w] = pred_rgb
    grid[margin+h+margin:margin+h+margin+h, margin+w+margin:margin+w+margin+w] = comparison_rgb
    
    # Convert to PIL and add text
    pil_img = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Add labels
    draw.text((margin + w//2 - 30, 5), "Input", fill=(0, 0, 0), font=font)
    draw.text((margin + w + margin + w//2 - 50, 5), "Ground Truth", fill=(0, 128, 0), font=font)
    draw.text((margin + w//2 - 35, margin + h + 10), "Prediction", fill=(128, 0, 0), font=font)
    draw.text((margin + w + margin + w//2 - 45, margin + h + 10), "Comparison", fill=(0, 0, 0), font=font)
    
    # Add title and dice score
    if dice_score is not None:
        title_text = f"{case_name} - Dice Score: {dice_score:.4f}"
    else:
        title_text = case_name
    
    # Save high-resolution output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img_upscaled = pil_img.resize((grid_w * 2, grid_h * 2), Image.Resampling.LANCZOS)
    pil_img_upscaled.save(output_path, dpi=(300, 300), quality=95)
    
    print(f"  ✓ Saved visualization: {output_path.name}")


def visualize_dataset_results(
    dataset_name: str,
    checkpoint_path: Path,
    dataset_root: Path = None,
    num_cases: int = 5,
    output_dir: Path = None
):
    """Generate clear visualizations for dataset."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup output dir
    if output_dir is None:
        output_dir = Path("visualization_outputs") / f"{dataset_name}_clear"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating Clear Visualizations: {dataset_name}")
    print(f"{'='*60}\n")
    
    # Dataset-specific configurations
    if dataset_name == "chest_xray_masks":
        volume_shape = (16, 128, 128)
        dataset_config = {
            "images_folder": "CXR_png",
            "masks_folder": "masks",
            "depth_slices": 16,
            "target_resolution": 128
        }
    elif dataset_name == "isic":
        volume_shape = (16, 256, 256)
        dataset_config = {
            "depth_slices": 16,
            "target_resolution": 256
        }
    elif dataset_name == "brain_tumor":
        volume_shape = (16, 256, 256)
        dataset_config = {
            "images_folder": "images",
            "masks_folder": "masks",
            "depth_slices": 16,
            "target_resolution": 256
        }
    elif dataset_name == "drive":
        volume_shape = (16, 256, 256)
        dataset_config = {
            "depth_slices": 16,
            "target_resolution": 256
        }
    elif dataset_name == "kvasir":
        volume_shape = (16, 256, 256)
        dataset_config = {
            "depth_slices": 16,
            "target_resolution": 256
        }
    elif dataset_name == "covid_ct":
        volume_shape = (128, 128, 128)
        dataset_config = {
            "mask_type": "infection",
            "target_resolution": 128
        }
    else:
        volume_shape = (128, 128, 128)
        dataset_config = {}
    
    # Load model - use appropriate volume shape
    model = IrisModel(
        in_channels=1,
        base_channels=16,
        num_query_tokens=8,
        volume_shape=volume_shape,
        use_memory_bank=True
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path.name}\n")
    
    # Load dataset
    try:
        # Build dataset
        if dataset_name == "chest_xray_masks":
            test_dataset = build_dataset(
                dataset_name,
                root=dataset_root or str(Path("datasets/chest_xray_masks/Lung Segmentation")),
                split=DatasetSplit.TEST,
                **dataset_config
            )
        elif dataset_name == "brain_tumor":
            test_dataset = build_dataset(
                dataset_name,
                root=dataset_root or str(Path("datasets/brain-tumor-dataset-includes-the-mask-and-images/data/data")),
                split=DatasetSplit.TEST,
                **dataset_config
            )
        elif dataset_name == "acdc":
            test_dataset = build_dataset(
                dataset_name,
                root=dataset_root or str(Path("datasets/acdc")),
                split=DatasetSplit.TEST,
                **dataset_config
            )
        else:
            test_dataset = build_dataset(
                dataset_name, 
                root=dataset_root, 
                split=DatasetSplit.TEST,
                **dataset_config
            )
        
        print(f"✓ Loaded {len(test_dataset)} test samples\n")
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # Generate visualizations
    results = []
    
    with torch.no_grad():
        for idx in range(min(num_cases, len(test_dataset))):
            print(f"Processing case {idx+1}/{min(num_cases, len(test_dataset))}...")
            
            sample = test_dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)
            
            # Ensure mask has correct shape: (B, K, D, H, W)
            if mask.dim() == 4:
                mask = mask.unsqueeze(1)  # Add class dimension
            
            # Inference
            support_out = model.encode_support(image, mask)
            outputs = model(image, support_out["task_embeddings"])
            logits = outputs["logits"]
            
            # Compute Dice
            pred = torch.sigmoid(logits) > 0.5
            intersection = (pred * mask).sum()
            union = pred.sum() + mask.sum()
            dice = (2.0 * intersection / (union + 1e-6)).item()
            
            # Create visualization
            output_path = output_dir / f"case_{idx+1:03d}_comparison.png"
            create_clear_comparison(
                image=image,
                ground_truth=mask,
                prediction=logits,
                output_path=output_path,
                case_name=f"Case {idx+1}",
                dice_score=dice
            )
            
            results.append({
                "case_id": idx + 1,
                "dice_score": dice,
                "output_file": str(output_path.name)
            })
    
    # Save summary
    summary_path = output_dir / "visualization_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "dataset": dataset_name,
            "num_cases": len(results),
            "average_dice": np.mean([r["dice_score"] for r in results]),
            "results": results
        }, f, indent=2)
    
    print(f"\n✓ Summary saved: {summary_path}")
    print(f"✓ All visualizations saved to: {output_dir}")
    print(f"\nAverage Dice Score: {np.mean([r['dice_score'] for r in results]):.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    
    args = parser.parse_args()
    
    visualize_dataset_results(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        num_cases=args.num_cases,
        output_dir=args.output_dir
    )
