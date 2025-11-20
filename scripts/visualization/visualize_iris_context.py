"""
Visualize IRIS in-context learning with support/reference images.
Shows the complete workflow: Support Images → Query Image → Prediction
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iris.model import IrisModel
from iris.data import build_dataset, DatasetSplit


def normalize_image(img):
    """Normalize image to 0-255 range."""
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)


def create_iris_visualization(
    support_images,
    support_masks,
    query_image,
    query_mask,
    prediction,
    case_id,
):
    """
    Create visualization showing IRIS in-context learning:
    Row 1: Support images with masks (reference examples)
    Row 2: Query image → Prediction → Comparison with GT
    """
    
    # Extract middle slices
    def extract_slice(tensor):
        arr = tensor.detach().cpu().numpy().squeeze()
        if arr.ndim == 3:
            return arr[arr.shape[0] // 2]
        return arr
    
    query_img = extract_slice(query_image)
    query_gt = extract_slice(query_mask)
    pred_logits = extract_slice(prediction)
    pred_prob = torch.sigmoid(torch.from_numpy(pred_logits)).numpy()
    
    # Normalize
    query_img_norm = normalize_image(query_img)
    h, w = query_img_norm.shape
    
    # Number of support images to show (max 3 for space)
    num_support = min(3, len(support_images))
    
    # Create support image panels
    support_panels = []
    for i in range(num_support):
        supp_img = extract_slice(support_images[i])
        supp_mask = extract_slice(support_masks[i])
        
        supp_img_norm = normalize_image(supp_img)
        
        # Resize to match query size
        from scipy.ndimage import zoom
        if supp_img_norm.shape != (h, w):
            scale_h = h / supp_img_norm.shape[0]
            scale_w = w / supp_img_norm.shape[1]
            supp_img_norm = zoom(supp_img_norm, (scale_h, scale_w), order=1)
            supp_mask = zoom(supp_mask, (scale_h, scale_w), order=0)
        
        # Create RGB with mask overlay (cyan for support)
        supp_rgb = np.stack([supp_img_norm, supp_img_norm, supp_img_norm], axis=-1)
        mask_overlay = (supp_mask > 0.5).astype(bool)
        supp_rgb[mask_overlay, 0] = 0  # Cyan overlay
        supp_rgb[mask_overlay, 1] = 255
        supp_rgb[mask_overlay, 2] = 255
        
        support_panels.append(supp_rgb)
    
    # Create query and prediction panels
    query_rgb = np.stack([query_img_norm, query_img_norm, query_img_norm], axis=-1)
    
    # Prediction overlay (red)
    pred_rgb = query_rgb.copy()
    pred_mask = (pred_prob > 0.5).astype(bool)
    pred_rgb[pred_mask, 0] = 255
    pred_rgb[pred_mask, 1] = pred_rgb[pred_mask, 1] // 2
    pred_rgb[pred_mask, 2] = pred_rgb[pred_mask, 2] // 2
    
    # Ground truth overlay (green)
    gt_rgb = query_rgb.copy()
    gt_mask = (query_gt > 0.5).astype(bool)
    gt_rgb[gt_mask, 1] = 255
    gt_rgb[gt_mask, 0] = gt_rgb[gt_mask, 0] // 2
    gt_rgb[gt_mask, 2] = gt_rgb[gt_mask, 2] // 2
    
    # Comparison (yellow=overlap, green=GT only, red=pred only)
    comp_rgb = query_rgb.copy()
    overlap = pred_mask & gt_mask
    gt_only = gt_mask & ~pred_mask
    pred_only = pred_mask & ~gt_mask
    
    comp_rgb[overlap] = [255, 255, 0]
    comp_rgb[gt_only] = [0, 255, 0]
    comp_rgb[pred_only] = [255, 0, 0]
    
    # Create layout: 2 rows
    # Row 1: Support images (3 max)
    # Row 2: Query | Prediction | GT | Comparison
    margin = 30
    title_space = 40
    
    row1_width = num_support * w + (num_support + 1) * margin
    row2_width = 4 * w + 5 * margin
    total_width = max(row1_width, row2_width)
    total_height = 2 * h + 3 * margin + 2 * title_space
    
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Place support images in row 1
    y_offset = margin + title_space
    x_offset = (total_width - row1_width) // 2 + margin
    for i, supp_panel in enumerate(support_panels):
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = supp_panel
        x_offset += w + margin
    
    # Place query/prediction panels in row 2
    y_offset = 2 * margin + title_space + h
    x_offset = (total_width - row2_width) // 2 + margin
    
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = query_rgb
    x_offset += w + margin
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = pred_rgb
    x_offset += w + margin
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = gt_rgb
    x_offset += w + margin
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = comp_rgb
    
    # Convert to PIL and add labels
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        title_font = ImageFont.truetype("arialbd.ttf", 18)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Title
    title = f"IRIS In-Context Learning - Case {case_id}"
    draw.text((total_width // 2 - 150, 10), title, fill=(0, 0, 0), font=title_font)
    
    # Support labels
    x_offset = (total_width - row1_width) // 2 + margin
    for i in range(num_support):
        label = f"Support {i+1}"
        draw.text((x_offset + w//2 - 30, margin + title_space - 25), label, fill=(0, 128, 128), font=font)
        x_offset += w + margin
    
    # Query/prediction labels
    y_pos = 2 * margin + title_space + h - 25
    x_offset = (total_width - row2_width) // 2 + margin
    
    draw.text((x_offset + w//2 - 20, y_pos), "Query", fill=(0, 0, 0), font=font)
    x_offset += w + margin
    draw.text((x_offset + w//2 - 35, y_pos), "Prediction", fill=(255, 0, 0), font=font)
    x_offset += w + margin
    draw.text((x_offset + w//2 - 40, y_pos), "Ground Truth", fill=(0, 128, 0), font=font)
    x_offset += w + margin
    draw.text((x_offset + w//2 - 40, y_pos), "Comparison", fill=(128, 0, 128), font=font)
    
    # Add legend at bottom
    legend_y = total_height - 20
    draw.text((20, legend_y), "Legend:", fill=(0, 0, 0), font=font)
    draw.text((100, legend_y), "Support (Cyan)", fill=(0, 128, 128), font=font)
    draw.text((250, legend_y), "Prediction (Red)", fill=(255, 0, 0), font=font)
    draw.text((420, legend_y), "GT (Green)", fill=(0, 128, 0), font=font)
    draw.text((570, legend_y), "Overlap (Yellow)", fill=(200, 200, 0), font=font)
    
    return pil_img


def visualize_iris_context(
    dataset_name: str,
    checkpoint_path: Path,
    dataset_root: Path = None,
    output_dir: Path = None,
    num_cases: int = 10,
):
    """Generate IRIS in-context visualizations."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print(f"IRIS In-Context Learning Visualization: {dataset_name}")
    print("="*60 + "\n")
    
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
    elif dataset_name == "kvasir":
        volume_shape = (16, 256, 256)
        dataset_config = {
            "depth_slices": 16,
            "target_resolution": 256
        }
    elif dataset_name == "drive":
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
    
    # Load model
    model = IrisModel(
        in_channels=1,
        base_channels=16,
        num_query_tokens=8,
        volume_shape=volume_shape,
        use_memory_bank=True
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✓ Loaded model from {checkpoint_path.name}\n")
    
    # Load dataset
    try:
        if dataset_name == "chest_xray_masks":
            test_dataset = build_dataset(
                dataset_name,
                root=str(dataset_root or Path("datasets/chest_xray_masks/Lung Segmentation")),
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
        elif dataset_name == "kvasir":
            test_dataset = build_dataset(
                dataset_name,
                root=dataset_root or str(Path("datasets/Kvasir-SEG Data (Polyp segmentation & detection)")),
                split=DatasetSplit.TEST,
                **dataset_config
            )
        elif dataset_name == "drive":
            test_dataset = build_dataset(
                dataset_name,
                root=dataset_root or str(Path("datasets/drive-digital-retinal-images-for-vessel-extraction")),
                split=DatasetSplit.TEST,
                **dataset_config
            )
        elif dataset_name == "covid_ct":
            test_dataset = build_dataset(
                dataset_name,
                root=dataset_root or str(Path("datasets/COVID-19 CT scans")),
                split=DatasetSplit.TEST,
                **dataset_config
            )
        else:
            test_dataset = build_dataset(
                dataset_name,
                root=str(dataset_root),
                split=DatasetSplit.TEST,
                **dataset_config
            )
        
        print(f"✓ Loaded {len(test_dataset)} test samples\n")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(f"visualization_outputs/{dataset_name}_iris_context")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    results = []
    num_to_visualize = min(num_cases, len(test_dataset))
    
    with torch.no_grad():
        for case_idx in range(num_to_visualize):
            print(f"Processing case {case_idx + 1}/{num_to_visualize}...")
            
            # Get query sample
            query_sample = test_dataset[case_idx]
            query_img = query_sample["image"].unsqueeze(0).to(device)
            query_mask = query_sample["mask"].unsqueeze(0).to(device)
            
            # Binarize mask
            query_mask = (query_mask > 0).float()
            
            # Get support samples (use different samples as support)
            num_support = 3
            support_indices = [(case_idx + i + 1) % len(test_dataset) for i in range(num_support)]
            
            support_images = []
            support_masks = []
            
            for supp_idx in support_indices:
                supp_sample = test_dataset[supp_idx]
                supp_img = supp_sample["image"].unsqueeze(0).to(device)
                supp_mask = supp_sample["mask"].unsqueeze(0).to(device)
                supp_mask = (supp_mask > 0).float()
                
                support_images.append(supp_img)
                support_masks.append(supp_mask)
            
            # Concatenate support samples
            support_img_batch = torch.cat(support_images, dim=0)
            support_mask_batch = torch.cat(support_masks, dim=0)
            
            # Ensure correct shape
            if support_mask_batch.dim() == 4:
                support_mask_batch = support_mask_batch.unsqueeze(1)
            
            # Encode support (get task embeddings from memory bank)
            # Process support images individually and average embeddings
            all_task_embeddings = []
            for supp_img, supp_mask in zip(support_images, support_masks):
                if supp_mask.dim() == 4:
                    supp_mask = supp_mask.unsqueeze(1)
                support_out = model.encode_support(supp_img, supp_mask)
                all_task_embeddings.append(support_out["task_embeddings"])
            
            # Average task embeddings from all support images
            task_embeddings = torch.stack(all_task_embeddings).mean(dim=0)
            
            # Run query through model with task embeddings
            outputs = model(query_img, task_embeddings)
            logits = outputs["logits"]
            
            # Calculate Dice score
            pred = torch.sigmoid(logits) > 0.5
            if query_mask.dim() == 4:
                query_mask_expand = query_mask.unsqueeze(1)
            else:
                query_mask_expand = query_mask
                
            intersection = (pred * query_mask_expand).sum()
            union = pred.sum() + query_mask_expand.sum()
            dice = (2.0 * intersection / (union + 1e-6)).item() if union > 0 else 0.0
            dice = max(0.0, min(1.0, dice))
            
            # Create visualization
            vis_img = create_iris_visualization(
                support_images,
                support_masks,
                query_img,
                query_mask,
                logits,
                case_idx + 1
            )
            
            # Save
            output_path = output_dir / f"case_{case_idx+1:03d}_iris_context.png"
            vis_img.save(output_path, dpi=(300, 300))
            print(f"  ✓ Saved: {output_path.name} (Dice: {dice:.4f})")
            
            results.append({
                "case_id": case_idx + 1,
                "dice_score": dice,
                "num_support_images": num_support
            })
    
    # Save summary
    summary = {
        "dataset": dataset_name,
        "num_cases": num_to_visualize,
        "average_dice": np.mean([r["dice_score"] for r in results]),
        "results": results
    }
    
    summary_path = output_dir / "iris_context_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved: {summary_path}")
    print(f"✓ All visualizations saved to: {output_dir}")
    print(f"\nAverage Dice Score: {summary['average_dice']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize IRIS in-context learning")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset root directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--num-cases", type=int, default=10, help="Number of cases to visualize")
    
    args = parser.parse_args()
    
    visualize_iris_context(
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        num_cases=args.num_cases
    )


if __name__ == "__main__":
    main()
