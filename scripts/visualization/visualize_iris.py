"""
Main IRIS visualization script - generates organized output folders for inference results.
This is the consolidated visualization script.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from iris.model import IrisModel
from iris.training import set_global_seed


def find_latest_checkpoint():
    """Find the latest checkpoint from training outputs."""
    checkpoint_dirs = [
        Path("outputs/training/checkpoints"),
        Path("demo_outputs/real_medical_gpu_training/checkpoints"),
        Path("demo_outputs/improved_medical_training/checkpoints"),
    ]
    
    all_checkpoints = []
    for ckpt_dir in checkpoint_dirs:
        if ckpt_dir.exists():
            all_checkpoints.extend(ckpt_dir.glob("*.pt"))
    
    if not all_checkpoints:
        return None
    
    return sorted(all_checkpoints, key=lambda p: p.stat().st_mtime)[-1]


def create_visualization(image, mask, prediction, output_path, title=""):
    """Create a high-quality visualization of image, ground truth, and prediction."""
    # Select middle slice
    depth = image.shape[-3]
    slice_idx = depth // 2
    
    img_slice = image[..., slice_idx, :, :].squeeze().cpu().numpy()
    mask_slice = mask[..., slice_idx, :, :].squeeze().cpu().numpy()
    pred_slice = prediction[..., slice_idx, :, :].squeeze().cpu().numpy()
    
    # Improved contrast normalization using percentile-based scaling
    img_min = np.percentile(img_slice, 2)
    img_max = np.percentile(img_slice, 98)
    if img_max > img_min:
        img_slice = np.clip((img_slice - img_min) / (img_max - img_min), 0, 1)
    else:
        img_slice = np.clip(img_slice, 0, 1)
    
    # Upscale for much better resolution (8x for ultra-high quality)
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


def main():
    output_dir = Path("outputs/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("IRIS Visualization - Organized Output")
    print("=" * 80)
    print()
    
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path is None:
        print("Error: No checkpoint found")
        print("Please run scripts/training/train_iris.py first")
        return
    
    print(f"Loading model from: {checkpoint_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        # Try to load directly, filtering out non-model keys
        model_dict = {k: v for k, v in checkpoint.items() if k.startswith(("encoder.", "task_encoder.", "mask_decoder."))}
        if model_dict:
            model.load_state_dict(model_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print()
    
    # Create synthetic test data
    print("Generating test cases...")
    test_cases = []
    for case_idx in range(4):
        # Generate synthetic volume
        volume_shape = (64, 64, 64)
        image = torch.rand(volume_shape) * 0.5 + 0.3
        
        # Create mask with 2-3 classes
        mask = torch.zeros(volume_shape, dtype=torch.int64)
        num_classes = torch.randint(2, 4, (1,)).item()
        
        for cls in range(1, num_classes + 1):
            center = torch.randint(15, 50, (3,))
            radius = torch.randint(8, 15, (1,)).item()
            coords = torch.meshgrid(
                torch.arange(64), torch.arange(64), torch.arange(64), indexing="ij"
            )
            dist = sum((c - center[i]) ** 2 for i, c in enumerate(coords))
            mask[dist < radius ** 2] = cls
        
        image = image.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, D, H, W)
        mask = mask.unsqueeze(0).to(device)  # (1, D, H, W)
        
        test_cases.append({
            "image": image,
            "mask": mask,
            "case_id": f"case_{case_idx + 1:02d}",
        })
    
    print(f"Generated {len(test_cases)} test cases")
    print()
    
    # Run inference for each case
    strategies = ["one_shot", "context_ensemble", "memory_retrieval", "in_context_tuning"]
    
    for case_idx, case in enumerate(test_cases):
        case_dir = output_dir / case["case_id"]
        case_dir.mkdir(exist_ok=True)
        
        print(f"Processing {case['case_id']}...")
        
        # Save input image with improved quality
        input_path = case_dir / "01_input.png"
        img_slice = case["image"][0, 0, 32, :, :].cpu().numpy()
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
        
        # Prepare support and query
        support_image = case["image"]
        support_mask = case["mask"]
        query_image = case["image"]  # Use same image as query for demo
        
        # Extract class IDs
        unique_classes = torch.unique(support_mask)
        class_ids = [int(c.item()) for c in unique_classes if int(c.item()) != 0]
        
        if not class_ids:
            print(f"  [SKIP] No foreground classes in {case['case_id']}")
            continue
        
        # Create binary masks for support
        # support_mask is (1, D, H, W), need (1, K, D, H, W)
        support_binary_list = []
        for cls in class_ids:
            cls_mask = (support_mask == cls).float()  # (1, D, H, W)
            support_binary_list.append(cls_mask.squeeze(0))  # (D, H, W)
        support_binary = torch.stack(support_binary_list, dim=0).unsqueeze(0).to(device)  # (1, K, D, H, W)
        
        # Run each inference strategy
        for strategy in strategies:
            strategy_dir = case_dir / f"02_{strategy}"
            strategy_dir.mkdir(exist_ok=True)
            
            with torch.no_grad():
                if strategy == "one_shot":
                    # One-shot inference
                    support_dict = model.encode_support(support_image, support_binary)
                    task_embeddings = support_dict["task_embeddings"]
                    outputs = model(query_image, task_embeddings)
                    logits = outputs["logits"]
                    
                elif strategy == "context_ensemble":
                    # Context ensemble (use multiple support samples)
                    support_dict = model.encode_support(support_image, support_binary)
                    task_embeddings = support_dict["task_embeddings"]
                    # Average with itself for demo (in real case, use multiple supports)
                    outputs = model(query_image, task_embeddings)
                    logits = outputs["logits"]
                    
                elif strategy == "memory_retrieval":
                    # Memory retrieval
                    if model.memory_bank is not None:
                        try:
                            memory_embeddings = model.retrieve_memory_embeddings(class_ids)
                            memory_embeddings = memory_embeddings.unsqueeze(0)
                            outputs = model(query_image, memory_embeddings)
                            logits = outputs["logits"]
                        except KeyError:
                            # Fallback to one-shot
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
                    # In-context tuning - simplified version (use one-shot for now)
                    # Full implementation requires InContextTuner from iris.model.tuning
                    support_dict = model.encode_support(support_image, support_binary)
                    task_embeddings = support_dict["task_embeddings"]
                    outputs = model(query_image, task_embeddings)
                    logits = outputs["logits"]
                    # Note: Full in-context tuning would use InContextTuner class
            
            # Convert logits to prediction
            pred = torch.sigmoid(logits) > 0.5  # (1, K, D, H, W)
            pred_mask = torch.zeros_like(case["mask"])  # (1, D, H, W)
            for i, cls in enumerate(class_ids):
                pred_mask[0][pred[0, i] > 0.5] = cls
            
            # Compute loss
            from iris.model.tuning import DiceCrossEntropyLoss
            from iris.training.utils import compute_class_weights
            loss_fn = DiceCrossEntropyLoss()
            query_binary = torch.stack(
                [(case["mask"] == cls).float() for cls in class_ids], dim=0
            ).unsqueeze(0).to(device)  # (1, K, D, H, W)
            class_weights = compute_class_weights(query_binary)
            loss_value = loss_fn(logits, query_binary, class_weights=class_weights).item()
            
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
            gt_mask = case["mask"].unsqueeze(0)  # (1, D, H, W)
            create_visualization(
                query_image, gt_mask.unsqueeze(0),
                pred_mask.unsqueeze(0),
                output_path, f"Output - {strategy}"
            )
            
            print(f"  [OK] {strategy} (Loss: {loss_value:.4f})")
        
        print(f"  [OK] {case['case_id']} completed")
        print()
    
    print("=" * 80)
    print(f"Visualization complete! Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

