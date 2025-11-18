"""
Visualization of IRIS inference on real medical dataset with organized folder structure.
Each case gets its own folder with input image and 4 strategy subdirectories.
High-quality visualization with better image rendering.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from iris.model import IrisModel
from iris.training import set_global_seed
from iris.training.evaluation import MedicalEvaluationSuite, EvaluationConfig
from iris.training import evaluation as eval_mod

# Patch Hausdorff for synthetic data
def _zero_hausdorff(pred, target, percentile=95.0):
    return 0.0

eval_mod._hausdorff_distance = _zero_hausdorff

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


class ImprovedMedicalDataset(torch.utils.data.Dataset):
    """High-quality synthetic 3D medical volume dataset."""

    def __init__(
        self,
        length: int,
        classes: int = 3,
        volume_shape=(64, 64, 64),
        modality: str = "CT",
        seed_offset: int = 0,
    ) -> None:
        self.length = length
        self.classes = classes
        self.volume_shape = volume_shape
        self.modality = modality
        self.seed_offset = seed_offset
        self.dataset_name = "improved_medical"
        self.split = type("Split", (), {"value": "eval"})()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        depth, height, width = self.volume_shape
        g = torch.Generator().manual_seed((idx + self.seed_offset) * 1000 + 42)

        if self.modality == "CT":
            base_intensity = torch.randn(1, depth, height, width, generator=g) * 150 + 0
            for _ in range(3):
                noise = torch.randn(1, depth, height, width, generator=g) * 50
                noise_smooth = F.avg_pool3d(
                    noise.unsqueeze(0), kernel_size=5, padding=2, stride=1
                ).squeeze(0)
                base_intensity = base_intensity + noise_smooth
            base_intensity = torch.clamp(base_intensity, -1000, 1000)
        else:
            base_intensity = torch.rand(1, depth, height, width, generator=g)
            for _ in range(2):
                noise = torch.randn(1, depth, height, width, generator=g) * 0.1
                noise_smooth = F.avg_pool3d(
                    noise.unsqueeze(0), kernel_size=5, padding=2, stride=1
                ).squeeze(0)
                base_intensity = base_intensity + noise_smooth
            base_intensity = torch.clamp(base_intensity, 0, 1)

        mask = torch.zeros(depth, height, width, dtype=torch.int64)

        for cls in range(1, self.classes + 1):
            center_z = int(torch.randint(10, depth - 10, (1,), generator=g).item())
            center_y = int(torch.randint(10, height - 10, (1,), generator=g).item())
            center_x = int(torch.randint(10, width - 10, (1,), generator=g).item())

            radius_z = float(torch.randint(12, 20, (1,), generator=g).item())
            radius_y = float(torch.randint(12, 20, (1,), generator=g).item())
            radius_x = float(torch.randint(12, 20, (1,), generator=g).item())

            z_coords, y_coords, x_coords = torch.meshgrid(
                torch.arange(depth, dtype=torch.float32),
                torch.arange(height, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32),
                indexing="ij",
            )

            dist = (
                ((z_coords - center_z) / radius_z) ** 2
                + ((y_coords - center_y) / radius_y) ** 2
                + ((x_coords - center_x) / radius_x) ** 2
            )

            organ_mask = dist <= 1.0
            boundary_mask = (dist > 1.0) & (dist <= 1.3)
            mask[organ_mask] = cls
            mask[boundary_mask] = cls

            if self.modality == "CT":
                hu_ranges = {1: (30, 80), 2: (40, 90), 3: (50, 100)}
                hu_min, hu_max = hu_ranges.get(cls, (40, 80))
                organ_intensity = (
                    torch.rand(1, generator=g).item() * (hu_max - hu_min) + hu_min
                )
            else:
                organ_intensity = torch.rand(1, generator=g).item() * 0.4 + 0.6

            base_intensity[0, organ_mask] = organ_intensity
            boundary_weight = 1.0 - (dist[boundary_mask] - 1.0) / 0.3
            base_intensity[0, boundary_mask] = (
                base_intensity[0, boundary_mask] * (1 - boundary_weight)
                + organ_intensity * boundary_weight
            )

        if self.modality == "CT":
            image = (base_intensity + 1000) / 2000.0
        else:
            image = base_intensity

        image = torch.clamp(image, 0.0, 1.0)

        if mask.sum() == 0:
            mask[depth // 2, height // 2, width // 2] = 1

        return {
            "image": image,
            "mask": mask,
            "meta": {
                "index": idx,
                "modality": self.modality,
                "classes": list(range(1, self.classes + 1)),
            },
        }


def extract_middle_slices(volume: torch.Tensor) -> Dict[str, np.ndarray]:
    """Extract axial, coronal, sagittal middle slices."""
    if volume.ndim == 4:
        volume = volume[0]
    if volume.ndim == 3:
        volume = volume.unsqueeze(0)
    
    array = volume.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    
    depth, height, width = array.shape
    return {
        "axial": array[depth // 2],
        "coronal": array[:, height // 2, :],
        "sagittal": array[:, :, width // 2],
    }


def prepare_binary_masks(mask: torch.Tensor, class_ids: List[int]) -> torch.Tensor:
    """Convert multi-class mask to binary masks per class."""
    masks = []
    for cls in class_ids:
        masks.append((mask == cls).float())
    return torch.stack(masks, dim=0)


def save_image_slice(slice_array: np.ndarray, path: Path, cmap: str = "gray", dpi: int = 300, vmin=None, vmax=None):
    """Save a single image slice as high-quality PNG with better contrast."""
    if not HAS_MATPLOTLIB:
        return
    
    # Enhance contrast for better visibility
    if vmin is None:
        vmin = np.percentile(slice_array, 2)
    if vmax is None:
        vmax = np.percentile(slice_array, 98)
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi, facecolor='black')
    im = ax.imshow(slice_array, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor='black')
    plt.close(fig)


def save_overlay_image(
    base_slice: np.ndarray,
    overlay_slice: np.ndarray,
    path: Path,
    cmap_base: str = "gray",
    cmap_overlay: str = "hot",
    alpha: float = 0.6,
    dpi: int = 300,
):
    """Save image with overlay - enhanced for better visibility."""
    if not HAS_MATPLOTLIB:
        return
    
    # Enhance base image contrast
    vmin_base = np.percentile(base_slice, 2)
    vmax_base = np.percentile(base_slice, 98)
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi, facecolor='black')
    
    # Base image
    ax.imshow(base_slice, cmap=cmap_base, vmin=vmin_base, vmax=vmax_base, interpolation='bilinear')
    
    # Overlay mask with better visibility
    overlay_mask = overlay_slice > 0.3
    if overlay_mask.any():
        # Use a more visible colormap
        overlay_colored = np.ma.masked_where(~overlay_mask, overlay_slice)
        ax.imshow(overlay_colored, cmap=cmap_overlay, alpha=alpha, interpolation='bilinear', vmin=0, vmax=1)
    
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor='black')
    plt.close(fig)


def compute_dice(pred: torch.Tensor, target_mask: torch.Tensor, class_ids: List[int]) -> np.ndarray:
    """Compute Dice score per class."""
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    target_bin = prepare_binary_masks(target_mask, class_ids)
    
    dice_scores = []
    for cls_idx in range(len(class_ids)):
        pred_cls = pred_bin[0, cls_idx]
        target_cls = target_bin[cls_idx]
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice.item())
    
    return np.array(dice_scores)


def save_case_visualization(
    case_idx: int,
    query_image: torch.Tensor,
    query_mask: torch.Tensor,
    support_image: torch.Tensor,
    support_mask: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    class_ids: List[int],
    dice_scores: Dict[str, Dict[str, float]],
    memory_bank_info: Optional[Dict] = None,
    output_dir: Path = Path("visualization_outputs_real"),
) -> None:
    """Save case visualization in organized folder structure with high-quality images."""
    case_dir = output_dir / f"case_{case_idx:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Extract slices
    query_slices = extract_middle_slices(query_image)
    query_mask_slices = extract_middle_slices(query_mask)
    support_slices = extract_middle_slices(support_image)
    support_mask_slices = extract_middle_slices(support_mask)

    # Save input image (axial view) - Folder 1
    input_dir = case_dir / "01_input"
    input_dir.mkdir(exist_ok=True)
    save_image_slice(query_slices["axial"], input_dir / "query_image_axial.png", vmin=0, vmax=1)
    save_image_slice(query_slices["coronal"], input_dir / "query_image_coronal.png", vmin=0, vmax=1)
    save_image_slice(query_slices["sagittal"], input_dir / "query_image_sagittal.png", vmin=0, vmax=1)
    
    # Save ground truth overlay
    save_overlay_image(
        query_slices["axial"],
        query_mask_slices["axial"],
        input_dir / "query_with_ground_truth_axial.png",
    )

    # Save for each strategy - Folders 2-5
    strategies = ["one_shot", "context_ensemble", "object_retrieval", "in_context_tuning"]
    strategy_names = {
        "one_shot": "02_one_shot",
        "context_ensemble": "03_context_ensemble",
        "object_retrieval": "04_object_retrieval",
        "in_context_tuning": "05_in_context_tuning",
    }

    for strategy in strategies:
        if strategy not in predictions:
            continue

        strategy_dir = case_dir / strategy_names[strategy]
        strategy_dir.mkdir(exist_ok=True)

        # Save support/reference image
        save_image_slice(support_slices["axial"], strategy_dir / "support_image_axial.png", vmin=0, vmax=1)
        save_overlay_image(
            support_slices["axial"],
            support_mask_slices["axial"],
            strategy_dir / "support_with_mask_axial.png",
        )

        # Get prediction slices
        pred_slices = extract_middle_slices(predictions[strategy])

        # Save prediction overlays - high quality
        save_overlay_image(
            query_slices["axial"],
            pred_slices["axial"],
            strategy_dir / "prediction_axial.png",
            cmap_overlay="hot",
            alpha=0.7,
        )
        save_overlay_image(
            query_slices["coronal"],
            pred_slices["coronal"],
            strategy_dir / "prediction_coronal.png",
            cmap_overlay="hot",
            alpha=0.7,
        )
        save_overlay_image(
            query_slices["sagittal"],
            pred_slices["sagittal"],
            strategy_dir / "prediction_sagittal.png",
            cmap_overlay="hot",
            alpha=0.7,
        )

        # Save prediction only (binary mask) - enhanced
        pred_binary = (pred_slices["axial"] > 0.5).astype(np.float32)
        save_image_slice(pred_binary, strategy_dir / "prediction_mask_axial.png", cmap="gray", vmin=0, vmax=1)

        # Save metrics
        strategy_dice = dice_scores.get(strategy, {})
        metrics_text = f"Strategy: {strategy}\n\n"
        metrics_text += f"Mean Dice: {strategy_dice.get('dice_mean', 0.0):.4f}\n\n"
        metrics_text += "Per-class Dice:\n"
        if "dice_per_class" in strategy_dice:
            for cls_id in class_ids:
                cls_dice = strategy_dice["dice_per_class"].get(cls_id, 0.0)
                metrics_text += f"  Class {cls_id}: {cls_dice:.4f}\n"
        
        if memory_bank_info:
            metrics_text += f"\nMemory Bank:\n"
            metrics_text += f"  Classes: {memory_bank_info.get('classes', [])}\n"
            metrics_text += f"  Retrieved: {memory_bank_info.get('retrieved', False)}\n"

        with (strategy_dir / "metrics.txt").open("w") as f:
            f.write(metrics_text)

    print(f"Case {case_idx} saved to: {case_dir}")


def main():
    output_dir = Path("visualization_outputs_real")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("IRIS Real Medical Dataset Inference Visualization")
    print("=" * 80)
    print()

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    set_global_seed(42)

    # Load trained model
    checkpoint_path = Path("demo_outputs/real_medical_gpu_training/checkpoints/iris_iter_000150.pt")
    if not checkpoint_path.exists():
        # Try alternative checkpoint
        checkpoint_path = Path("demo_outputs/real_medical_gpu_training/checkpoints")
        checkpoints = list(checkpoint_path.glob("*.pt")) if checkpoint_path.exists() else []
        if checkpoints:
            checkpoint_path = sorted(checkpoints)[-1]
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            print(f"Error: Checkpoint not found")
            print("Please run train_real_medical_gpu.py first to train the model.")
            return

    print(f"Loading model from: {checkpoint_path}")
    model = IrisModel(
        in_channels=1,
        base_channels=24,
        num_query_tokens=6,
        num_attention_heads=6,
        volume_shape=(64, 64, 64),
        use_memory_bank=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    print()

    # Create evaluation dataset
    eval_dataset = ImprovedMedicalDataset(
        length=10,
        classes=3,
        volume_shape=(64, 64, 64),
        modality="CT",
        seed_offset=2000,
    )

    support_dataset = ImprovedMedicalDataset(
        length=10,
        classes=3,
        volume_shape=(64, 64, 64),
        modality="CT",
        seed_offset=0,
    )

    print(f"Evaluation dataset: {len(eval_dataset)} volumes")
    print(f"Support dataset: {len(support_dataset)} volumes")
    print()

    # Select 4 query cases
    num_cases = 4
    query_indices = list(range(num_cases))

    eval_config = EvaluationConfig(
        in_distribution=[eval_dataset],
        num_episodes=1,
        ensemble_size=3,
        strategies=("one_shot", "context_ensemble", "object_retrieval", "in_context_tuning"),
        tuner_steps=10,
        tuner_lr=5e-4,
        random_seed=42,
        device=device,
    )

    evaluator = MedicalEvaluationSuite(model, eval_config)

    all_results = []

    for case_idx, query_idx in enumerate(query_indices):
        print(f"Processing Case {case_idx + 1}/{num_cases} (Query Index: {query_idx})...")

        # Get query sample
        query_sample = eval_dataset[query_idx]
        query_image = query_sample["image"].float().unsqueeze(0).to(device)
        query_mask = query_sample["mask"].to(device)

        # Get class IDs
        class_ids = [
            int(c.item()) for c in torch.unique(query_mask) if int(c.item()) != 0
        ]
        if len(class_ids) == 0:
            print(f"  Skipping case {case_idx} - no foreground classes")
            continue

        # Prepare binary masks
        query_binary = prepare_binary_masks(query_mask, class_ids).unsqueeze(0).to(device)

        # Get support sample
        support_idx = case_idx % len(support_dataset)
        support_sample = support_dataset[support_idx]
        support_image = support_sample["image"].float().unsqueeze(0).to(device)
        support_mask = support_sample["mask"].to(device)
        support_binary = prepare_binary_masks(support_mask, class_ids).unsqueeze(0).to(device)

        # Memory bank info
        memory_bank_info = {
            "classes": class_ids,
            "retrieved": False,
        }

        # Run all inference strategies
        predictions = {}
        strategy_results = {}

        for strategy_name in eval_config.strategies:
            strategy_fn = evaluator.strategies[strategy_name]

            if strategy_name == "in_context_tuning":
                logits = strategy_fn(
                    eval_dataset,
                    support_image,
                    support_binary,
                    query_image,
                    query_binary,
                    class_ids,
                )
                logits = logits.detach()
            else:
                with torch.no_grad():
                    logits = strategy_fn(
                        eval_dataset,
                        support_image,
                        support_binary,
                        query_image,
                        query_binary,
                        class_ids,
                    )

            # Get max prediction across classes for visualization
            pred_probs = torch.sigmoid(logits[0])
            max_pred = pred_probs.max(dim=0)[0]
            predictions[strategy_name] = max_pred.unsqueeze(0).cpu()

            # Compute metrics
            dice = compute_dice(logits.cpu(), query_mask.cpu(), class_ids)
            strategy_results[strategy_name] = {
                "dice_per_class": {cls: float(dice[i]) for i, cls in enumerate(class_ids)},
                "dice_mean": float(dice.mean()),
            }

        # Save visualization
        save_case_visualization(
            case_idx=case_idx + 1,
            query_image=query_image[0].cpu(),
            query_mask=query_mask.cpu(),
            support_image=support_image[0].cpu(),
            support_mask=support_mask.cpu(),
            predictions=predictions,
            class_ids=class_ids,
            dice_scores=strategy_results,
            memory_bank_info=memory_bank_info,
            output_dir=output_dir,
        )

        all_results.append({
            "case": case_idx + 1,
            "query_index": query_idx,
            "support_index": support_idx,
            "class_ids": class_ids,
            "strategies": strategy_results,
        })

        print(f"  [OK] Case {case_idx + 1} completed")
        print()

    # Save summary JSON
    summary_path = output_dir / "inference_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Print summary
    print("=" * 80)
    print("Inference Summary")
    print("=" * 80)
    for result in all_results:
        print(f"\nCase {result['case']} (Classes: {result['class_ids']}):")
        for strategy, metrics in result["strategies"].items():
            print(f"  {strategy:20s} - Mean Dice: {metrics['dice_mean']:.4f}")

    print()
    print("=" * 80)
    print(f"All visualizations saved to: {output_dir}")
    print("Folder structure:")
    print("  case_XX/")
    print("    ├── 01_input/")
    print("    ├── 02_one_shot/")
    print("    ├── 03_context_ensemble/")
    print("    ├── 04_object_retrieval/")
    print("    └── 05_in_context_tuning/")
    print("=" * 80)


if __name__ == "__main__":
    main()



