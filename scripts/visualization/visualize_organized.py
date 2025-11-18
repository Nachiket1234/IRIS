"""
Organized visualization of IRIS inference with separate folders for each case and strategy.
Each case folder contains: input image + 4 strategy folders (each with support + output images).
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
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


class HighQualityMedicalDataset(torch.utils.data.Dataset):
    """High-quality synthetic 3D medical volume dataset."""

    def __init__(
        self,
        length: int,
        classes: int = 3,
        volume_shape=(96, 96, 96),
        modality: str = "CT",
        seed_offset: int = 0,
    ) -> None:
        self.length = length
        self.classes = classes
        self.volume_shape = volume_shape
        self.modality = modality
        self.seed_offset = seed_offset
        self.dataset_name = "high_quality_medical"
        self.split = type("Split", (), {"value": "eval"})()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        depth, height, width = self.volume_shape
        g = torch.Generator().manual_seed((idx + self.seed_offset) * 1000 + 42)

        if self.modality == "CT":
            base_intensity = torch.randn(1, depth, height, width, generator=g) * 150 + 0
            base_intensity = torch.clamp(base_intensity, -1000, 1000)
        else:
            base_intensity = torch.rand(1, depth, height, width, generator=g)

        mask = torch.zeros(depth, height, width, dtype=torch.int64)

        for cls in range(1, self.classes + 1):
            center_z = int(torch.randint(10, depth - 10, (1,), generator=g).item())
            center_y = int(torch.randint(10, height - 10, (1,), generator=g).item())
            center_x = int(torch.randint(10, width - 10, (1,), generator=g).item())

            radius_z = int(torch.randint(12, 20, (1,), generator=g).item())
            radius_y = int(torch.randint(12, 20, (1,), generator=g).item())
            radius_x = int(torch.randint(12, 20, (1,), generator=g).item())

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

            organ_mask = dist <= 1.2
            mask[organ_mask] = cls

            if self.modality == "CT":
                organ_intensities = {1: 50, 2: 150, 3: 250}
                organ_intensity = organ_intensities.get(cls, torch.randn(1, generator=g).item() * 100 + 100)
            else:
                organ_intensity = (cls / self.classes) * 0.4 + 0.3

            base_intensity[0, organ_mask] = organ_intensity

        noise = torch.randn(1, depth, height, width, generator=g) * 10
        base_intensity = base_intensity + noise

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


def save_image_with_overlay(
    image: np.ndarray,
    overlay: Optional[np.ndarray],
    output_path: Path,
    title: str = "",
    cmap: str = "gray",
    overlay_cmap: str = "jet",
    alpha: float = 0.5,
    dpi: int = 300,
):
    """Save a single image with optional overlay."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
    
    if overlay is not None:
        overlay_mask = overlay > 0.5
        ax.imshow(
            np.ma.masked_where(~overlay_mask, overlay),
            cmap=overlay_cmap,
            alpha=alpha,
            vmin=0,
            vmax=1,
        )
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_three_views(
    image: torch.Tensor,
    overlay: Optional[torch.Tensor],
    output_path: Path,
    title: str = "",
    cmap: str = "gray",
    overlay_cmap: str = "jet",
    alpha: float = 0.5,
    dpi: int = 300,
):
    """Save three orthogonal views (axial, coronal, sagittal)."""
    if not HAS_MATPLOTLIB:
        return
    
    slices = extract_middle_slices(image)
    overlay_slices = extract_middle_slices(overlay) if overlay is not None else None
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    views = ["axial", "coronal", "sagittal"]
    view_titles = ["Axial View", "Coronal View", "Sagittal View"]
    
    for idx, (view, view_title) in enumerate(zip(views, view_titles)):
        ax = axes[idx]
        ax.imshow(slices[view], cmap=cmap, vmin=0, vmax=1)
        
        if overlay_slices is not None:
            overlay_mask = overlay_slices[view] > 0.5
            ax.imshow(
                np.ma.masked_where(~overlay_mask, overlay_slices[view]),
                cmap=overlay_cmap,
                alpha=alpha,
                vmin=0,
                vmax=1,
            )
        
        ax.set_title(view_title, fontsize=12, fontweight="bold")
        ax.axis("off")
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
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


def main():
    base_output_dir = Path("visualization_outputs_organized")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("IRIS Organized Visualization with High-Quality Images")
    print("=" * 80)
    print()

    set_global_seed(42)

    # Load trained model
    checkpoint_path = Path("demo_outputs/high_quality_training/checkpoints/iris_iter_000050.pt")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please run train_better_quality.py first to train the model.")
        return

    print(f"Loading model from: {checkpoint_path}")
    model = IrisModel(
        in_channels=1,
        base_channels=16,
        num_query_tokens=4,
        num_attention_heads=4,
        volume_shape=(96, 96, 96),
        use_memory_bank=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("Model loaded successfully!")
    print()

    # Create datasets
    eval_dataset = HighQualityMedicalDataset(
        length=10,
        classes=3,
        volume_shape=(96, 96, 96),
        modality="CT",
        seed_offset=200,
    )

    support_dataset = HighQualityMedicalDataset(
        length=10,
        classes=3,
        volume_shape=(96, 96, 96),
        modality="CT",
        seed_offset=0,
    )

    print(f"Evaluation dataset: {len(eval_dataset)} volumes")
    print(f"Support dataset: {len(support_dataset)} volumes")
    print()

    eval_config = EvaluationConfig(
        in_distribution=[eval_dataset],
        num_episodes=1,
        ensemble_size=3,
        strategies=("one_shot", "context_ensemble", "object_retrieval", "in_context_tuning"),
        tuner_steps=10,
        tuner_lr=5e-4,
        random_seed=42,
    )

    evaluator = MedicalEvaluationSuite(model, eval_config)

    # Process 4 cases
    num_cases = 4
    query_indices = list(range(num_cases))
    
    all_results = []

    for case_idx, query_idx in enumerate(query_indices):
        print(f"Processing Case {case_idx + 1}/{num_cases} (Query Index: {query_idx})...")

        # Create case directory
        case_dir = base_output_dir / f"case_{case_idx + 1:02d}"
        case_dir.mkdir(exist_ok=True)

        # Get query sample
        query_sample = eval_dataset[query_idx]
        query_image = query_sample["image"].float().unsqueeze(0)
        query_mask = query_sample["mask"]

        # Get class IDs
        class_ids = [
            int(c.item()) for c in torch.unique(query_mask) if int(c.item()) != 0
        ]
        if len(class_ids) == 0:
            print(f"  Skipping case {case_idx + 1} - no foreground classes")
            continue

        # Save input image (three views)
        save_three_views(
            query_image[0],
            query_mask.float(),
            case_dir / "01_input_image_with_ground_truth.png",
            title=f"Case {case_idx + 1}: Input Image with Ground Truth",
        )

        # Prepare binary masks
        query_binary = prepare_binary_masks(query_mask, class_ids).unsqueeze(0)

        # Get support sample
        support_idx = case_idx % len(support_dataset)
        support_sample = support_dataset[support_idx]
        support_image = support_sample["image"].float().unsqueeze(0)
        support_mask = support_sample["mask"]
        support_binary = prepare_binary_masks(support_mask, class_ids).unsqueeze(0)

        # Memory bank info
        memory_bank_info = {
            "classes": class_ids,
            "retrieved": False,
        }

        # Process each strategy
        strategy_results = {}
        strategy_names = ["one_shot", "context_ensemble", "object_retrieval", "in_context_tuning"]
        strategy_display_names = {
            "one_shot": "One-Shot Inference",
            "context_ensemble": "Context Ensemble",
            "object_retrieval": "Object Retrieval",
            "in_context_tuning": "In-Context Tuning",
        }

        for strategy_name in strategy_names:
            print(f"  Running {strategy_display_names[strategy_name]}...")
            
            # Create strategy directory
            strategy_dir = case_dir / f"strategy_{strategy_name}"
            strategy_dir.mkdir(exist_ok=True)

            strategy_fn = evaluator.strategies[strategy_name]

            # Run inference
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

            # Get prediction (max across classes)
            pred_probs = torch.sigmoid(logits[0])
            max_pred = pred_probs.max(dim=0)[0]

            # Save support/reference image
            save_three_views(
                support_image[0],
                support_mask.float(),
                strategy_dir / "support_reference_image.png",
                title=f"{strategy_display_names[strategy_name]}: Support/Reference Image",
            )

            # Save output prediction
            save_three_views(
                query_image[0],
                max_pred,
                strategy_dir / "output_prediction.png",
                title=f"{strategy_display_names[strategy_name]}: Output Prediction",
                overlay_cmap="hot",
            )

            # Save output prediction overlay (axial view only, larger)
            query_slices = extract_middle_slices(query_image[0])
            pred_slices = extract_middle_slices(max_pred.unsqueeze(0))
            
            save_image_with_overlay(
                query_slices["axial"],
                pred_slices["axial"],
                strategy_dir / "output_prediction_axial_large.png",
                title=f"{strategy_display_names[strategy_name]}: Axial View with Prediction",
                overlay_cmap="hot",
                dpi=300,
            )

            # Compute metrics
            dice = compute_dice(logits, query_mask, class_ids)
            strategy_results[strategy_name] = {
                "dice_per_class": {cls: float(dice[i]) for i, cls in enumerate(class_ids)},
                "dice_mean": float(dice.mean()),
            }

            # Save strategy info
            info_path = strategy_dir / "strategy_info.txt"
            with info_path.open("w") as fp:
                fp.write(f"Strategy: {strategy_display_names[strategy_name]}\n")
                fp.write(f"=" * 50 + "\n\n")
                fp.write(f"Dice Scores:\n")
                for cls_id in class_ids:
                    fp.write(f"  Class {cls_id}: {strategy_results[strategy_name]['dice_per_class'][cls_id]:.4f}\n")
                fp.write(f"Mean Dice: {strategy_results[strategy_name]['dice_mean']:.4f}\n")
                fp.write(f"\nMemory Bank: {'Retrieved' if memory_bank_info['retrieved'] else 'One-shot encoding'}\n")

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
    summary_path = base_output_dir / "inference_summary.json"
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
            print(f"  {strategy_display_names[strategy]:25s} - Mean Dice: {metrics['dice_mean']:.4f}")

    print()
    print("=" * 80)
    print(f"All visualizations saved to: {base_output_dir}")
    print("=" * 80)
    print("\nFolder structure:")
    print("  case_01/")
    print("    ├── 01_input_image_with_ground_truth.png")
    print("    ├── strategy_one_shot/")
    print("    │   ├── support_reference_image.png")
    print("    │   ├── output_prediction.png")
    print("    │   ├── output_prediction_axial_large.png")
    print("    │   └── strategy_info.txt")
    print("    ├── strategy_context_ensemble/")
    print("    ├── strategy_object_retrieval/")
    print("    └── strategy_in_context_tuning/")
    print("  ... (case_02, case_03, case_04)")


if __name__ == "__main__":
    main()

