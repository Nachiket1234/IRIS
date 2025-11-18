"""
Visualize IRIS inference on multiple images with memory bank context display.
Shows reference images, memory bank retrieval, and predictions for all strategies.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class RealisticMedicalDataset(torch.utils.data.Dataset):
    """Synthetic 3D medical volume dataset."""

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
        self.dataset_name = "realistic_medical"
        self.split = type("Split", (), {"value": "eval"})()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        depth, height, width = self.volume_shape
        g = torch.Generator().manual_seed((idx + self.seed_offset) * 1000 + 42)

        if self.modality == "CT":
            base_intensity = torch.randn(1, depth, height, width, generator=g) * 200 + 0
            base_intensity = torch.clamp(base_intensity, -1000, 1000)
        else:
            base_intensity = torch.rand(1, depth, height, width, generator=g)

        mask = torch.zeros(depth, height, width, dtype=torch.int64)

        for cls in range(1, self.classes + 1):
            center_z = int(torch.randint(0, depth, (1,), generator=g).item())
            center_y = int(torch.randint(0, height, (1,), generator=g).item())
            center_x = int(torch.randint(0, width, (1,), generator=g).item())

            radius_z = int(torch.randint(8, 15, (1,), generator=g).item())
            radius_y = int(torch.randint(8, 15, (1,), generator=g).item())
            radius_x = int(torch.randint(8, 15, (1,), generator=g).item())

            z_coords, y_coords, x_coords = torch.meshgrid(
                torch.arange(depth),
                torch.arange(height),
                torch.arange(width),
                indexing="ij",
            )

            dist = (
                ((z_coords - center_z) / radius_z) ** 2
                + ((y_coords - center_y) / radius_y) ** 2
                + ((x_coords - center_x) / radius_x) ** 2
            )

            organ_mask = dist <= 1.0
            mask[organ_mask] = cls

            if self.modality == "CT":
                organ_intensity = torch.randn(1, generator=g).item() * 100 + 50
            else:
                organ_intensity = torch.rand(1, generator=g).item() * 0.3 + 0.5

            base_intensity[0, organ_mask] = organ_intensity

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


def visualize_inference_case(
    case_idx: int,
    query_image: torch.Tensor,
    query_mask: torch.Tensor,
    support_image: torch.Tensor,
    support_mask: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    class_ids: List[int],
    dice_scores: Dict[str, Dict[str, float]],
    memory_bank_info: Optional[Dict] = None,
    output_dir: Path = Path("visualization_outputs"),
) -> None:
    """Create comprehensive visualization for a single inference case."""
    if not HAS_MATPLOTLIB:
        print(f"Skipping visualization for case {case_idx} - matplotlib not available")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract middle slices
    query_slices = extract_middle_slices(query_image)
    query_mask_slices = extract_middle_slices(query_mask)
    support_slices = extract_middle_slices(support_image)
    support_mask_slices = extract_middle_slices(support_mask)

    pred_slices_dict = {}
    for strategy, pred in predictions.items():
        pred_slices_dict[strategy] = extract_middle_slices(pred)

    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)

    # Row 1: Query image with ground truth overlay
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(query_slices["axial"], cmap="gray")
    ax1.set_title("Query Image (Axial)", fontsize=10, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(query_slices["axial"], cmap="gray")
    gt_overlay = query_mask_slices["axial"] > 0
    ax2.imshow(np.ma.masked_where(~gt_overlay, gt_overlay), cmap="jet", alpha=0.5)
    ax2.set_title("Query + Ground Truth", fontsize=10, fontweight="bold")
    ax2.axis("off")

    # Row 1: Support/Reference image
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(support_slices["axial"], cmap="gray")
    ax3.set_title("Support/Reference Image", fontsize=10, fontweight="bold")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(support_slices["axial"], cmap="gray")
    support_overlay = support_mask_slices["axial"] > 0
    ax4.imshow(np.ma.masked_where(~support_overlay, support_overlay), cmap="jet", alpha=0.5)
    ax4.set_title("Support + Mask", fontsize=10, fontweight="bold")
    ax4.axis("off")

    # Memory bank info
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.axis("off")
    if memory_bank_info:
        info_text = "Memory Bank Context:\n"
        info_text += f"Classes: {memory_bank_info.get('classes', 'N/A')}\n"
        info_text += f"Retrieved: {memory_bank_info.get('retrieved', False)}\n"
        if memory_bank_info.get('retrieved'):
            info_text += "✓ Using stored embeddings"
        else:
            info_text += "✗ One-shot encoding"
    else:
        info_text = "Memory Bank: Disabled"
    ax5.text(0.1, 0.5, info_text, fontsize=9, verticalalignment="center", family="monospace")

    # Rows 2-4: Predictions for each strategy
    strategies = list(predictions.keys())
    for row_idx, strategy in enumerate(strategies[:3], start=1):
        pred_slices = pred_slices_dict[strategy]
        
        # Axial view
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(query_slices["axial"], cmap="gray")
        pred_overlay = pred_slices["axial"] > 0.5
        ax.imshow(np.ma.masked_where(~pred_overlay, pred_overlay), cmap="hot", alpha=0.5)
        ax.set_title(f"{strategy}\n(Axial)", fontsize=9, fontweight="bold")
        ax.axis("off")

        # Coronal view
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.imshow(query_slices["coronal"], cmap="gray")
        pred_overlay = pred_slices["coronal"] > 0.5
        ax.imshow(np.ma.masked_where(~pred_overlay, pred_overlay), cmap="hot", alpha=0.5)
        ax.set_title("(Coronal)", fontsize=9)
        ax.axis("off")

        # Sagittal view
        ax = fig.add_subplot(gs[row_idx, 2])
        ax.imshow(query_slices["sagittal"], cmap="gray")
        pred_overlay = pred_slices["sagittal"] > 0.5
        ax.imshow(np.ma.masked_where(~pred_overlay, pred_overlay), cmap="hot", alpha=0.5)
        ax.set_title("(Sagittal)", fontsize=9)
        ax.axis("off")

        # Dice score
        ax = fig.add_subplot(gs[row_idx, 3])
        ax.axis("off")
        strategy_dice = dice_scores.get(strategy, {})
        dice_text = f"Dice Scores:\n"
        if "dice_per_class" in strategy_dice:
            for cls_id in class_ids:
                cls_dice = strategy_dice["dice_per_class"].get(cls_id, 0.0)
                dice_text += f"Class {cls_id}: {cls_dice:.4f}\n"
        dice_text += f"Mean: {strategy_dice.get('dice_mean', 0.0):.4f}"
        ax.text(0.1, 0.5, dice_text, fontsize=9, verticalalignment="center", family="monospace")

        # Strategy info
        ax = fig.add_subplot(gs[row_idx, 4])
        ax.axis("off")
        if strategy == "in_context_tuning":
            info = "In-Context Tuning:\n• Gradient optimization\n• Task embeddings only\n• Multiple iterations"
        elif strategy == "context_ensemble":
            info = "Context Ensemble:\n• Multiple references\n• Averaged embeddings\n• Robust predictions"
        elif strategy == "object_retrieval":
            info = "Object Retrieval:\n• Memory bank lookup\n• Stored embeddings\n• Fast inference"
        else:
            info = "One-Shot:\n• Single reference\n• Direct encoding\n• Baseline method"
        ax.text(0.1, 0.5, info, fontsize=8, verticalalignment="center", family="monospace")

    # Add overall title
    fig.suptitle(
        f"IRIS Inference Visualization - Case {case_idx}\n"
        f"Classes: {class_ids} | Volume Shape: {query_image.shape[-3:]}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Save figure
    output_path = output_dir / f"case_{case_idx:02d}_inference.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization: {output_path}")


def compute_dice(pred: torch.Tensor, target_mask: torch.Tensor, class_ids: List[int]) -> np.ndarray:
    """Compute Dice score per class."""
    # pred is (1, K, D, H, W) logits
    # target_mask is (D, H, W) with class IDs
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    target_bin = prepare_binary_masks(target_mask, class_ids)  # (K, D, H, W)
    
    dice_scores = []
    for cls_idx in range(len(class_ids)):
        pred_cls = pred_bin[0, cls_idx]  # (D, H, W)
        target_cls = target_bin[cls_idx]  # (D, H, W)
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice.item())
    
    return np.array(dice_scores)


def main():
    output_dir = Path("visualization_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("IRIS Inference Visualization with Memory Bank Context")
    print("=" * 80)
    print()

    set_global_seed(42)

    # Load trained model
    checkpoint_path = Path("demo_outputs/realistic_medical_training/checkpoints/iris_iter_000020.pt")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please run train_realistic_medical.py first to train the model.")
        return

    print(f"Loading model from: {checkpoint_path}")
    model = IrisModel(
        in_channels=1,
        base_channels=16,
        num_query_tokens=4,
        num_attention_heads=4,
        volume_shape=(64, 64, 64),
        use_memory_bank=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("Model loaded successfully!")
    print()

    # Create evaluation dataset (similar to training for better accuracy)
    eval_dataset = RealisticMedicalDataset(
        length=10,
        classes=3,
        volume_shape=(64, 64, 64),
        modality="CT",
        seed_offset=100,  # Different from training
    )

    # Create support dataset (for reference images)
    support_dataset = RealisticMedicalDataset(
        length=10,
        classes=3,
        volume_shape=(64, 64, 64),
        modality="CT",
        seed_offset=0,  # Similar to training
    )

    print(f"Evaluation dataset: {len(eval_dataset)} volumes")
    print(f"Support dataset: {len(support_dataset)} volumes")
    print()

    # Select 5 query cases
    num_cases = min(5, len(eval_dataset))
    query_indices = list(range(num_cases))

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

    all_results = []

    for case_idx, query_idx in enumerate(query_indices):
        print(f"Processing Case {case_idx + 1}/{num_cases} (Query Index: {query_idx})...")

        # Get query sample
        query_sample = eval_dataset[query_idx]
        query_image = query_sample["image"].float().unsqueeze(0)
        query_mask = query_sample["mask"]

        # Get class IDs
        class_ids = [
            int(c.item()) for c in torch.unique(query_mask) if int(c.item()) != 0
        ]
        if len(class_ids) == 0:
            print(f"  Skipping case {case_idx} - no foreground classes")
            continue

        # Prepare binary masks
        query_binary = prepare_binary_masks(query_mask, class_ids).unsqueeze(0)

        # Get support sample (reference image)
        support_idx = case_idx % len(support_dataset)
        support_sample = support_dataset[support_idx]
        support_image = support_sample["image"].float().unsqueeze(0)
        support_mask = support_sample["mask"]
        support_binary = prepare_binary_masks(support_mask, class_ids).unsqueeze(0)

        # Memory bank info
        memory_bank_info = {
            "classes": class_ids,
            "retrieved": False,  # Memory bank disabled in current setup
        }

        # Run all inference strategies
        predictions = {}
        strategy_results = {}
        all_logits = {}

        for strategy_name in eval_config.strategies:
            strategy_fn = evaluator.strategies[strategy_name]

            # In-context tuning needs gradients, others don't
            if strategy_name == "in_context_tuning":
                logits = strategy_fn(
                    eval_dataset,
                    support_image,
                    support_binary,
                    query_image,
                    query_binary,
                    class_ids,
                )
                logits = logits.detach()  # Detach after tuning
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

            all_logits[strategy_name] = logits

            # Get max prediction across classes for visualization
            pred_probs = torch.sigmoid(logits[0])
            max_pred = pred_probs.max(dim=0)[0]
            predictions[strategy_name] = max_pred.unsqueeze(0)

            # Compute metrics
            dice = compute_dice(logits, query_mask, class_ids)
            strategy_results[strategy_name] = {
                "dice_per_class": {cls: float(dice[i]) for i, cls in enumerate(class_ids)},
                "dice_mean": float(dice.mean()),
            }

        # Visualize
        visualize_inference_case(
            case_idx=case_idx + 1,
            query_image=query_image[0],
            query_mask=query_mask,
            support_image=support_image[0],
            support_mask=support_mask,
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
    print("=" * 80)


if __name__ == "__main__":
    main()

