"""Comprehensive evaluation suite covering all IRIS inference strategies."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from iris.model import IrisModel
from iris.model.tuning import InContextTuner
from iris.training.utils import LOGGER

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "SciPy is required for medical evaluation (Hausdorff distance). "
        "Install it via `pip install scipy`."
    ) from exc

def _dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(-3, -2, -1))
    denom = pred.sum(dim=(-3, -2, -1)) + target.sum(dim=(-3, -2, -1))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return dice


def _surface_map(volume: np.ndarray) -> np.ndarray:
    structure = generate_binary_structure(rank=3, connectivity=1)
    eroded = binary_erosion(volume, structure, border_value=0)
    return np.logical_and(volume, np.logical_not(eroded))


def _hausdorff_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    percentile: float = 95.0,
) -> float:
    pred_bin = (torch.sigmoid(pred) > 0.5).cpu().numpy().astype(bool)
    target_bin = target.cpu().numpy().astype(bool)

    if not pred_bin.any() and not target_bin.any():
        return 0.0
    if not pred_bin.any() or not target_bin.any():
        return float("inf")

    pred_surface = _surface_map(pred_bin)
    target_surface = _surface_map(target_bin)

    if not pred_surface.any():
        pred_surface = pred_bin
    if not target_surface.any():
        target_surface = target_bin

    dt_target = distance_transform_edt(~target_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    distances_pred = dt_target[pred_surface]
    distances_target = dt_pred[target_surface]

    if distances_pred.size == 0 and distances_target.size == 0:
        return 0.0

    distances = np.concatenate([distances_pred, distances_target])
    return float(np.percentile(distances, percentile))


def _prepare_binary_masks(mask: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
    masks = []
    for cls in class_ids:
        masks.append((mask == cls).float())
    return torch.stack(masks, dim=0)


def _aggregate_metric_runs(metrics_list: List[Dict[str, object]]) -> Dict[str, object]:
    if len(metrics_list) == 1:
        return metrics_list[0]

    aggregated: Dict[str, object] = {}
    for key in metrics_list[0].keys():
        values = [metrics[key] for metrics in metrics_list if key in metrics]
        sample_value = values[0]
        if isinstance(sample_value, dict):
            merged: Dict[str, List[float]] = {}
            for metrics in values:
                for sub_key, sub_value in metrics.items():  # type: ignore[assignment]
                    merged.setdefault(sub_key, []).append(float(sub_value))
            aggregated[key] = {sub_key: float(np.mean(sub_vals)) for sub_key, sub_vals in merged.items()}
        else:
            numeric_values = np.array(values, dtype=float)
            aggregated[key] = float(np.mean(numeric_values))
            aggregated[f"{key}_runs_std"] = float(np.std(numeric_values, ddof=0))
    return aggregated


@dataclass
class EvaluationConfig:
    in_distribution: Sequence[torch.utils.data.Dataset] = field(default_factory=list)
    out_of_distribution: Sequence[torch.utils.data.Dataset] = field(default_factory=list)
    novel_classes: Sequence[torch.utils.data.Dataset] = field(default_factory=list)
    num_episodes: int = 16
    ensemble_size: int = 4
    tuner_steps: int = 25
    tuner_lr: float = 5e-4
    strategies: Sequence[str] = ("one_shot", "context_ensemble", "object_retrieval", "in_context_tuning")
    device: Optional[str] = None
    baseline_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    repetitions: int = 1
    random_seed: Optional[int] = None


class MedicalEvaluationSuite:
    """
    Evaluate the IRIS model across the dimensions highlighted in Section 4.
    """

    def __init__(
        self,
        model: IrisModel,
        config: EvaluationConfig,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.logger = logger or LOGGER
        self.device = torch.device(
            config.device or next(model.parameters()).device
        )
        self.model.to(self.device)

        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
            np.random.seed(config.random_seed)

        self.strategies = {
            "one_shot": self._one_shot_inference,
            "context_ensemble": self._context_ensemble_inference,
            "object_retrieval": self._memory_retrieval_inference,
            "in_context_tuning": self._in_context_tuning_inference,
        }

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def evaluate(self) -> Dict[str, Dict[str, Dict[str, dict]]]:
        results: Dict[str, Dict[str, Dict[str, dict]]] = {}
        results["in_distribution"] = self._evaluate_group(self.config.in_distribution, "in-distribution")
        results["out_of_distribution"] = self._evaluate_group(self.config.out_of_distribution, "OOD")
        results["novel_classes"] = self._evaluate_group(self.config.novel_classes, "novel-class")
        return results

    # ------------------------------------------------------------------ #
    # Evaluation helpers
    # ------------------------------------------------------------------ #
    def _evaluate_group(
        self,
        datasets: Sequence[torch.utils.data.Dataset],
        label: str,
    ) -> Dict[str, Dict[str, dict]]:
        group_metrics: Dict[str, Dict[str, dict]] = {}
        for dataset in datasets:
            dataset_name = getattr(dataset, "dataset_name", dataset.__class__.__name__)
            self.logger.info("Evaluating %s dataset '%s'", label, dataset_name)
            dataset_metrics: Dict[str, dict] = {"strategies": {}}
            for strategy_name in self.config.strategies:
                if strategy_name not in self.strategies:
                    raise KeyError(f"Unknown evaluation strategy: {strategy_name}")
                runs: List[Dict[str, object]] = []
                for rep in range(self.config.repetitions):
                    self.logger.debug(
                        "Strategy %s repetition %d/%d",
                        strategy_name,
                        rep + 1,
                        self.config.repetitions,
                    )
                    metrics = self._evaluate_dataset_with_strategy(dataset, strategy_name)
                    runs.append(metrics)
                dataset_metrics["strategies"][strategy_name] = _aggregate_metric_runs(runs)
            if dataset_name in self.config.baseline_scores:
                dataset_metrics["baselines"] = self.config.baseline_scores[dataset_name]
            group_metrics[dataset_name] = dataset_metrics
        return group_metrics

    def _evaluate_dataset_with_strategy(
        self,
        dataset: torch.utils.data.Dataset,
        strategy_name: str,
    ) -> Dict[str, object]:
        dice_scores: List[torch.Tensor] = []
        hd_scores: List[torch.Tensor] = []
        inference_times: List[float] = []
        memory_usage: List[float] = []
        per_class_dice: Dict[int, List[float]] = {}
        per_class_hd: Dict[int, List[float]] = {}

        strategy_fn = self.strategies[strategy_name]

        for _ in range(self.config.num_episodes):
            support_idx, query_idx = self._sample_indices(dataset)
            support_sample = dataset[support_idx]
            query_sample = dataset[query_idx]
            support_image = support_sample["image"].float().to(self.device)
            support_mask = support_sample["mask"].to(self.device)
            query_image = query_sample["image"].float().to(self.device)
            query_mask = query_sample["mask"].to(self.device)

            class_ids = [
                int(c.item()) for c in torch.unique(support_mask) if int(c.item()) != 0
            ]
            if len(class_ids) == 0:
                continue

            binary_support = _prepare_binary_masks(support_mask, class_ids).unsqueeze(0).to(self.device)
            binary_query = _prepare_binary_masks(query_mask, class_ids).unsqueeze(0).to(self.device)
            support_image = support_image.unsqueeze(0)
            query_image = query_image.unsqueeze(0)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            start = time.perf_counter()
            logits = strategy_fn(
                dataset,
                support_image,
                binary_support,
                query_image,
                binary_query,
                class_ids,
            )
            elapsed = time.perf_counter() - start
            inference_times.append(elapsed)

            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            elif psutil:
                process = psutil.Process()
                mem = process.memory_info().rss / (1024 ** 2)
            else:
                mem = 0.0
            memory_usage.append(mem)

            dice = _dice_score(logits, binary_query)
            dice_scores.append(dice.cpu())
            for cls_idx, cls in enumerate(class_ids):
                per_class_dice.setdefault(cls, []).append(float(dice[0, cls_idx].cpu()))
                cls_logit = logits[:, cls_idx : cls_idx + 1]
                cls_target = binary_query[:, cls_idx : cls_idx + 1]
                hd_value = _hausdorff_distance(cls_logit, cls_target)
                per_class_hd.setdefault(cls, []).append(hd_value)
            hd_scores.append(
                torch.tensor([per_class_hd[cls][-1] for cls in class_ids], dtype=torch.float32)
            )

        if not dice_scores:
            return {
                "dice_mean": 0.0,
                "dice_std": 0.0,
                "hausdorff_mean": float("inf"),
                "hausdorff_std": float("inf"),
                "inference_time_mean": 0.0,
                "inference_time_std": 0.0,
                "memory_mb_mean": 0.0,
                "memory_mb_std": 0.0,
                "per_class_dice": {},
                "per_class_hausdorff": {},
            }

        stacked = torch.stack(dice_scores, dim=0)
        dice_mean = float(stacked.mean().item())
        dice_std = float(stacked.std(unbiased=False).item())

        hd_tensor = torch.stack(hd_scores, dim=0)
        hd_mean = float(torch.mean(hd_tensor).item())
        hd_std = float(torch.std(hd_tensor, unbiased=False).item())

        class_dice_means = {cls: float(np.mean(scores)) for cls, scores in per_class_dice.items()}
        class_hd_means = {cls: float(np.mean(scores)) for cls, scores in per_class_hd.items()}

        return {
            "dice_mean": dice_mean,
            "dice_std": dice_std,
            "hausdorff_mean": hd_mean,
            "hausdorff_std": hd_std,
            "inference_time_mean": float(np.mean(inference_times)),
            "inference_time_std": float(np.std(inference_times)),
            "memory_mb_mean": float(np.mean(memory_usage)),
            "memory_mb_std": float(np.std(memory_usage)),
            "per_class_dice": class_dice_means,
            "per_class_hausdorff": class_hd_means,
        }

    def _sample_indices(self, dataset: torch.utils.data.Dataset) -> Tuple[int, int]:
        if len(dataset) < 2:
            raise ValueError("Dataset must contain at least two samples for evaluation episodes.")
        indices = np.random.choice(len(dataset), size=2, replace=False)
        return int(indices[0]), int(indices[1])

    # ------------------------------------------------------------------ #
    # Inference strategies
    # ------------------------------------------------------------------ #
    def _one_shot_inference(
        self,
        dataset: torch.utils.data.Dataset,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        query_images: torch.Tensor,
        query_masks: torch.Tensor,
        class_ids: Sequence[int],
    ) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            task_embeddings = self.model.encode_support(support_images, support_masks)["task_embeddings"]
            outputs = self.model(query_images, task_embeddings)
            return outputs["logits"]

    def _context_ensemble_inference(
        self,
        dataset: torch.utils.data.Dataset,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        query_images: torch.Tensor,
        query_masks: torch.Tensor,
        class_ids: Sequence[int],
    ) -> torch.Tensor:
        if self.config.ensemble_size <= 1:
            return self._one_shot_inference(
                dataset,
                support_images,
                support_masks,
                query_images,
                query_masks,
                class_ids,
            )

        embeddings = []
        with torch.no_grad():
            embeddings.append(self.model.encode_support(support_images, support_masks)["task_embeddings"])
            for _ in range(self.config.ensemble_size - 1):
                idx_support, _ = self._sample_indices(dataset)
                extra_sample = dataset[idx_support]
                extra_image = extra_sample["image"].float().to(self.device).unsqueeze(0)
                extra_mask = extra_sample["mask"].to(self.device)
                extra_binary = _prepare_binary_masks(extra_mask, class_ids).unsqueeze(0).to(self.device)
                embeddings.append(self.model.encode_support(extra_image, extra_binary)["task_embeddings"])

            stacked = torch.stack(embeddings, dim=0).mean(dim=0)
            outputs = self.model(query_images, stacked)
        return outputs["logits"]

    def _memory_retrieval_inference(
        self,
        dataset: torch.utils.data.Dataset,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        query_images: torch.Tensor,
        query_masks: torch.Tensor,
        class_ids: Sequence[int],
    ) -> torch.Tensor:
        if self.model.memory_bank is None:
            return self._one_shot_inference(
                dataset,
                support_images,
                support_masks,
                query_images,
                query_masks,
                class_ids,
            )

        try:
            memory_embeddings = self.model.retrieve_memory_embeddings(class_ids)
        except KeyError:
            return self._one_shot_inference(
                dataset,
                support_images,
                support_masks,
                query_images,
                query_masks,
                class_ids,
            )

        memory_embeddings = memory_embeddings.unsqueeze(0)  # add batch dimension
        outputs = self.model(query_images, memory_embeddings)
        return outputs["logits"]

    def _in_context_tuning_inference(
        self,
        dataset: torch.utils.data.Dataset,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        query_images: torch.Tensor,
        query_masks: torch.Tensor,
        class_ids: Sequence[int],
    ) -> torch.Tensor:
        tuner = InContextTuner(
            model=self.model,
            lr=self.config.tuner_lr,
            steps=self.config.tuner_steps,
        )
        with torch.no_grad():
            initial_embeddings = self.model.encode_support(support_images, support_masks)["task_embeddings"]
        tuned = tuner.tune(
            query_images,
            query_masks,
            initial_embeddings,
            class_ids=[class_ids],
            steps=self.config.tuner_steps,
            update_memory=False,
        )
        outputs = self.model(query_images, tuned)
        return outputs["logits"]


