"""Demonstration utilities showcasing IRIS medical capabilities."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from iris.model import IrisModel
from iris.training.evaluation import MedicalEvaluationSuite, _dice_score, _hausdorff_distance
from iris.training.utils import LOGGER, ensure_directory, set_global_seed
from iris.training.visualization import (
    plot_performance_dashboard,
    render_multi_planar_views,
)


@dataclass
class ClinicalDemoConfig:
    """
    Configuration governing the medical demonstration workflow.
    """

    num_examples: int = 3
    strategies: Sequence[str] = ("one_shot", "context_ensemble", "object_retrieval", "in_context_tuning")
    output_dir: str | Path = "demo_outputs"
    save_visualizations: bool = True
    save_reports: bool = True
    seed: int = 7
    include_memory_bank_summary: bool = True


class MedicalDemoRunner:
    """
    Coordinate end-to-end demonstrations of IRIS inference capabilities.
    """

    def __init__(
        self,
        model: IrisModel,
        evaluation_suite: MedicalEvaluationSuite,
        config: ClinicalDemoConfig,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.evaluation_suite = evaluation_suite
        self.config = config
        self.logger = logger or LOGGER
        self.output_dir = ensure_directory(config.output_dir)
        self.device = evaluation_suite.device
        self.model.to(self.device)
        set_global_seed(config.seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run_demo(
        self,
        datasets: Sequence[torch.utils.data.Dataset],
    ) -> Dict[str, object]:
        """
        Execute the demonstration across provided datasets.
        """
        case_summaries: List[Dict[str, object]] = []
        for dataset in datasets:
            dataset_name = getattr(dataset, "dataset_name", dataset.__class__.__name__)
            self.logger.info("Running demo on dataset %s", dataset_name)
            for case_id in range(self.config.num_examples):
                summary = self._run_single_case(dataset, dataset_name, case_id)
                if summary is not None:
                    case_summaries.append(summary)

        evaluation_report = self.evaluation_suite.evaluate()
        dashboard_path = None
        if self.config.save_visualizations:
            dashboard_path = self.output_dir / "performance_dashboard.png"
            plot_performance_dashboard(
                evaluation_report.get("in_distribution", {}),
                save_path=dashboard_path,
            )

        report = {
            "cases": case_summaries,
            "evaluation": evaluation_report,
            "dashboard": str(dashboard_path) if dashboard_path else None,
            "clinical_notes": self._clinical_considerations(),
        }

        if self.config.include_memory_bank_summary and self.model.memory_bank is not None:
            report["memory_bank_summary"] = {
                "num_classes": len(list(self.model.memory_bank.items())),
                "classes": list(self.model.memory_bank.summary().keys()),
            }

        if self.config.save_reports:
            report_path = self.output_dir / "demo_report.json"
            with open(report_path, "w", encoding="utf-8") as fp:
                json.dump(report, fp, indent=2)
            self.logger.info("Demo report saved to %s", report_path)

        return report

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _run_single_case(
        self,
        dataset: torch.utils.data.Dataset,
        dataset_name: str,
        case_index: int,
    ) -> Optional[Dict[str, object]]:
        if len(dataset) < 2:
            self.logger.warning("Dataset %s has fewer than two samples; skipping case study.", dataset_name)
            return None
        support_idx = np.random.randint(0, len(dataset))
        query_idx = (support_idx + 1) % len(dataset)

        support_sample = dataset[support_idx]
        query_sample = dataset[query_idx]

        support_image = support_sample["image"].float().to(self.device).unsqueeze(0)
        support_mask = support_sample["mask"].to(self.device)
        query_image = query_sample["image"].float().to(self.device).unsqueeze(0)
        query_mask = query_sample["mask"].to(self.device)

        class_ids = [
            int(c.item()) for c in torch.unique(support_mask) if int(c.item()) != 0
        ]
        if not class_ids:
            self.logger.debug("No foreground classes in support sample; skipping case.")
            return None

        support_binary = torch.stack([(support_mask == cls).float() for cls in class_ids], dim=0).unsqueeze(0)
        query_binary = torch.stack([(query_mask == cls).float() for cls in class_ids], dim=0).unsqueeze(0)

        case_metrics: Dict[str, dict] = {}
        visual_paths: Dict[str, str] = {}

        support_binary = support_binary.to(self.device)
        query_binary = query_binary.to(self.device)

        for strategy in self.config.strategies:
            strategy_fn = self.evaluation_suite.strategies[strategy]
            start = time.perf_counter()
            logits = strategy_fn(
                dataset,
                support_image,
                support_binary,
                query_image,
                query_binary,
                class_ids,
            )
            elapsed = time.perf_counter() - start

            dice = _dice_score(logits, query_binary)
            hausdorff = [
                _hausdorff_distance(logits[:, idx : idx + 1], query_binary[:, idx : idx + 1])
                for idx in range(len(class_ids))
            ]

            case_metrics[strategy] = {
                "dice_per_class": {cls: float(dice[0, idx].cpu()) for idx, cls in enumerate(class_ids)},
                "dice_mean": float(dice.mean().item()),
                "hausdorff_per_class": {cls: float(hausdorff[idx]) for idx, cls in enumerate(class_ids)},
                "inference_time": elapsed,
            }

            if self.config.save_visualizations:
                try:
                    pred_overlay = torch.sigmoid(logits[0]).max(dim=0)[0]
                    target_overlay = query_binary[0].max(dim=0)[0]
                    reference_overlay = support_binary[0].max(dim=0)[0]
                    fig = render_multi_planar_views(
                        query_image[0],
                        prediction=pred_overlay,
                        target=target_overlay,
                        reference=reference_overlay,
                    )
                    visual_path = self.output_dir / f"{dataset_name}_case{case_index}_{strategy}.png"
                    fig.savefig(visual_path, dpi=200, bbox_inches="tight")
                    visual_paths[strategy] = str(visual_path)
                    plt_close(fig=fig)
                except ImportError:
                    self.logger.debug("Matplotlib not available; skipping visualization for strategy %s.", strategy)

        return {
            "dataset": dataset_name,
            "case_index": case_index,
            "class_ids": class_ids,
            "metrics": case_metrics,
            "visualizations": visual_paths,
        }

    def _clinical_considerations(self) -> Dict[str, str]:
        """
        Provide qualitative guidance for clinical deployment.
        """
        return {
            "cross_modality": "Evaluate memory-bank initialisation when adapting from CT to MRI protocols.",
            "multi_center": "Use evaluation splits reflecting scanner diversity; monitor Dice drop across centres.",
            "few_shot": "Prefer context ensemble or in-context tuning when only a handful of references are available.",
            "efficiency": "Bidirectional attention dominates latency; pre-cache embeddings for frequently observed classes.",
        }


def plt_close(fig) -> None:
    """Utility for closing matplotlib figures without importing pyplot globally in tests."""
    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    plt.close(fig)

