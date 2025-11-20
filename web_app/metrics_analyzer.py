"""Metrics analysis and loading utilities for IRIS web application."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class MetricsAnalyzer:
    """Analyze and load metrics from training and visualization outputs."""
    
    def __init__(self, base_dir: Path = None):
        """
        Initialize metrics analyzer.
        
        Args:
            base_dir: Base directory of the IRIS project
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.training_metrics_dir = self.base_dir / "outputs" / "training_with_metrics"
        self.viz_dir = self.base_dir / "visualization_outputs"
        
    def get_available_datasets(self) -> List[str]:
        """Get list of datasets with available metrics."""
        datasets = []
        if self.training_metrics_dir.exists():
            for dataset_dir in self.training_metrics_dir.iterdir():
                if dataset_dir.is_dir() and (dataset_dir / "training_metrics.json").exists():
                    datasets.append(dataset_dir.name)
        return sorted(datasets)
    
    def load_training_metrics(self, dataset: str) -> Optional[Dict]:
        """
        Load training metrics for a dataset.
        
        Args:
            dataset: Dataset name (e.g., 'kvasir', 'drive')
            
        Returns:
            Dictionary with training metrics or None if not found
        """
        metrics_file = self.training_metrics_dir / dataset / "training_metrics.json"
        if not metrics_file.exists():
            return None
            
        with metrics_file.open('r') as f:
            return json.load(f)
    
    def load_variant_comparison(self, dataset: str) -> Optional[Dict]:
        """
        Load variant comparison metrics.
        
        Args:
            dataset: Dataset name
            
        Returns:
            Variant comparison data or None
        """
        variant_file = self.viz_dir / f"{dataset}_variants_comparison" / "variants_comparison_summary.json"
        if not variant_file.exists():
            return None
            
        with variant_file.open('r') as f:
            return json.load(f)
    
    def load_iris_context_metrics(self, dataset: str) -> Optional[Dict]:
        """
        Load IRIS context metrics.
        
        Args:
            dataset: Dataset name
            
        Returns:
            IRIS context data or None
        """
        context_file = self.viz_dir / f"{dataset}_iris_context" / "iris_context_summary.json"
        if not context_file.exists():
            return None
            
        with context_file.open('r') as f:
            return json.load(f)
    
    def extract_training_curves(self, dataset: str) -> Optional[Tuple[List[int], List[float], List[float]]]:
        """
        Extract training loss and validation Dice curves.
        
        Args:
            dataset: Dataset name
            
        Returns:
            Tuple of (iterations, training_losses, validation_dice) or None
        """
        metrics = self.load_training_metrics(dataset)
        if not metrics or 'metrics' not in metrics:
            return None
        
        train_loss = metrics['metrics'].get('training_loss', [])
        val_dice = metrics['metrics'].get('validation_dice', [])
        
        iterations = [entry['iteration'] for entry in train_loss]
        losses = [entry['value'] for entry in train_loss]
        dice_scores = [entry['value'] for entry in val_dice] if val_dice else []
        
        return iterations, losses, dice_scores
    
    def get_summary_stats(self, dataset: str) -> Dict:
        """
        Get summary statistics for a dataset.
        
        Args:
            dataset: Dataset name
            
        Returns:
            Dictionary with summary statistics
        """
        metrics = self.load_training_metrics(dataset)
        variant_comp = self.load_variant_comparison(dataset)
        iris_context = self.load_iris_context_metrics(dataset)
        
        stats = {
            'dataset': dataset,
            'training_available': metrics is not None,
            'variant_comparison_available': variant_comp is not None,
            'iris_context_available': iris_context is not None,
        }
        
        if metrics:
            stats.update({
                'total_training_time': metrics.get('total_training_time_seconds', 0),
                'total_iterations': metrics.get('total_iterations', 0),
                'final_train_loss': metrics.get('final_train_loss', 0),
                'final_val_loss': metrics.get('final_val_loss', 0),
                'best_val_dice': metrics.get('best_val_dice', 0),
            })
        
        if variant_comp:
            stats.update({
                'avg_oneshot_dice': variant_comp['average_dice'].get('oneshot', 0),
                'avg_ensemble_dice': variant_comp['average_dice'].get('ensemble', 0),
                'avg_full_dice': variant_comp['average_dice'].get('full', 0),
                'ensemble_improvement': variant_comp['improvements'].get('ensemble_vs_oneshot', 0),
                'memory_bank_contribution': variant_comp['improvements'].get('memory_bank_contribution', 0),
            })
        
        if iris_context:
            stats.update({
                'avg_context_dice': iris_context.get('average_dice', 0),
                'num_context_cases': len(iris_context.get('results', [])),
            })
        
        return stats
    
    def compare_datasets(self, datasets: List[str]) -> Dict:
        """
        Compare multiple datasets.
        
        Args:
            datasets: List of dataset names
            
        Returns:
            Comparison data
        """
        comparison = {
            'datasets': [],
            'training_times': [],
            'best_val_dice': [],
            'oneshot_dice': [],
            'ensemble_dice': [],
            'full_dice': [],
            'memory_bank_contribution': [],
        }
        
        for dataset in datasets:
            stats = self.get_summary_stats(dataset)
            if stats['training_available']:
                comparison['datasets'].append(dataset)
                comparison['training_times'].append(stats.get('total_training_time', 0) / 60)  # Convert to minutes
                comparison['best_val_dice'].append(stats.get('best_val_dice', 0) * 100)  # Convert to percentage
                comparison['oneshot_dice'].append(stats.get('avg_oneshot_dice', 0) * 100)
                comparison['ensemble_dice'].append(stats.get('avg_ensemble_dice', 0) * 100)
                comparison['full_dice'].append(stats.get('avg_full_dice', 0) * 100)
                comparison['memory_bank_contribution'].append(stats.get('memory_bank_contribution', 0))
        
        return comparison
    
    def get_per_case_results(self, dataset: str) -> Optional[List[Dict]]:
        """
        Get per-case results for variant comparison.
        
        Args:
            dataset: Dataset name
            
        Returns:
            List of per-case results or None
        """
        variant_comp = self.load_variant_comparison(dataset)
        if not variant_comp or 'cases' not in variant_comp:
            return None
        
        return variant_comp['cases']
