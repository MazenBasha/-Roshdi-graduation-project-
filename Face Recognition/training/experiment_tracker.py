"""
Experiment tracking system for managing training runs.

Features:
- Automatic experiment numbering and directories
- Configuration storage
- Results tracking
- Comparison between experiments
- Results export (JSON, CSV)
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np


class ExperimentTracker:
    """Track and manage experiments."""

    def __init__(self, experiments_root: Path = None):
        """
        Initialize experiment tracker.

        Args:
            experiments_root: Root directory for experiments.
        """
        if experiments_root is None:
            experiments_root = Path(__file__).parent.parent / "experiments"

        self.root = Path(experiments_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_exp_dir = None
        self.current_exp_id = None

    def create_experiment(self, name: str = None) -> Path:
        """
        Create a new experiment directory.

        Args:
            name: Optional experiment name.

        Returns:
            Path to experiment directory.
        """
        # Find next experiment number
        existing_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if existing_dirs:
            last_num = int(existing_dirs[-1].name.split("_")[0])
            next_num = last_num + 1
        else:
            next_num = 1

        exp_id = f"{next_num:03d}"
        exp_name = f"{exp_id}{'_' + name if name else ''}"

        exp_dir = self.root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        self.current_exp_dir = exp_dir
        self.current_exp_id = exp_id

        print(f"\n[Experiment] Created: {exp_dir}")

        return exp_dir

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save experiment configuration.

        Args:
            config: Configuration dictionary.
        """
        if self.current_exp_dir is None:
            raise RuntimeError("No experiment created. Call create_experiment first.")

        config["timestamp"] = datetime.now().isoformat()
        config["experiment_id"] = self.current_exp_id

        config_path = self.current_exp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"[Experiment] Config saved: {config_path}")

    def save_training_history(self, history: Dict[str, List[float]]) -> None:
        """
        Save training history.

        Args:
            history: Training history dictionary.
        """
        if self.current_exp_dir is None:
            raise RuntimeError("No experiment created. Call create_experiment first.")

        history_path = self.current_exp_dir / "training_history.json"

        # Convert numpy arrays to lists
        clean_history = {}
        for key, values in history.items():
            if isinstance(values, np.ndarray):
                clean_history[key] = values.tolist()
            else:
                clean_history[key] = values

        with open(history_path, "w") as f:
            json.dump(clean_history, f, indent=2)

        print(f"[Experiment] History saved: {history_path}")

    def save_metrics(self, metrics: Dict[str, float], stage: str = "final") -> None:
        """
        Save metrics.

        Args:
            metrics: Metrics dictionary.
            stage: 'final', 'best', 'val', etc.
        """
        if self.current_exp_dir is None:
            raise RuntimeError("No experiment created. Call create_experiment first.")

        metrics_path = self.current_exp_dir / f"metrics_{stage}.json"

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[Experiment] Metrics ({stage}) saved: {metrics_path}")

    def save_logs(self, log_text: str, filename: str = "training.log") -> None:
        """
        Save training logs.

        Args:
            log_text: Log text.
            filename: Log filename.
        """
        if self.current_exp_dir is None:
            raise RuntimeError("No experiment created. Call create_experiment first.")

        log_path = self.current_exp_dir / filename
        with open(log_path, "w") as f:
            f.write(log_text)

        print(f"[Experiment] Logs saved: {log_path}")

    def list_experiments(self) -> List[Dict]:
        """
        List all experiments.

        Returns:
            List of experiment info dictionaries.
        """
        experiments = []

        for exp_dir in sorted(self.root.iterdir()):
            if not exp_dir.is_dir():
                continue

            config_file = exp_dir / "config.json"
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = json.load(f)

                metrics_file = exp_dir / "metrics_final.json"
                metrics = {}
                if metrics_file.exists():
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)

                experiments.append({
                    "id": exp_dir.name,
                    "config": config,
                    "metrics": metrics,
                    "path": str(exp_dir),
                })

        return experiments

    def compare_experiments(
        self, metric_keys: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare experiments in a table.

        Args:
            metric_keys: Metrics to compare (if None, compare all).

        Returns:
            DataFrame with experiment comparison.
        """
        experiments = self.list_experiments()

        if not experiments:
            print("No experiments to compare.")
            return pd.DataFrame()

        rows = []
        for exp in experiments:
            row = {"Experiment": exp["id"]}

            # Add config values
            row.update(exp["config"])

            # Add metrics
            row.update(exp["metrics"])

            rows.append(row)

        df = pd.DataFrame(rows)

        # Filter columns if specified
        if metric_keys:
            keep_cols = ["Experiment"] + metric_keys
            keep_cols = [c for c in keep_cols if c in df.columns]
            df = df[keep_cols]

        return df

    def generate_report(self, exp_id: str = None) -> str:
        """
        Generate experiment report.

        Args:
            exp_id: Experiment ID (if None, use current).

        Returns:
            Report string.
        """
        if exp_id is None:
            exp_dir = self.current_exp_dir
        else:
            # Find experiment by ID
            exp_dirs = [d for d in self.root.iterdir() if d.name.startswith(exp_id)]
            if not exp_dirs:
                raise ValueError(f"Experiment not found: {exp_id}")
            exp_dir = exp_dirs[0]

        # Load files
        config = {}
        history = {}
        metrics = {}

        config_file = exp_dir / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)

        history_file = exp_dir / "training_history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)

        metrics_file = exp_dir / "metrics_final.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

        # Generate report
        report = f"""
{'=' * 80}
EXPERIMENT REPORT: {exp_dir.name}
{'=' * 80}

CONFIGURATION
{'-' * 80}
"""
        for key, value in config.items():
            if key != "timestamp":
                report += f"{key:.<40} {value}\n"

        if history:
            report += f"""
TRAINING HISTORY
{'-' * 80}
Epochs: {len(history.get('loss', []))}
"""
            if "loss" in history:
                report += f"Initial Loss: {history['loss'][0]:.6f}\n"
                report += f"Final Loss:   {history['loss'][-1]:.6f}\n"
                report += f"Best Loss:    {min(history['loss']):.6f}\n"

            if "val_loss" in history:
                report += f"\nInitial Val Loss: {history['val_loss'][0]:.6f}\n"
                report += f"Final Val Loss:   {history['val_loss'][-1]:.6f}\n"
                report += f"Best Val Loss:    {min(history['val_loss']):.6f}\n"

            if "accuracy" in history:
                report += f"\nInitial Accuracy: {history['accuracy'][0]:.4f}\n"
                report += f"Final Accuracy:   {history['accuracy'][-1]:.4f}\n"

            if "val_accuracy" in history:
                report += f"Initial Val Accuracy: {history['val_accuracy'][0]:.4f}\n"
                report += f"Final Val Accuracy:   {history['val_accuracy'][-1]:.4f}\n"
                report += f"Best Val Accuracy:    {max(history['val_accuracy']):.4f}\n"

        if metrics:
            report += f"""
FINAL METRICS
{'-' * 80}
"""
            for key, value in metrics.items():
                report += f"{key:.<40} {value}\n"

        report += "\n" + "=" * 80 + "\n"

        return report


def create_experiment_tracker(root: Path = None) -> ExperimentTracker:
    """Create experiment tracker."""
    return ExperimentTracker(root)
