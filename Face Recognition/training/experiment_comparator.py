"""
Experiment comparison and report generation utilities.

Provides tools to:
- Compare multiple experiments
- Generate comparison reports
- Visualize experiment metrics
- Export comparison tables
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from training.experiment_tracker import ExperimentTracker


class ExperimentComparator:
    """Compare and analyze multiple experiments."""

    def __init__(self, experiments_root: Path = None):
        """Initialize comparator."""
        self.tracker = ExperimentTracker(experiments_root)

    def get_all_experiments(self) -> List[Dict]:
        """Get all experiments."""
        return self.tracker.list_experiments()

    def compare_by_metrics(
        self, metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare experiments by specific metrics.

        Args:
            metrics: List of metrics to compare.

        Returns:
            DataFrame with comparison.
        """
        experiments = self.get_all_experiments()

        if not experiments:
            print("No experiments found.")
            return pd.DataFrame()

        rows = []
        for exp in experiments:
            row = {"Experiment": exp["id"]}

            # Add config
            if "config" in exp:
                config = exp["config"]
                row["num_epochs"] = config.get("num_epochs", "?")
                row["batch_size"] = config.get("batch_size", "?")
                row["learning_rate"] = config.get("learning_rate", "?")
                row["augmentation_strength"] = config.get("augmentation_strength", "?")
                row["arcface_margin"] = config.get("arcface_margin", "?")
                row["embedding_dropout"] = config.get("embedding_dropout", "?")
                row["l2_reg"] = config.get("l2_reg", "?")

            # Add metrics
            if "metrics" in exp:
                metrics_dict = exp["metrics"]
                row["epochs_trained"] = metrics_dict.get("epochs_trained", "?")
                row["best_epoch"] = metrics_dict.get("best_epoch", "?")
                row["best_val_loss"] = metrics_dict.get("best_val_loss", "?")
                row["best_val_accuracy"] = metrics_dict.get("best_val_accuracy", "?")
                row["final_val_accuracy"] = metrics_dict.get("final_val_accuracy", "?")
                row["final_train_accuracy"] = metrics_dict.get("final_train_accuracy", "?")

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def plot_all_training_curves(self, output_dir: Path = None):
        """
        Plot training curves for all experiments.

        Args:
            output_dir: Directory to save plots.
        """
        if output_dir is None:
            output_dir = self.tracker.root / "comparison_plots"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        experiments = self.get_all_experiments()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for exp in experiments:
            exp_dir = Path(exp["path"])
            history_file = exp_dir / "training_history.json"

            if not history_file.exists():
                continue

            with open(history_file) as f:
                history = json.load(f)

            epochs = range(1, len(history["loss"]) + 1)

            # Plot loss
            ax1.plot(epochs, history["loss"], label=exp["id"], marker="o", markersize=3)
            ax1.plot(epochs, history["val_loss"], linestyle="--", alpha=0.7)

            # Plot accuracy
            ax2.plot(epochs, np.array(history["accuracy"]) * 100, label=exp["id"], marker="o", markersize=3)
            ax2.plot(epochs, np.array(history["val_accuracy"]) * 100, linestyle="--", alpha=0.7)

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training Accuracy")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(output_dir / "all_training_curves.png"), dpi=150)
        print(f"Plot saved: {output_dir / 'all_training_curves.png'}")

    def generate_comparison_report(self, output_file: Path = None) -> str:
        """
        Generate comprehensive comparison report.

        Args:
            output_file: File to save report.

        Returns:
            Report string.
        """
        experiments = self.get_all_experiments()

        if not experiments:
            return "No experiments found."

        # Sort by best validation accuracy
        experiments_sorted = sorted(
            experiments,
            key=lambda x: x.get("metrics", {}).get("best_val_accuracy", 0),
            reverse=True,
        )

        report = """
═════════════════════════════════════════════════════════════════════════════
                    EXPERIMENT COMPARISON REPORT
═════════════════════════════════════════════════════════════════════════════

"""

        # Summary table
        report += "EXPERIMENT SUMMARY (sorted by validation accuracy)\n"
        report += "─" * 120 + "\n"
        report += f"{'#':<5} {'ID':<10} {'Epochs':<8} {'Best VA':<10} {'Best VL':<10} {'Margin':<8} {'Aug':<8} {'L2':<8}\n"
        report += "─" * 120 + "\n"

        for idx, exp in enumerate(experiments_sorted, 1):
            exp_id = exp["id"]
            config = exp.get("config", {})
            metrics = exp.get("metrics", {})

            epochs = config.get("num_epochs", "?")
            best_val_acc = metrics.get("best_val_accuracy", "?")
            best_val_loss = metrics.get("best_val_loss", "?")
            margin = config.get("arcface_margin", "?")
            aug = config.get("augmentation_strength", "?")[:4]
            l2 = config.get("l2_reg", "?")

            if isinstance(best_val_acc, float):
                best_val_acc = f"{best_val_acc:.4f}"
            if isinstance(best_val_loss, float):
                best_val_loss = f"{best_val_loss:.4f}"

            report += f"{idx:<5} {str(exp_id):<10} {str(epochs):<8} {str(best_val_acc):<10} {str(best_val_loss):<10} {str(margin):<8} {str(aug):<8} {str(l2):<8}\n"

        report += "\n"

        # Top experiment details
        if experiments_sorted:
            top_exp = experiments_sorted[0]
            report += f"""
BEST EXPERIMENT: {top_exp['id']}
{'-' * 120}

Configuration:
"""
            for key, value in top_exp.get("config", {}).items():
                if key != "timestamp":
                    report += f"\n  {key:.<40} {value}"

            report += f"""


Metrics:
"""
            for key, value in top_exp.get("metrics", {}).items():
                if isinstance(value, float):
                    report += f"\n  {key:.<40} {value:.6f}"
                else:
                    report += f"\n  {key:.<40} {value}"

        # Comparison insights
        report += f"""


COMPARATIVE INSIGHTS
{'-' * 120}

Best Validation Accuracy:  {experiments_sorted[0].get('metrics', {}).get('best_val_accuracy', '?'):.6f} ({experiments_sorted[0]['id']})
Worst Validation Accuracy: {experiments_sorted[-1].get('metrics', {}).get('best_val_accuracy', '?'):.6f} ({experiments_sorted[-1]['id']})
Average Validation Accuracy: {np.mean([e.get('metrics', {}).get('best_val_accuracy', 0) for e in experiments_sorted]):.6f}

Best Final Validation Accuracy:  {experiments_sorted[0].get('metrics', {}).get('final_val_accuracy', '?'):.6f}
Best Training Accuracy:          {experiments_sorted[0].get('metrics', {}).get('final_train_accuracy', '?'):.6f}

═════════════════════════════════════════════════════════════════════════════
"""

        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            print(f"Report saved: {output_file}")

        return report

    def export_to_csv(self, output_file: Path = None) -> None:
        """Export comparison to CSV."""
        if output_file is None:
            output_file = self.tracker.root / "experiments_comparison.csv"

        df = self.compare_by_metrics()

        if not df.empty:
            df.to_csv(output_file, index=False)
            print(f"CSV exported: {output_file}")
        else:
            print("No data to export.")

    def print_summary(self) -> None:
        """Print experiment summary to console."""
        df = self.compare_by_metrics()

        if not df.empty:
            print("\n" + "=" * 120)
            print("EXPERIMENT SUMMARY")
            print("=" * 120)
            print(df.to_string(index=False))
            print("=" * 120)
        else:
            print("No experiments found.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare experiments")
    parser.add_argument("--experiments_root", type=str, default="experiments", help="Experiments root directory")
    parser.add_argument("--action", type=str, default="summary", help="Action: summary, report, csv, plot")
    parser.add_argument("--output", type=str, default=None, help="Output file path")

    args = parser.parse_args()

    comparator = ExperimentComparator(Path(args.experiments_root))

    if args.action == "summary":
        comparator.print_summary()
    elif args.action == "report":
        output_file = Path(args.output) if args.output else Path(args.experiments_root) / "COMPARISON_REPORT.md"
        report = comparator.generate_comparison_report(output_file)
        print(report)
    elif args.action == "csv":
        output_file = Path(args.output) if args.output else Path(args.experiments_root) / "experiments_comparison.csv"
        comparator.export_to_csv(output_file)
    elif args.action == "plot":
        output_dir = Path(args.output) if args.output else Path(args.experiments_root) / "comparison_plots"
        comparator.plot_all_training_curves(output_dir)
    else:
        print(f"Unknown action: {args.action}")


if __name__ == "__main__":
    main()
