"""
ExperimentManager for tracking and saving all experiment metadata, configs, history, metrics, plots, and hardware info.
"""
import os
import json
import platform
from datetime import datetime

class ExperimentManager:
    def __init__(self, base_dir="experiments"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def create_experiment(self, name=None):
        if name is None:
            name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = os.path.join(self.base_dir, name)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "model"), exist_ok=True)
        return exp_dir

    def save_config(self, exp_dir, config):
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def save_history(self, exp_dir, history):
        with open(os.path.join(exp_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    def save_metrics(self, exp_dir, metrics):
        with open(os.path.join(exp_dir, "evaluation_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    def save_plot(self, exp_dir, plot_name, fig):
        fig.savefig(os.path.join(exp_dir, "plots", plot_name))

    def save_model(self, exp_dir, model, name="backbone.h5"):
        model.save(os.path.join(exp_dir, "model", name))

    def get_hardware_info(self):
        return {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "python_version": platform.python_version(),
        }
