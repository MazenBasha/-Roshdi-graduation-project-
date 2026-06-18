from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PARAMS = {
    "model_family": "Residual CNN",
    "task": "Arabic wake-word detection",
    "wake_word": "رشدي",
    "sample_rate_hz": 16000,
    "window_seconds": 1.0,
    "n_mels": 64,
    "n_frames": 32,
    "optimizer": "Adam",
    "loss": "Weighted CrossEntropyLoss",
    "export_format": "PyTorch Lite",
}


def load_metrics(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in raw.items() if isinstance(value, (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Log Rushdey wake-word model metadata to MLflow")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--notebook", type=Path, default=Path("wake_word_final.ipynb"))
    parser.add_argument("--experiment", default="rushdey-wake-word")
    parser.add_argument("--run-name", default="wake-word-rushdey")
    args = parser.parse_args()

    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit("Install MLflow first: pip install mlflow") from exc

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(DEFAULT_PARAMS)
        mlflow.log_param("model_path", str(args.model_path))

        for metric, value in load_metrics(args.metrics_json).items():
            mlflow.log_metric(metric, value)

        if args.model_path.exists():
            mlflow.log_artifact(str(args.model_path), artifact_path="model")
        if args.notebook.exists():
            mlflow.log_artifact(str(args.notebook), artifact_path="notebook")

        mlflow.set_tags(
            {
                "component": "wake_word",
                "runtime": "android_on_device",
                "privacy": "offline_audio_inference",
            }
        )


if __name__ == "__main__":
    main()
