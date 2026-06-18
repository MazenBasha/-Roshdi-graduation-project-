"""
Baseline comparison: from-scratch iResNet vs pretrained FaceNet/VGGFace2.

Calls evaluate_full.py on the from-scratch checkpoint with --compare-backbone
facenet_vggface2, so both models are scored on identical LFW pairs and a
paired bootstrap test is run.

This is a tiny shim — the heavy lifting is in evaluate_full.py — but it
documents the canonical baseline comparison command so it never gets lost.

An MLflow parent run under experiment=compare_baselines tracks the comparison
itself; the child evaluate_full process opens its own run under experiment=
evaluate_full. After the child finishes, the paired-bootstrap delta from
reports/baselines/metrics.json is lifted into the parent run's metrics so
the dashboard shows the headline 'from_scratch vs pretrained' result without
having to drill into the child.

Run:
    python compare_baselines.py
"""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import config
import mlflow_utils as mlu


def main():
    primary_ckpt = Path("checkpoints/casia_v4_iresnet18.best.pt")
    if not primary_ckpt.exists():
        candidates = sorted(Path("checkpoints").glob("*.best.pt"))
        if not candidates:
            print("No .best.pt checkpoint found under checkpoints/")
            sys.exit(2)
        primary_ckpt = candidates[-1]

    report_dir = Path("reports/baselines")
    cmd = [
        sys.executable, "evaluate_full.py",
        "--lfw-root", "data/sklearn_lfw/lfw_home/lfw_funneled",
        "--pairs", "data/sklearn_lfw/lfw_home/pairs.txt",
        "--checkpoint", str(primary_ckpt),
        "--backbone", "iresnet18",
        "--label", "from_scratch_iresnet18",
        "--compare-backbone", "facenet_vggface2",
        "--compare-label", "pretrained_facenet_vggface2",
        "--report-dir", str(report_dir),
        "--bootstrap", "1000",
    ]

    with mlu.init_run(
        experiment="compare_baselines",
        run_name=f"compare_{primary_ckpt.stem}_vs_facenet_vggface2",
        category="Evaluation",
        params={
            "primary_checkpoint": str(primary_ckpt),
            "primary_backbone": "iresnet18",
            "compare_backbone": "facenet_vggface2",
            "bootstrap_iters": 1000,
            "report_dir": str(report_dir),
        },
        tags={"step": "compare_baselines",
              "primary_backbone": "iresnet18",
              "compare_backbone": "facenet_vggface2"},
    ):
        print("[compare] running:", " ".join(shlex.quote(c) for c in cmd))
        subprocess.run(cmd, check=True)

        # Lift the paired-bootstrap delta + both backbones' headline numbers
        # from the child's metrics.json so the parent run is browsable
        # standalone in the MLflow UI. Skipped silently if the file is missing
        # (e.g. evaluate_full failed before writing), so we never fabricate.
        metrics_path = report_dir / "metrics.json"
        if metrics_path.is_file():
            data = json.loads(metrics_path.read_text())
            primary = data.get("primary") or {}
            secondary = data.get("secondary") or {}
            paired = data.get("paired_bootstrap") or {}
            to_log: dict[str, float] = {}
            for label, blob in (("primary", primary), ("secondary", secondary)):
                if not blob:
                    continue
                for k in ("kfold_mean_accuracy", "kfold_std_accuracy",
                          "roc_auc", "eer"):
                    if isinstance(blob.get(k), (int, float)):
                        to_log[f"{label}.{k}"] = float(blob[k])
            for k in ("delta_accuracy", "ci_lo", "ci_hi", "p_value_one_sided"):
                if isinstance(paired.get(k), (int, float)):
                    to_log[f"paired.{k}"] = float(paired[k])
            mlu.log_metrics_flat(to_log)
            mlu.log_artifact_file(metrics_path, artifact_path="baselines")
            for extra in ("summary.md", "threshold.json"):
                p = report_dir / extra
                if p.is_file():
                    mlu.log_artifact_file(p, artifact_path="baselines")
            for sub in report_dir.iterdir():
                if sub.is_dir():
                    mlu.log_artifact_dir(sub, artifact_path=f"baselines/{sub.name}")


if __name__ == "__main__":
    main()
