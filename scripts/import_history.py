"""
Import verifiable training histories from logs/<timestamp>/ into MLflow.

Each logs/<ts>/ directory left behind by train.py contains:
    args.json     -- the exact CLI args used for the run
    history.json  -- a list of per-epoch dicts: {stage, epoch, train:{loss,acc,seconds}, val:{...}}
    events.*      -- TensorBoard event file (also logged as an artifact)

We replay each epoch into MLflow under experiment="train" so old training runs
that pre-date the MLflow integration become first-class entries in the
dashboard. Nothing is fabricated: only fields literally present in the JSON
are logged, and we tag every imported run `historical=true` +
`source_log_dir=<dirname>` so you can always tell it apart from a fresh run.

If a matching checkpoint exists in checkpoints/ (matched against the
`output` arg from args.json, plus its .best.pt sibling), it's logged as an
artifact under `checkpoints/`.

Idempotent: a second invocation skips any logs/ dir that already has an
imported run with the same `source_log_dir` tag.

Usage (from project root):
    python scripts/import_history.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlflow_utils as mlu  # noqa: E402

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    print("[import_history] mlflow is not installed; nothing to import")
    sys.exit(0)


EXPERIMENT = "train"
CATEGORY = "Training"


def parse_dir_timestamp(dirname: str) -> int | None:
    """Convert a `YYYYMMDD_HHMMSS` log dir name into an epoch-ms timestamp,
    interpreted as local time. Returns None if it doesn't match the pattern."""
    try:
        dt = datetime.strptime(dirname, "%Y%m%d_%H%M%S")
        return int(dt.replace(tzinfo=None).timestamp() * 1000)
    except ValueError:
        return None


def find_checkpoints(args: dict) -> list[Path]:
    """Resolve every checkpoint file the run actually produced.
    We log both the periodic .pt and the .best.pt sibling when present."""
    out = []
    raw = args.get("output")
    if not raw:
        return out
    p = ROOT / raw
    if p.is_file():
        out.append(p)
    best = p.with_suffix("")
    best = best.with_suffix(".best.pt")
    if best.is_file() and best not in out:
        out.append(best)
    return out


def already_imported(client: MlflowClient, exp_id: str, source: str) -> bool:
    """Skip dirs we've already imported."""
    runs = client.search_runs(
        [exp_id],
        filter_string=f"tags.source_log_dir = '{source}'",
        max_results=1,
    )
    return bool(runs)


def import_one(log_dir: Path, client: MlflowClient, exp_id: str) -> str | None:
    args_path = log_dir / "args.json"
    hist_path = log_dir / "history.json"
    if not args_path.is_file() or not hist_path.is_file():
        return None

    try:
        args = json.loads(args_path.read_text())
        history = json.loads(hist_path.read_text())
    except Exception as e:
        print(f"[import_history] {log_dir.name}: could not parse JSON ({e}); skipped")
        return None
    if not isinstance(history, list) or not history:
        print(f"[import_history] {log_dir.name}: empty history; skipped")
        return None

    if already_imported(client, exp_id, log_dir.name):
        print(f"[import_history] {log_dir.name}: already imported; skipped")
        return None

    ckpt_files = find_checkpoints(args)
    ckpt_stem = Path(args.get("output", log_dir.name)).stem or log_dir.name
    run_name = f"historical_{log_dir.name}_{ckpt_stem}"
    start_ms = parse_dir_timestamp(log_dir.name) or int(time.time() * 1000)

    run = client.create_run(
        experiment_id=exp_id,
        start_time=start_ms,
        tags={
            "mlflow.runName": run_name,
            "historical": "true",
            "source_log_dir": log_dir.name,
            "step": "train",
            "backbone": str(args.get("backbone", "")),
            "dataset": Path(str(args.get("data", ""))).name,
        },
    )
    run_id = run.info.run_id

    # Params: every key in args.json, string-coerced + 6kB-capped.
    params_to_log = {}
    for k, v in args.items():
        if v is None:
            continue
        s = str(v)
        params_to_log[k[:240]] = s if len(s) <= 5990 else s[:5990] + "...<truncated>"
    for i in range(0, len(params_to_log), 90):
        chunk = list(params_to_log.items())[i:i + 90]
        client.log_batch(
            run_id=run_id,
            params=[mlflow.entities.Param(k, v) for k, v in chunk],
        )

    # Per-epoch metrics replayed at step = epoch index (global epoch across
    # both stages, since the JSON exposes both stage 1 and stage 2 in order).
    metrics_logged = 0
    n_epochs = 0
    final_train_acc = None
    best_val_acc = None
    best_val_loss = None
    for global_idx, entry in enumerate(history):
        if not isinstance(entry, dict):
            continue
        n_epochs += 1
        epoch_local = int(entry.get("epoch", global_idx))
        stage = int(entry.get("stage", 0))
        ts_ms = start_ms + global_idx * 1000  # monotone, no fake wallclock
        batch_metrics = []

        def add(key, val):
            nonlocal metrics_logged
            if isinstance(val, (int, float)) and val == val:
                batch_metrics.append(
                    mlflow.entities.Metric(key, float(val), ts_ms, global_idx)
                )
                metrics_logged += 1

        add("stage", stage)
        add("epoch_in_stage", epoch_local)
        train = entry.get("train") or {}
        for k in ("loss", "acc", "seconds"):
            if k in train:
                add(f"train.{k}", train[k])
        val = entry.get("val") or {}
        for k in ("loss", "acc", "seconds"):
            if k in val:
                add(f"val.{k}", val[k])

        if isinstance(train.get("acc"), (int, float)):
            final_train_acc = float(train["acc"])
        if isinstance(val.get("acc"), (int, float)):
            v_acc = float(val["acc"])
            if best_val_acc is None or v_acc > best_val_acc:
                best_val_acc = v_acc
        if isinstance(val.get("loss"), (int, float)):
            v_loss = float(val["loss"])
            if best_val_loss is None or v_loss < best_val_loss:
                best_val_loss = v_loss

        if batch_metrics:
            client.log_batch(run_id=run_id, metrics=batch_metrics)

    # Summary metrics (no aggregation invented — these are taken straight
    # from the per-epoch values).
    summary = {"epochs_recorded": n_epochs}
    if final_train_acc is not None:
        summary["final_train_acc"] = final_train_acc
    if best_val_acc is not None:
        summary["best_val_acc"] = best_val_acc
    if best_val_loss is not None:
        summary["best_val_loss"] = best_val_loss
    end_ms = start_ms + max(0, n_epochs) * 1000 + 1000
    client.log_batch(
        run_id=run_id,
        metrics=[mlflow.entities.Metric(k, float(v), end_ms, n_epochs)
                 for k, v in summary.items()],
    )

    # Artifacts: the source JSON files + TensorBoard events + matching .pt(s).
    # MlflowClient.log_artifact(run_id, local_path, artifact_path=None).
    client.log_artifact(run_id, str(args_path), artifact_path="history")
    client.log_artifact(run_id, str(hist_path), artifact_path="history")
    for ev in sorted(log_dir.glob("events.out.tfevents.*")):
        client.log_artifact(run_id, str(ev), artifact_path="tensorboard")
    for ck in ckpt_files:
        client.log_artifact(run_id, str(ck), artifact_path="checkpoints")

    client.set_terminated(run_id, status="FINISHED", end_time=end_ms)
    print(f"[import_history] {log_dir.name}: imported run {run_id} "
          f"({n_epochs} epochs, {metrics_logged} metric points, "
          f"{len(ckpt_files)} ckpt files)")
    return run_id


def main():
    tracking_uri = (mlu.os.environ.get("MLFLOW_TRACKING_URI")
                    or mlu._CFG.get("tracking_uri")
                    or mlu.DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)

    # Make sure the train experiment exists with the right category tag, even
    # if no fresh train.py run has hit it yet.
    mlu._ensure_experiment(EXPERIMENT, category=CATEGORY)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        print(f"[import_history] could not get/create experiment {EXPERIMENT!r}")
        sys.exit(2)
    exp_id = exp.experiment_id

    logs_root = ROOT / "logs"
    if not logs_root.is_dir():
        print("[import_history] no logs/ directory; nothing to do")
        return

    dirs = sorted(d for d in logs_root.iterdir()
                  if d.is_dir() and parse_dir_timestamp(d.name) is not None)
    if not dirs:
        print("[import_history] no timestamped log dirs found")
        return

    print(f"[import_history] {len(dirs)} candidate dirs under {logs_root}")
    imported = 0
    for d in dirs:
        if import_one(d, client, exp_id):
            imported += 1
    print(f"[import_history] done. imported {imported}/{len(dirs)} dirs.")


if __name__ == "__main__":
    main()
