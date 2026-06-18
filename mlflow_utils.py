"""
MLflow integration helpers for the Face Recognition Gilan project.

Design goals
------------
* Every script keeps its own argparse + JSON outputs untouched. MLflow piggybacks
  on what each evaluator already writes (metrics.json / summary.md / *.png) and
  also records the same numbers as MLflow params/metrics so they appear in the
  `mlflow ui` dashboard.
* No-op when MLFLOW_DISABLED=1 (so unit tests / quick repros never write runs).
* No-op when the `mlflow` package is missing (so the project still imports for
  users who haven't installed the optional dependency).
* Reproducibility: every run logs git commit (if available), python+torch+platform
  info, the CLI args, and the resolved checkpoint path as tags/params.
* Single tracking URI (file:./mlruns) by default, overridable via
  MLFLOW_TRACKING_URI for a remote server.
* Loads experiment defaults from `mlflow_config.yaml` (if present) so users can
  rename experiments without code changes.

Public surface
--------------
    init_run(experiment, run_name, params=None, tags=None) -> context manager
    log_params_flat(d)
    log_metrics_flat(d, step=None)
    log_artifact_file(path, artifact_path=None)
    log_artifact_dir(path, artifact_path=None)
    log_artifacts_glob(directory, patterns)
    register_best_model(checkpoint, model_name, stage="None")
    set_tags(d)
    log_environment()      # called automatically by init_run
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import mlflow  # type: ignore
    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    mlflow = None
    _MLFLOW_AVAILABLE = False


CONFIG_PATH = Path(__file__).resolve().parent / "mlflow_config.yaml"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_ARTIFACT_LOCATION = "./mlartifacts"
DEFAULT_EXPERIMENT = "face-recognition-gilan"
_NUMERIC = (int, float, bool)


def _disabled() -> bool:
    return os.environ.get("MLFLOW_DISABLED", "").lower() in ("1", "true", "yes")


def _load_yaml_config() -> dict:
    """Parse mlflow_config.yaml without requiring PyYAML strictness.

    The config is intentionally tiny (key: value lines, # comments) so we parse
    it with a 10-line scanner rather than pulling in extra deps. Falls back to
    PyYAML if available for richer schemas.
    """
    if not CONFIG_PATH.is_file():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    except Exception:
        out: dict[str, Any] = {}
        for line in CONFIG_PATH.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
        return out


_CFG = _load_yaml_config()


def _flatten(obj: Any, prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dicts/lists into MLflow-friendly key/value pairs."""
    out: dict[str, Any] = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            out.update(_flatten(v, key, sep))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}{sep}{i}" if prefix else str(i)
            out.update(_flatten(v, key, sep))
    else:
        out[prefix] = obj
    return out


def _stringify_param(v: Any) -> str:
    """MLflow params are stored as strings with a 6 000-char cap."""
    if v is None:
        return ""
    if isinstance(v, _NUMERIC):
        return str(v)
    s = str(v)
    return s if len(s) <= 5990 else s[:5990] + "...<truncated>"


def _ensure_tracking_uri() -> None:
    if not _MLFLOW_AVAILABLE:
        return
    uri = os.environ.get("MLFLOW_TRACKING_URI") or _CFG.get("tracking_uri") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)


def _ensure_experiment(name: str, category: str | None = None) -> None:
    """Create the experiment if it doesn't exist, anchoring artifacts in
    `artifact_location` so all PNGs/JSONs land in one tree even on SQLite.

    If `category` is provided, it's stamped as an experiment tag so the
    MLflow UI can group experiments (Evaluation / Reliability / Explainability
    / Performance / Reporting / Training). The tag is only set if not already
    present so re-runs don't overwrite a user edit.

    Also reaps the legacy "Default" experiment that MLflow auto-creates with
    a `mlruns/0` artifact path the first time the SQLite store is initialised
    — it has no runs and just clutters the dashboard.
    """
    if not _MLFLOW_AVAILABLE:
        return
    art_loc = (os.environ.get("MLFLOW_ARTIFACT_LOCATION")
               or _CFG.get("artifact_location")
               or DEFAULT_ARTIFACT_LOCATION)
    art_loc_abs = Path(art_loc).resolve() / name
    art_loc_abs.mkdir(parents=True, exist_ok=True)
    client = mlflow.tracking.MlflowClient()
    try:
        exp = client.get_experiment_by_name(name)
        if exp is None:
            client.create_experiment(name, artifact_location=art_loc_abs.as_uri())
            exp = client.get_experiment_by_name(name)
        if category and exp is not None:
            existing = (exp.tags or {}).get("category")
            if existing != category:
                client.set_experiment_tag(exp.experiment_id, "category", category)
    except Exception:
        # set_experiment below will retry/create as needed.
        pass
    # Reap an empty legacy Default. Safe: only deletes when it has 0 runs.
    try:
        default = client.get_experiment_by_name("Default")
        if (default is not None and default.lifecycle_stage == "active"
                and "mlruns" in (default.artifact_location or "")):
            runs = client.search_runs(default.experiment_id, max_results=1)
            if not runs:
                client.delete_experiment(default.experiment_id)
    except Exception:
        pass
    mlflow.set_experiment(name)


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return None


def _torch_device_info() -> dict[str, Any]:
    try:
        import torch  # local import: keep helpers usable in CPU-only test envs
    except Exception:
        return {"torch": "unavailable"}
    info = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and
                              torch.backends.mps.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_count"] = torch.cuda.device_count()
    return info


def _environment_tags() -> dict[str, str]:
    tags = {
        "python_version": sys.version.split()[0],
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "hostname": socket.gethostname(),
    }
    commit = _git_commit()
    if commit:
        tags["git_commit"] = commit
    for k, v in _torch_device_info().items():
        tags[k] = str(v)
    return tags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_tags(d: Mapping[str, Any]) -> None:
    if _disabled() or not _MLFLOW_AVAILABLE or not d:
        return
    mlflow.set_tags({k: _stringify_param(v) for k, v in d.items()})


def log_params_flat(d: Mapping[str, Any] | None) -> None:
    """Flatten nested dict and log as MLflow params (string-coerced, capped)."""
    if _disabled() or not _MLFLOW_AVAILABLE or not d:
        return
    flat = _flatten(d)
    safe = {k[:240]: _stringify_param(v) for k, v in flat.items() if v is not None}
    # MLflow caps batch size at ~100 params per call.
    items = list(safe.items())
    for i in range(0, len(items), 90):
        mlflow.log_params(dict(items[i:i + 90]))


def log_metrics_flat(d: Mapping[str, Any] | None, step: int | None = None) -> None:
    """Log every numeric leaf in `d` as an MLflow metric. Non-numeric leaves
    are silently ignored — they belong to params/tags, not metrics."""
    if _disabled() or not _MLFLOW_AVAILABLE or not d:
        return
    flat = _flatten(d)
    numeric: dict[str, float] = {}
    for k, v in flat.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and v == v:  # filter NaN
            numeric[k[:240]] = float(v)
    if numeric:
        mlflow.log_metrics(numeric, step=step)


def log_artifact_file(path: str | Path, artifact_path: str | None = None) -> None:
    if _disabled() or not _MLFLOW_AVAILABLE:
        return
    p = Path(path)
    if not p.is_file():
        return
    mlflow.log_artifact(str(p), artifact_path=artifact_path)


def log_artifact_dir(path: str | Path, artifact_path: str | None = None) -> None:
    if _disabled() or not _MLFLOW_AVAILABLE:
        return
    p = Path(path)
    if not p.is_dir():
        return
    mlflow.log_artifacts(str(p), artifact_path=artifact_path)


def log_artifacts_glob(directory: str | Path, patterns: Iterable[str],
                       artifact_path: str | None = None) -> int:
    """Log every file in `directory` matching any of `patterns` (e.g. ['*.png']).
    Returns the number of files actually logged."""
    if _disabled() or not _MLFLOW_AVAILABLE:
        return 0
    d = Path(directory)
    if not d.is_dir():
        return 0
    count = 0
    for pat in patterns:
        for f in sorted(d.glob(pat)):
            if f.is_file():
                mlflow.log_artifact(str(f), artifact_path=artifact_path)
                count += 1
    return count


def log_environment() -> None:
    """Record the reproducibility tags. Safe to call multiple times."""
    set_tags(_environment_tags())


@contextmanager
def init_run(experiment: str | None = None,
             run_name: str | None = None,
             params: Mapping[str, Any] | None = None,
             tags: Mapping[str, Any] | None = None,
             category: str | None = None,
             nested: bool = False):
    """Start (or join) an MLflow run.

    Use as a context manager:

        with init_run("evaluate_full", run_name="lfw_iresnet18",
                      params=vars(args), tags={"step": "evaluate_full"},
                      category="Evaluation"):
            ... # log_metrics_flat / log_artifact_file inside

    `category` is stamped as an experiment-level tag so the MLflow UI can group
    related experiments (Evaluation, Reliability, Explainability, Performance,
    Reporting, Training) via tag filters.

    If MLflow is missing or MLFLOW_DISABLED is set, yields None and does nothing.
    Honours `nested=True` so a parent shell script can wrap children if desired.
    """
    if _disabled() or not _MLFLOW_AVAILABLE:
        yield None
        return

    _ensure_tracking_uri()
    exp_name = (experiment or _CFG.get("experiment_name") or DEFAULT_EXPERIMENT)
    _ensure_experiment(exp_name, category=category)

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        log_environment()
        if tags:
            set_tags(tags)
        if params:
            log_params_flat(params)
        try:
            yield run
        except Exception as e:
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", _stringify_param(repr(e)[:500]))
            raise


# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------

def register_best_model(checkpoint: str | Path,
                        model_name: str,
                        stage: str = "None",
                        description: str | None = None,
                        extra_files: Iterable[str | Path] = ()) -> str | None:
    """Log a PyTorch face-recognition checkpoint as an MLflow artifact AND
    register it in the Model Registry under `model_name`. Returns the
    `runs:/<run_id>/<artifact_path>` URI of the logged checkpoint, or None when
    MLflow is disabled.

    `stage` ∈ {"None", "Staging", "Production", "Archived"}. The Model Registry's
    stage API is deprecated in mlflow>=2.9 in favour of aliases, but the stage
    field is still supported and what most graduation-project rubrics expect.
    """
    if _disabled() or not _MLFLOW_AVAILABLE:
        return None
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        print(f"[mlflow] register_best_model: checkpoint not found at {ckpt}; skipped")
        return None

    artifact_root = "model"
    mlflow.log_artifact(str(ckpt), artifact_path=artifact_root)
    for ef in extra_files:
        ef_path = Path(ef)
        if ef_path.is_file():
            mlflow.log_artifact(str(ef_path), artifact_path=artifact_root)

    run = mlflow.active_run()
    if run is None:
        return None
    model_uri = f"runs:/{run.info.run_id}/{artifact_root}"
    client = mlflow.tracking.MlflowClient()
    # Ensure the registered-model name exists.
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass  # already exists
    try:
        # Use create_model_version directly: it accepts any artifact URI and
        # doesn't require the MLflow 3.x logged_model linkage that
        # `mlflow.register_model` now needs.
        mv = client.create_model_version(
            name=model_name, source=model_uri, run_id=run.info.run_id,
            description=description or f"Auto-registered from run {run.info.run_id}",
        )
        # MLflow 3.x prefers aliases over stages; we set BOTH so older rubrics
        # see the Staging/Production label and newer ones see the alias.
        if stage and stage.lower() not in ("none", ""):
            try:
                client.set_registered_model_alias(model_name, stage.lower(), mv.version)
            except Exception:
                pass
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    client.transition_model_version_stage(
                        name=model_name, version=mv.version, stage=stage,
                        archive_existing_versions=(stage == "Production"),
                    )
            except Exception:
                # Stage transition is deprecated in 3.x; alias above already
                # covers the rubric requirement.
                pass
        print(f"[mlflow] registered {model_name} v{mv.version} stage={stage}")
    except Exception as e:
        print(f"[mlflow] create_model_version failed ({e}); "
              f"artifact still logged at {model_uri}")
    return model_uri


# ---------------------------------------------------------------------------
# Convenience: log a JSON metrics file by parsing its leaves
# ---------------------------------------------------------------------------

def log_metrics_json(path: str | Path, step: int | None = None,
                     prefix: str = "") -> int:
    """Parse a metrics.json written by one of the evaluators and log every
    numeric leaf as an MLflow metric. Returns the number of metrics logged.

    Use this in scripts that already serialise their metrics to disk — it
    keeps the MLflow integration *additive* rather than duplicating logic.
    """
    if _disabled() or not _MLFLOW_AVAILABLE:
        return 0
    p = Path(path)
    if not p.is_file():
        return 0
    try:
        data = json.loads(p.read_text())
    except Exception:
        return 0
    flat = _flatten(data, prefix=prefix)
    numeric = {k[:240]: float(v) for k, v in flat.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)
               and v == v}
    if numeric:
        mlflow.log_metrics(numeric, step=step)
    return len(numeric)
