#!/usr/bin/env bash
# Run the full production AI evaluation suite end-to-end.
# Each step writes machine-readable JSON and a human-readable summary.md
# under reports/<step>/.
#
# Usage:
#   ./run_all_evaluations.sh                        # uses casia_v4_iresnet18.best.pt
#   CKPT=checkpoints/two_stage_v2_iresnet18.best.pt ./run_all_evaluations.sh
#
# Skip a step:
#   SKIP_ROBUSTNESS=1 ./run_all_evaluations.sh

set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; fi

CKPT="${CKPT:-checkpoints/casia_v4_iresnet18.best.pt}"
BACKBONE="${BACKBONE:-iresnet18}"
LFW_ROOT="${LFW_ROOT:-data/sklearn_lfw/lfw_home/lfw_funneled}"
PAIRS="${PAIRS:-data/sklearn_lfw/lfw_home/pairs.txt}"
ONNX="${ONNX:-checkpoints/iresnet18.onnx}"
INT8="${INT8:-checkpoints/iresnet18_int8.pt}"

mkdir -p reports

# Auto-discover a probe folder under LFW_ROOT: pick the identity with the
# most images. No hardcoded names. Override with PROBE=... if you want a
# specific identity for the XAI step.
if [ -z "$PROBE" ]; then
  PROBE="$($PY - <<PY
from pathlib import Path
root = Path("$LFW_ROOT")
if root.is_dir():
    candidates = [(d, sum(1 for _ in d.glob("*.jpg"))) for d in root.iterdir() if d.is_dir()]
    candidates = [c for c in candidates if c[1] >= 10]
    candidates.sort(key=lambda x: -x[1])
    if candidates:
        print(candidates[0][0])
PY
)"
fi
if [ -n "$PROBE" ]; then
  log() { echo "[$(date +%H:%M:%S)] $*"; }
  log "Selected probe folder: $PROBE"
fi

stamp() { date +"%H:%M:%S"; }
log() { echo "[$(stamp)] $*"; }

if [ -z "$SKIP_FULL" ]; then
  log "=== 1/8 Full LFW eval (EER / DET / ROC / P/R/F1 / bootstrap) ==="
  "$PY" evaluate_full.py \
      --lfw-root "$LFW_ROOT" --pairs "$PAIRS" \
      --checkpoint "$CKPT" --backbone "$BACKBONE" \
      --label "from_scratch_iresnet18" \
      --compare-backbone facenet_vggface2 \
      --compare-label "pretrained_facenet_vggface2" \
      --bootstrap 1000 --report-dir reports/lfw_full
fi

if [ -z "$SKIP_ROBUSTNESS" ]; then
  log "=== 2/8 Robustness suite (illumination/blur/occlusion/...) ==="
  "$PY" robustness_eval.py \
      --lfw-root "$LFW_ROOT" --pairs "$PAIRS" \
      --checkpoint "$CKPT" --backbone "$BACKBONE" \
      --report-dir reports/robustness
fi

if [ -z "$SKIP_FAIRNESS" ]; then
  log "=== 3/8 Fairness / demographic-slice eval ==="
  "$PY" fairness_eval.py \
      --lfw-root "$LFW_ROOT" --pairs "$PAIRS" \
      --checkpoint "$CKPT" --backbone "$BACKBONE" \
      --threshold-from reports/lfw_full \
      --report-dir reports/fairness
fi

if [ -z "$SKIP_FAILURES" ]; then
  log "=== 4/8 Failure analysis (FP/FN galleries + categorization) ==="
  "$PY" failure_analysis.py \
      --lfw-root "$LFW_ROOT" --pairs "$PAIRS" \
      --checkpoint "$CKPT" --backbone "$BACKBONE" \
      --threshold-from reports/lfw_full \
      --report-dir reports/failures
fi

if [ -z "$SKIP_XAI" ]; then
  log "=== 5/8 Explainability (Top-K + t-SNE + Grad-CAM) ==="
  if [ -n "$PROBE" ] && [ -d "$PROBE" ]; then
    "$PY" xai.py \
        --gallery "$LFW_ROOT" \
        --probes "$PROBE" \
        --checkpoint "$CKPT" --backbone "$BACKBONE" \
        --max-identities 20 --max-per-identity 8 \
        --tsne --gradcam --gradcam-variant vanilla \
        --report-dir reports/xai
  else
    log "XAI step skipped: no usable probe folder found under $LFW_ROOT"
  fi
fi

if [ -z "$SKIP_ABLATION" ]; then
  log "=== 6/8 Ablation (CLAHE x TTA x crop) ==="
  "$PY" ablation_study.py \
      --lfw-root "$LFW_ROOT" --pairs "$PAIRS" \
      --checkpoint "$CKPT" --backbone "$BACKBONE" \
      --max-pairs 1500 --report-dir reports/ablation
fi

if [ -z "$SKIP_BENCH" ]; then
  log "=== 7/8 Production AI metrics (latency / memory / model size) ==="
  EXTRA_ARGS=()
  [ -f "$ONNX" ] && EXTRA_ARGS+=(--onnx "$ONNX")
  [ -f "$INT8" ] && EXTRA_ARGS+=(--int8 "$INT8")
  "$PY" benchmark_production.py \
      --checkpoint "$CKPT" --backbone "$BACKBONE" \
      "${EXTRA_ARGS[@]}" \
      --report-dir reports/production
fi

if [ -z "$SKIP_TESTS" ]; then
  log "=== 8/8 Unit tests (safety/reliability) ==="
  "$PY" -m pytest tests/ -v --tb=short
fi

log "=== aggregate ==="
"$PY" aggregate_reports.py --reports-dir reports --out reports/FINAL_REPORT.md
log "Done. See reports/FINAL_REPORT.md"
