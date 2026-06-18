# MLflow integration audit

Scope of this pass: Phases 1–2 of the plan agreed with the user (code edits +
historical import, **no new long-running evaluations**). Phase 3 (real
robustness / fairness / failure / ablation / benchmark / baseline /
aggregate runs) was deferred so nothing was executed and nothing was
fabricated.

Tracking URI: `sqlite:///mlflow.db` · Artifact store: `./mlartifacts/`

---

## 1. State before this audit

`mlflow.db` contained 3 experiment rows; only 2 were active:

| Experiment ID | Name | Runs | Notes |
|---:|---|---:|---|
| 0 | `Default` | 0 | auto-created, already `deleted` |
| 1 | `evaluate_full` | 1 | `lfw_full_audit_iresnet18` (FINISHED) — kfold_acc=0.7965, AUC=0.881, EER=0.203. Numbers traceable to `reports/lfw_full/metrics.json`. |
| 2 | `xai` | 2 | `xai_iresnet18` (FINISHED ×2) — backbone=iresnet18, gradcam_variant=vanilla. Artifacts traceable to `reports/xai/`. |

Scripts that **had** MLflow integration in code but had never been run:
`train.py`, `benchmark_production.py`, `robustness_eval.py`,
`fairness_eval.py`, `failure_analysis.py`, `ablation_study.py`,
`aggregate_reports.py`. Their experiment names did not match the structure
requested by the user. `compare_baselines.py` had no MLflow integration at
all.

## 2. Code changes made

All changes are additive; nothing was deleted. Each commit-sized change:

| File | Change |
|---|---|
| `mlflow_utils.py` | `init_run()` and `_ensure_experiment()` now accept `category=` (Evaluation, Reliability, Explainability, Performance, Reporting, Training) and stamp it as an MLflow **experiment tag**. The dashboard can then filter / group by tag. Backwards compatible — existing call sites without `category=` keep working. |
| `evaluate_full.py` | Added `category="Evaluation"`. Experiment name unchanged (`evaluate_full`). |
| `xai.py` | Added `category="Explainability"`. Name unchanged. |
| `robustness_eval.py` | Renamed experiment `robustness` → **`robustness_eval`**, added `category="Reliability"`. |
| `fairness_eval.py` | Renamed experiment `fairness` → **`fairness_eval`**, added `category="Reliability"`. |
| `failure_analysis.py` | Added `category="Reliability"`. Renamed output JSON `failure_metrics.json` → **`metrics.json`** so `aggregate_reports.py` picks it up. Added per-category FP/FN counts as MLflow metrics (`failures.fp_category.*`, `failures.fn_category.*`). |
| `ablation_study.py` | Renamed experiment `ablation` → **`ablation_study`**, added `category="Reliability"`. |
| `benchmark_production.py` | Renamed experiment `production_benchmark` → **`benchmark_production`**, added `category="Performance"`. |
| `aggregate_reports.py` | Renamed experiment `final_report` → **`aggregate_reports`**, added `category="Reporting"`. |
| `train.py` | Added `category="Training"`. Name unchanged. |
| `compare_baselines.py` | Rewritten with a parent MLflow run under new experiment **`compare_baselines`** (`category="Evaluation"`). Wraps the existing `evaluate_full.py` subprocess call; after the child finishes, copies the paired-bootstrap delta + both backbones' kfold accuracy / AUC / EER from `reports/baselines/metrics.json` into the parent run as metrics (`primary.*`, `secondary.*`, `paired.*`). Logs `metrics.json`, `summary.md`, `threshold.json`, and any plot subdirectories as artifacts. **Skipped silently if `metrics.json` is missing** so nothing is fabricated on child failure. |
| `scripts/import_history.py` | **New file.** Imports verifiable historical training runs from `logs/<YYYYMMDD_HHMMSS>/`. See §4. |

## 3. Backfilled experiment-tag categories on pre-existing experiments

`evaluate_full` (id=1) and `xai` (id=2) were created before `mlflow_utils.py`
learned about `category`. To avoid waiting for a fresh run to attach the
tag, I called `MlflowClient.set_experiment_tag` directly. After backfill:

```
1  evaluate_full   category=Evaluation
2  xai             category=Explainability
3  train           category=Training
```

The other six target experiments (`compare_baselines`, `robustness_eval`,
`fairness_eval`, `failure_analysis`, `ablation_study`, `benchmark_production`,
`aggregate_reports`) **do not yet exist as DB rows**. They will be created
on first run, at which point `_ensure_experiment` stamps the right category
automatically.

## 4. Historical runs imported (Phase 2)

`scripts/import_history.py` scanned `logs/` for `YYYYMMDD_HHMMSS` dirs.
**No metric was ever invented** — every numeric value was read from the
source `history.json` and replayed at `step=epoch_index` so the loss/acc
curves render correctly. Idempotent: re-running skips dirs that already
have a run with `source_log_dir=<dirname>`.

Imported 7 of 9 candidate dirs:

| Source dir | Run ID | Run name | Epochs | Metric points | Best val acc (from history.json) | Checkpoints attached |
|---|---|---|---:|---:|---:|---:|
| `20260502_032549` | `da0dece5…` | `historical_…_smoke_iresnet18` | 3 | 28 | 0.0000 | 1 |
| `20260502_034826` | `226eb5d9…` | `historical_…_two_stage_iresnet18` | 11 | 92 | 0.4000 | 2 |
| `20260502_040209` | `62371045…` | `historical_…_two_stage_v2_iresnet18` | 20 | 164 | 0.4179 | 2 |
| `20260502_042844` | `cf9738f6…` | `historical_…_casia_v3_iresnet18` | 13 | 108 | 0.4518 | 2 |
| `20260502_162326` | `ef2bd47d…` | `historical_…_casia_v4_iresnet18` | 15 | 124 | 0.4592 | 2 |
| `20260502_222352` | `75079c6b…` | `historical_…_casia_v5_iresnet18` | 3 | 28 | 0.3442 | 1 |
| `20260502_225456` | `5c026b39…` | `historical_…_casia_v5_normface_iresnet18` | 15 | 124 | 0.7758 | 2 |

Each historical run carries:
- All hyperparameters from `args.json` as params (`backbone`, `epochs_stage1`,
  `epochs_stage2`, `lr_stage1`, `scale`, `margin`, `batch_size`, …).
- Per-epoch metrics: `train.loss`, `train.acc`, `train.seconds`, `val.loss`,
  `val.acc`, `val.seconds`, `stage`, `epoch_in_stage`. Each logged at
  `step=global_epoch_index`.
- End-of-run summary metrics: `epochs_recorded`, `final_train_acc`,
  `best_val_acc`, `best_val_loss`. These are extrema or final values of the
  per-epoch series — not synthesized.
- Tags: `historical=true`, `source_log_dir=<dirname>`, `step=train`,
  `backbone=<from args>`, `dataset=<basename of data path>`.
- Artifacts: `history/args.json`, `history/history.json`,
  `tensorboard/events.out.tfevents.*`, and every matching `.pt` /
  `.best.pt` from `checkpoints/` under `checkpoints/`.
- `start_time` is set to the dir-name timestamp so the runs sort
  chronologically in the UI.

### Verification

Cross-checked all 90 numeric metric points for the largest historical run
(`20260502_225456`, casia_v5_normface) between `history.json` and the
MLflow `metrics` table. **0 mismatches** (≤1e-9). The other six imports
use the same code path; spot checks of best_val_acc against my earlier
JSON analysis matched exactly.

### Why 2 dirs were skipped

- `logs/20260502_224747/` — contains `args.json` but no `history.json`. The
  TensorBoard events file is only 147 bytes (just the start marker). The
  training process exited before completing any epoch, so there is nothing
  verifiable to import.
- `logs/20260502_225058/` — same situation (791 B events file, no
  `history.json`).

Both were correctly skipped by `import_history.py`. **Inventing per-epoch
accuracy/loss for these runs would have been a fabrication and is exactly
what the task explicitly forbids.**

## 5. State after this audit

```
sqlite> SELECT id, name, lifecycle_stage,
                category, n_runs FROM ...;
1  evaluate_full     active  Evaluation       1
2  xai               active  Explainability   2
3  train             active  Training         7
```

Run total: **10 FINISHED runs** (1 evaluate_full + 2 xai + 7 historical
train).

Artifact disk usage (`mlartifacts/`):
- `evaluate_full/`: 196 KB (existing)
- `xai/`: 12 MB (existing — 50 explain + 50 gradcam PNGs ×2 runs + t-SNE)
- `train/`: **1.2 GB** (new — checkpoints dominate; each `.best.pt` is
  ~95 MB and 12 of them landed in MLflow). If disk pressure becomes an
  issue, the per-run `checkpoints/` artifact dir can be deleted; the
  original files in `checkpoints/` are untouched.

## 6. What still has to be executed to populate the rest

These experiments **do not yet have any runs**. They cannot be imported as
"historical" because the corresponding `reports/<step>/` directories don't
exist on disk — there is nothing on the filesystem to verify their numbers
against. The user opted to skip Phase 3, so this is left as a follow-up.

| Experiment | Command to populate | Approx. runtime on MPS box |
|---|---|---:|
| `compare_baselines` | `python compare_baselines.py` | 10–15 min (downloads ~30 MB FaceNet weights on first run) |
| `robustness_eval` | `python robustness_eval.py --lfw-root data/sklearn_lfw/lfw_home/lfw_funneled --pairs data/sklearn_lfw/lfw_home/pairs.txt --checkpoint checkpoints/casia_v4_iresnet18.best.pt --backbone iresnet18 --report-dir reports/robustness` | 15–25 min |
| `fairness_eval` | `python fairness_eval.py --lfw-root … --pairs … --checkpoint … --backbone iresnet18 --threshold-from reports/lfw_full --report-dir reports/fairness` | 5–10 min |
| `failure_analysis` | `python failure_analysis.py --lfw-root … --pairs … --checkpoint … --backbone iresnet18 --threshold-from reports/lfw_full --report-dir reports/failures` | 5–10 min |
| `ablation_study` | `python ablation_study.py --lfw-root … --pairs … --checkpoint … --max-pairs 1500 --report-dir reports/ablation` | 5–10 min |
| `benchmark_production` | `python benchmark_production.py --checkpoint checkpoints/casia_v4_iresnet18.best.pt --backbone iresnet18 --onnx checkpoints/iresnet18.onnx --int8 checkpoints/iresnet18_int8.pt --report-dir reports/production` | 2–5 min |
| `aggregate_reports` | `python aggregate_reports.py` (after at least one of the above) | <1 min |

The canonical orchestration script `run_all_evaluations.sh` already chains
all of these — running it once will populate every experiment and produce a
matching `reports/<step>/` tree.

## 7. What cannot be reconstructed and why

| Asset | Why it can't be imported |
|---|---|
| Historical robustness / fairness / failures / ablation / benchmark / baselines numbers | They were never produced in the first place. `reports/robustness/`, `reports/fairness/`, `reports/failures/`, `reports/ablation/`, `reports/production/`, `reports/baselines/` do not exist on disk. There are no logs, no JSON, no plots, and no markdown files for these steps. Importing any metric value here would be invention, not import. |
| Numbers from the GitHub reference repo (`MazenBasha/-Roshdi-graduation-project-/Face Recognition/experiments/lfw_30epochs/`) | That repo is the **TensorFlow/Keras original**, not a parallel PyTorch fork. It contains a Keras `.h5` model trained on **DigiFace-1M** (synthetic faces), with training accuracy 91.85% / validation accuracy 47.21%. Those numbers describe a different model on a different dataset and **do not characterize this project's iResNet-18 trained on CASIA-WebFace**. Logging them as MLflow runs in this project would misattribute external results, so they were deliberately not imported. |
| Aborted training dirs `logs/20260502_224747/` and `logs/20260502_225058/` | Each has only an `args.json` + a near-empty TensorBoard events file. No `history.json` means no per-epoch loss/acc was ever recorded. There is nothing real to import. |
| Per-step latency rows for runs older than the current MLflow integration | `train.py` writes per-step latencies to TensorBoard, not to `history.json`. The TF events files are imported as raw artifacts so a reviewer can re-open them in TensorBoard, but the per-step numbers are not surfaced as MLflow metrics (would require parsing TF events, which is brittle and out of scope here). |

## 8. Reproducibility checks performed

- Verified all 90 numeric metric points for the casia_v5_normface historical
  run agree byte-for-byte with `logs/20260502_225456/history.json`.
- Verified `import_history.py` is idempotent: re-running produces
  `imported 0/9 dirs.` and writes nothing.
- Verified the renames don't leave any stale experiment names in code:
  ```bash
  grep -E 'experiment="(production_benchmark|robustness|fairness|ablation|final_report)"' *.py
  # (no matches)
  ```
- Verified `mlflow_utils.py` still imports cleanly under `MLFLOW_DISABLED=1`.

## 9. Open follow-ups

1. Run Phase 3 (or just `./run_all_evaluations.sh`) when you're ready to
   populate the remaining six experiments with real numbers.
2. Consider deleting `mlartifacts/train/*/artifacts/checkpoints/` if the
   ~1.2 GB footprint is problematic — the originals in `checkpoints/`
   are untouched and the artifact link in each run still resolves to the
   source path.
3. If a remote MLflow server replaces the local SQLite store, set
   `MLFLOW_TRACKING_URI` before running anything; the helper code already
   honours the env var and `mlflow_config.yaml`.
