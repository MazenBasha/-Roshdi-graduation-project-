# Rushdey Voice Commands

This directory contains the voice AI work for **Rushdey (رشدي)**:

- Wake-word dataset labeling and training for the custom "رشدي" wake-word model.
- Arabic speech-to-text intent routing with Vosk.
- MLflow tracking helpers for model artifacts, metrics, and reproducible experiment records.
- Production-readiness documentation for the AI checklist used in the graduation project.

## Directory Layout

```text
Voice commands/
├── app.py
├── wake_word_final.ipynb
├── requirements.txt
├── config/
│   └── intent_phrases_ar.json
├── intent_detection/
│   └── arabic_intent_router.py
├── mlflow_tracking/
│   ├── wake_word_mlflow.py
│   └── vosk_intent_mlflow.py
├── docs/
│   └── AI_PRODUCTION_CHECKLIST.md
├── templates/
│   └── index.html
└── screenshots/
```

## Components

### 1. Wake-Word Dataset Labeling

`app.py` is a Flask annotation tool for sorting raw audio into positive and negative wake-word samples.

It supports:

- Multi-user sample assignment.
- Stale assignment recovery.
- Keyboard shortcuts for fast labeling.
- Positive/negative dataset output folders.

Run it with:

```powershell
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://localhost:5000
```

### 2. Wake-Word Model Training

`wake_word_final.ipynb` is the training notebook for the from-scratch wake-word model.

High-level design:

- Input: 1-second audio windows at 16 kHz.
- Feature extraction: 64-band Mel spectrogram.
- Model: compact residual CNN classifier.
- Output: two classes, wake word vs non-wake word.
- Export target: PyTorch Lite model used by the Android app.

The mobile app loads the exported model from:

```text
rushdey/assets/models/wake_word_trial2.ptl
```

### 3. Vosk Arabic Intent Detection

`intent_detection/arabic_intent_router.py` mirrors the Android intent logic in a testable Python file.

Supported intents:

- `face_who_is_in_front`
- `currency_count`
- `ocr_read_text`
- `unknown`

Example:

```powershell
python intent_detection/arabic_intent_router.py "اقرأ المكتوب"
```

### 4. MLflow Tracking

Wake-word model tracking:

```powershell
python mlflow_tracking/wake_word_mlflow.py `
  --model-path "../rushdey/assets/models/wake_word_trial2.ptl" `
  --metrics-json "metrics/wake_word_metrics.json"
```

Vosk intent router tracking:

```powershell
python mlflow_tracking/vosk_intent_mlflow.py
```

MLflow is used for:

- Tracking parameters such as sample rate, Mel bands, duration, thresholds, and model artifact paths.
- Logging evaluation metrics.
- Saving model artifacts and intent-router configuration.
- Making experiments reproducible for thesis review.

Vosk itself is a pretrained offline ASR model, so we do not train it with MLflow. Instead, MLflow records the intent-routing layer that uses Vosk output.

## AI Production Checklist

The checklist answers are documented in:

```text
docs/AI_PRODUCTION_CHECKLIST.md
```

This covers:

- AI justification.
- Model design and limitations.
- Inputs and validation.
- Evaluation metrics.
- Testing strategy.
- Reliability and fallback handling.
- Safety, privacy, and monitoring.
- Explainability and improvement strategy.

## Mobile Integration

The Android/Flutter app uses:

- `WakeWordModelEngine.kt` for wake-word inference.
- `VoskIntentEngine.kt` for Arabic STT and intent routing.
- Camera dispatch after intent detection:
  - Face recognition.
  - Currency detection.
  - OCR reading.

The app directory is:

```text
rushdey/
```

## Notes

- Keep raw datasets out of Git unless they are small curated examples.
- Keep exported production models under `rushdey/assets/models/`.
- Use MLflow runs to preserve the evidence for each model version.
- For thesis demos, test in this order: wake word, Vosk intent, model dispatch, then result speech output.
