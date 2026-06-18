# AI Component Production Checklist

This document maps Rushdey's voice AI components to the production-grade checklist required for the graduation project.

## 0. AI Usage Justification

Rushdey uses AI because blind users need natural Arabic interaction in noisy real-world situations. A rule-based button flow would require the user to navigate the screen, which defeats the accessibility goal. Removing AI would remove hands-free wake-up, spoken intent selection, and robust perception tasks.

## 1. Model Understanding

Wake word: a from-scratch residual CNN binary classifier trained on Mel-spectrograms for "رشدي" vs non-wake-word audio.

Intent detection: Vosk Arabic offline ASR converts speech to text, then a deterministic Arabic intent router maps text to face, currency, OCR, or unknown.

Limitations: accents, low microphone volume, heavy noise, unseen command wording, and false wake activations can reduce performance.

## 2. Data & Inputs

Wake word input is 1-second 16 kHz audio. Vosk input is microphone speech audio. Inputs are normalized, converted to Mel-spectrograms for the wake model, and Arabic-normalized for intent routing.

Noisy or unexpected inputs are rejected through confidence thresholds, RMS checks, timeouts, and `unknown` intent handling.

## 3. Evaluation & Metrics

Wake word is evaluated with accuracy, precision, recall, F1-score, confusion matrix, and false-positive behavior on negative audio.

Intent routing is evaluated with static Arabic command cases and accuracy over expected intent labels.

Baseline: no wake-word model and no intent router means the user must manually select every function.

Good enough: high recall for "رشدي" while keeping false activations low enough that the app does not open the camera randomly.

## 4. Testing AI Behavior

Tests include positive wake samples, negative silence/noise samples, expected Arabic command phrases, blank OCR images, non-currency images, and unknown face examples.

Outputs are deterministic after model inference thresholds. Vosk transcription can vary with audio quality, so the router supports phrase normalization and fuzzy matching.

## 5. Reliability & Failure Handling

Wrong output: the app speaks the result and can be retried by saying the wake word again.

Timeout: Vosk emits an intent timeout and stops listening.

Failure: engines return Arabic error messages and avoid blocking the app.

Fallbacks: unknown intent, no clear OCR text, no face detected, and no currency detected messages.

## 6. Safety & Governance

The system runs offline on device for wake word, Vosk intent, OCR, face recognition, and currency detection. No cloud API is required for private speech or camera data.

PII risk exists in face recognition because names and embeddings are personal. The app stores enrolled templates locally and should not upload them.

## 7. Prompt / Model Design

No LLM prompt is used. The intent layer is deterministic and documented in `config/intent_phrases_ar.json`.

## 8. System Integration

AI sits in the Android runtime:

1. Wake-word model listens continuously.
2. Vosk listens after wake detection.
3. Intent router selects the target model.
4. Camera capture is sent to face, currency, or OCR.
5. Text-to-speech speaks the Arabic result.

The wake loop is asynchronous. Image inference runs off the UI thread to keep the app responsive.

## 9. Performance & Cost

The system is offline, so there is no token or API cost per request. Runtime cost is device CPU and memory.

Large assets such as Vosk increase APK/storage size, but they protect privacy and allow the app to work without internet.

## 10. Monitoring AI in Production

The app logs confidence, microphone level, Vosk status, intent result, and model errors. MLflow scripts track offline model versions, parameters, metrics, and artifacts before release.

Bad decisions can be traced through logged confidence, recognized text, selected intent, and model result.

## 11. Explainability

Wake word exposes confidence. Intent routing can show the recognized text and matched intent. Face/currency/OCR responses include the selected label or a clear failure message.

## 12. Improvement Strategy

Wake word improves by collecting more positive/negative samples, retraining, and logging experiments with MLflow.

Intent routing improves by adding real Vosk transcripts that failed in testing to `intent_phrases_ar.json`.

Model drift is handled by retesting with new samples and comparing MLflow metrics between versions.

## 13. Ethical & Responsible AI

Bias risk: wake-word and intent recognition may work better for voices and accents represented in the dataset. Face recognition can also vary across lighting, pose, and demographics.

Privacy: camera and microphone processing should stay on-device. Face templates should remain local.

If the AI gives a wrong or harmful result, the app should avoid overclaiming certainty, speak clear uncertainty messages, and allow immediate retry.
