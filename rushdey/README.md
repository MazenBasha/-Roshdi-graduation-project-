# Rushdey Mobile Application

Rushdey (رشدي) is the Android/Flutter mobile application for the graduation project. It is designed as an assistive app for blind and visually impaired users, using offline AI components where possible.

## Core Flow

1. The wake-word engine listens for "رشدي".
2. After wake detection, Vosk Arabic STT listens for the user's intent.
3. The app maps the intent to one of the vision models.
4. The camera captures an image.
5. The selected model runs on-device.
6. Arabic text-to-speech speaks the result.

## AI Components

- Wake word: custom from-scratch PyTorch Lite model in `assets/models/wake_word_trial2.ptl`.
- Intent detection: Vosk Arabic offline ASR plus `VoskIntentEngine.kt`.
- Face recognition: TFLite embedding model plus local enrollment templates.
- Currency detection: PyTorch Lite model for Egyptian currency.
- OCR: offline Arabic OCR using bundled Arabic trained data.

## Important Android Files

```text
android/app/src/main/kotlin/com/example/rushdey/
├── MainActivity.kt
├── WakeWordModelEngine.kt
├── WakeWordService.kt
├── VoskIntentEngine.kt
├── FaceEngine.kt
├── CurrencyEngine.kt
├── OCREngine.kt
├── MelSpectrogram.kt
└── StaticImageTestReceiver.kt
```

## Model Assets

```text
assets/models/
├── wake_word_trial2.ptl
├── face_model.tflite
├── currency_model.ptl
├── best.ptl
├── labels.txt
└── config.json
```

OCR assets:

```text
assets/tessdata/ara.traineddata
```

## Build

```powershell
flutter pub get
flutter build apk --debug
```

Expected debug APK path:

```text
build/app/outputs/flutter-apk/app-debug.apk
```

## Static Image Testing

The Android project includes `StaticImageTestReceiver.kt` for repeatable local testing of face, currency, and OCR behavior without needing a live camera capture every time.

## Notes for Thesis Demo

- Keep the device microphone permission enabled.
- Test wake word in a quiet room first, then with mild background noise.
- Enroll a face using several clear captures.
- For OCR, use well-lit printed Arabic text.
- For currency, keep the note flat and fill most of the camera frame.
