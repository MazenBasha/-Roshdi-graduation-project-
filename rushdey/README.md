# Rushdey Mobile Application

Rushdey (رشدي) is the Android/Flutter mobile application for the graduation project. It is designed as an assistive app for blind and visually impaired users, using offline AI components where possible.

## Core Flow

1. The wake-word engine listens for "رشدي" (using `WakeWordDetector.kt` with fuzzy matching for resilience against speech-to-text variations).
2. After wake detection, Vosk Arabic STT listens for the user's intent.
3. The app maps the intent to one of the vision models using `VoskIntentEngine.kt`.
4. The camera captures an image.
5. The selected model runs on-device using native Android (Kotlin) ML engines.
6. Arabic text-to-speech speaks the result.

## AI Components

- Wake word & Intent detection: Offline Arabic ASR powered by Vosk. `WakeWordDetector.kt` robustly detects the wake word and its variants via fuzzy matching, and `VoskIntentEngine.kt` handles command routing.
- Face recognition: TFLite embedding model plus local enrollment templates (`FaceEngine.kt`).
- Currency detection: Custom native Android implementation (`CurrencyEngine.kt`) handling both YOLOv8 object detection boxes and classic PyTorch Lite classification logits directly via PyTorch Mobile, resolving Flutter-plugin tensor scaling limitations.
- Object Detection (Obstacles): YOLO-based real-time obstacle detection engine (`ObjectDetectionEngine.kt`) providing distance hints and walking assistance.
- OCR: offline Arabic OCR using bundled Arabic trained data (`OCREngine.kt`).

## Important Android Files

```text
android/app/src/main/kotlin/com/example/rushdey/
├── MainActivity.kt
├── WakeWordDetector.kt        <-- Fuzzy matcher for Arabic wake word variations
├── WakeWordModelEngine.kt
├── WakeWordService.kt
├── VoskIntentEngine.kt        <-- Maps Vosk ASR output to app commands
├── FaceEngine.kt              <-- Native TFLite face recognition
├── CurrencyEngine.kt          <-- Native PyTorch Mobile currency detection (YOLOv8 + Classifier)
├── ObjectDetectionEngine.kt   <-- Native PyTorch Mobile obstacle detection
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
