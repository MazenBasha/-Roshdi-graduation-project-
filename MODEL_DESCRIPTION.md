# Face Recognition Model — Description & Flutter Integration

## 1. Model Overview

The project trains a **face embedding model** (not a classifier). Given a
face image, it produces a 512-dimensional vector that uniquely represents
that person's identity. Two faces of the same person produce vectors that
are *close* (high cosine similarity); two different people produce vectors
that are *far apart*.

### Architecture

- **Backbone:** iResNet (Improved ResNet) — variants `iresnet18`,
  `iresnet34`, `iresnet50`, `iresnet100`. Default for production:
  `iresnet50`. The currently trained checkpoints in `checkpoints/` use
  `iresnet18` (faster, mobile-friendly).
- **Loss:** Two-stage training in `train.py`:
  1. Stage 1 — `CosineHead` warmup with plain cross-entropy on scaled
     cosine logits.
  2. Stage 2 — `ArcFaceHead` with additive angular margin
     (m = 0.5, s = 64), based on Deng et al. (CVPR 2019).
- **Pretrained fallback:** If no trained checkpoint is on disk,
  `inference.py` falls back to `facenet_vggface2`
  (InceptionResnetV1 from `facenet-pytorch`), so the system works
  out-of-the-box.

### Files
- `model.py` — backbone + heads
- `train.py` — two-stage training loop
- `inference.py` — end-to-end recognition pipeline
- `quantize.py` — INT8 / ONNX export for edge deployment
- `config.py` — central hyperparameters

---

## 2. Inputs

### Training input (`train.py`)
- A face dataset directory or MXNet RecordIO (e.g., CASIA-WebFace, MS1MV2).
- Images are pre-aligned to **112 × 112 RGB** using 5-point landmarks
  and ArcFace's canonical template (`utils.align_face`).
- Each image carries an integer class label (person ID).

### Inference input (`FaceRecognizer.recognize_image`)
- A single BGR image as a `numpy.ndarray` of shape `(H, W, 3)` —
  the raw output of `cv2.imread` or a webcam frame.
- The pipeline internally:
  1. Runs MTCNN to detect faces and 5 landmarks.
  2. Aligns each face to a 112×112 BGR crop.
  3. Applies CLAHE (optional) and normalizes to `[-1, 1]`.
  4. Feeds the batch tensor `(N, 3, 112, 112)` to the backbone.

### Model tensor input (raw)
- **Shape:** `(B, 3, 112, 112)`
- **Dtype:** `float32`
- **Range:** `[-1, 1]` (i.e., `(pixel/255 - 0.5) / 0.5`)
- **Channel order:** RGB

---

## 3. Outputs

### Model raw output
- **Shape:** `(B, 512)`
- **Dtype:** `float32`
- **Semantics:** face embedding. After L2 normalization, two
  embeddings can be compared with cosine similarity (a simple dot
  product on unit vectors).

### `FaceRecognizer.recognize_image(image)` — Python API
Returns a list of `FaceResult` (one per detected face):

```python
FaceResult(
  bbox: np.ndarray,         # [x1, y1, x2, y2]
  landmarks: np.ndarray,    # (5, 2)
  detection_score: float,   # MTCNN confidence
  embedding: np.ndarray,    # (512,) L2-normalized
  name: str | None,         # None means "unknown"
  similarity: float,        # best cosine vs the face DB
  aligned_face: np.ndarray, # 112x112 BGR crop
)
```

### `FaceRecognizer.recognize_image_dicts(image)` — JSON-friendly
Returns a list of dicts (the shape used by the multi-person API):

```json
[
  {
    "name": "Ali",
    "confidence": 0.823,
    "bbox": [120, 88, 254, 260],
    "detection_score": 0.998
  },
  {
    "name": "Unknown",
    "confidence": 0.21,
    "bbox": [310, 95, 412, 240],
    "detection_score": 0.991
  }
]
```

A face is labeled `"Unknown"` when its best cosine similarity to the
face DB is below `MATCH_THRESHOLD` (default 0.40 in `config.py`).

---

## 4. Linking the Model with Flutter

There are three deployment patterns. Pick one based on whether the
phone has internet and how much compute you want on-device.

### Option A — Python HTTP/WebSocket server (recommended for Roshdi)

Wrap `FaceRecognizer` in a small FastAPI / Flask server and let the
Flutter app upload frames over the network. This is the simplest
and most accurate path, since the full PyTorch model runs on the
server (laptop, Jetson, cloud).

**Server side (`server.py`, sketch):**

```python
from fastapi import FastAPI, UploadFile
import cv2, numpy as np
from inference import FaceRecognizer

app = FastAPI()
rec = FaceRecognizer()  # loads checkpoint + face DB once at startup

@app.post("/recognize")
async def recognize(file: UploadFile):
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return {"faces": rec.recognize_image_dicts(img)}

@app.post("/enroll")
async def enroll(name: str, file: UploadFile):
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    n = rec.enroll_from_image(img, name)
    return {"enrolled": n}
```

Run with: `uvicorn server:app --host 0.0.0.0 --port 8000`

**Flutter side:**

```dart
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'dart:convert';

Future<List<dynamic>> recognize(String imagePath) async {
  final uri = Uri.parse('http://<server-ip>:8000/recognize');
  final req = http.MultipartRequest('POST', uri)
    ..files.add(await http.MultipartFile.fromPath('file', imagePath));
  final res = await req.send();
  final body = await res.stream.bytesToString();
  return jsonDecode(body)['faces'];
}
```

Each item in the returned list has `name`, `confidence`, `bbox`,
`detection_score` — draw the boxes on a `CustomPaint` over the camera
preview, and speak the names via `flutter_tts` for the assistive use
case.

For real-time camera streams, replace the multipart upload with a
WebSocket that sends JPEG-encoded frames at ~5–10 FPS.

### Option B — On-device ONNX (offline, mobile)

`quantize.py` already exports an ONNX file (see
`checkpoints/iresnet18.onnx`). Run it on the phone via the
`onnxruntime` Flutter binding.

1. Export the embedding model:
   ```
   python quantize.py \
     --checkpoint checkpoints/two_stage_v2_iresnet18.best.pt \
     --backbone iresnet18 \
     --out-onnx checkpoints/iresnet18.onnx
   ```
2. Bundle `iresnet18.onnx` as a Flutter asset.
3. In `pubspec.yaml`:
   ```yaml
   dependencies:
     onnxruntime: ^1.4.1
     google_mlkit_face_detection: ^0.10.0   # for detection + landmarks
     image: ^4.0.0
   flutter:
     assets:
       - assets/iresnet18.onnx
   ```
4. In Dart:
   - Use `google_mlkit_face_detection` to get bounding boxes and
     landmarks.
   - Crop and warp each face to 112×112 (the same alignment template
     as `utils.align_face`).
   - Convert to a `Float32List` of shape `[1, 3, 112, 112]`,
     normalized to `[-1, 1]`, channel order RGB.
   - Run the ONNX session — output is `[1, 512]`.
   - L2-normalize, then cosine-compare against your stored embeddings
     (a JSON file in app storage).

### Option C — TensorFlow Lite (smallest binary, Android-friendly)

Convert the ONNX file to TFLite via `onnx-tf` or `onnx2tf` and run
through the `tflite_flutter` package. The model I/O contract is
identical (input `[1, 3, 112, 112]` float32, output `[1, 512]`
float32).

---

## 5. Quick Reference (cheat sheet)

| Item | Value |
|---|---|
| Backbone | iResNet50 (prod) / iResNet18 (mobile) |
| Loss | ArcFace (s=64, m=0.5) |
| Input shape | `(B, 3, 112, 112)` float32, RGB, range `[-1, 1]` |
| Output shape | `(B, 512)` float32 embedding |
| Compare with | Cosine similarity (dot product after L2 norm) |
| Match threshold | 0.40 (configurable in `config.py`) |
| Detection | MTCNN, min face 40 px, conf ≥ 0.90 |
| Mobile export | `quantize.py` → ONNX or INT8 |
| Flutter recommended | HTTP/WebSocket to Python server (Option A) |
