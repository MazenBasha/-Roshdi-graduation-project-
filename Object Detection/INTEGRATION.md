# Integrating into the Roshdi App

This module exposes **two integration paths**. The Roshdi app already runs
on-device models via `pytorch_lite` / TFLite (currency module), so path A is
the default. No training is required for either path — both consume the
**pretrained YOLO11n** as-is.

## A — On-device TFLite (recommended, offline, free)

1. Generate the mobile artefacts once from the pretrained checkpoint:
   ```bash
   pip install tensorflow coremltools          # heavy export deps
   python export.py --formats tflite coreml --int8 --data coco128.yaml --nms
   ```
   This produces `yolo11n_int8.tflite` and `yolo11n.mlpackage` in the project
   root. `coco128.yaml` (auto-downloaded by Ultralytics) supplies the INT8
   calibration images — no Roboflow account or custom dataset needed.

2. Drop the artefacts into the Roshdi app:
   ```
   rushdey/
     assets/
       models/
         yolo11n_coco_int8.tflite      <-- copy yolo11n_int8.tflite
         coco_labels.txt               <-- copy flutter_integration/coco_labels.txt
   ```

3. Add them to `pubspec.yaml`:
   ```yaml
   flutter:
     assets:
       - assets/models/yolo11n_coco_int8.tflite
       - assets/models/coco_labels.txt
   dependencies:
     tflite_flutter: ^0.10.4
     image: ^4.1.7
   ```

4. Copy `flutter_integration/object_detector.dart` into `rushdey/lib/services/`.

5. Wire into the camera/voice pipeline alongside the currency detector:
   ```dart
   final detector = await ObjectDetector.load();
   cameraController.startImageStream((frame) async {
     final dets = await detector.detect(frame.bytes);
     for (final d in dets.where((d) => d.confidence > 0.4)) {
       voice.speak(d.speak());          // "chair, near, on the left"
     }
   });
   ```

The `pytorch_lite` plugin can also load the YOLO TorchScript export if you
prefer a single runtime across modules — `python export.py --formats
torchscript` produces it. The Dart call site is almost identical; swap
`tflite_flutter` for `pytorch_lite` and the asset path for the `.pt` file.

## B — Cloud HTTP fallback (online, optional)

Useful for: device debugging, batch evaluation of new test images, or
fallback when the on-device model fails to load.

1. Deploy `docker compose up -d` on any small VM (1 vCPU / 2 GB suffices for
   YOLO11n; ~$5/mo on Hetzner / DigitalOcean). The Docker image ships with
   `yolo11n.pt` baked in, so the container is fully offline.
2. From Flutter:
   ```dart
   final r = await http.post(
     Uri.parse('https://od.roshdi.app/v1/detect'),
     headers: {'X-API-Key': apiKey, 'X-Request-ID': uuid()},
     body: http.MultipartFile.fromBytes('image', jpegBytes, filename: 'f.jpg'),
   );
   ```
3. Response matches the `DetectResponse` schema in `server/schemas.py` — same
   field names as the on-device Dart class, so the downstream voice layer is
   unchanged.

## Custom fine-tuning (optional, future)

If accuracy on Egyptian street scenes proves unsatisfactory after launch:

1. Collect ~500–2000 frames from real Roshdi sessions (with user consent).
2. Auto-label with the current model, then human-correct in Roboflow.
3. Run `kaggle_pipeline.ipynb` against the new export.
4. The retrained `best.pt` is a drop-in replacement: set
   `ROSHDI_OD_WEIGHTS_PATH=/app/weights/best.pt` on the server, or replace
   the TFLite asset in the Flutter bundle. **No Flutter code changes** —
   the wire format is identical.
