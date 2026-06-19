// On-device YOLO11n object detector for the Roshdi Flutter app.
//
// Mirrors the existing Egyptian-currency module's pattern: load the .tflite
// asset once, run inference per frame, return structured detections.
//
// Pubspec deps (add if missing):
//   tflite_flutter: ^0.10.4
//   image: ^4.1.7
//
// Asset:
//   assets:
//     - assets/models/yolo11n_coco_int8.tflite
//
// Usage:
//   final det = await ObjectDetector.load();
//   final results = await det.detect(imageBytes);
//   for (final r in results) {
//     debugPrint('${r.label} ${r.confidence.toStringAsFixed(2)} ${r.position}');
//   }

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class Detection {
  final String label;
  final double confidence;
  final double x1, y1, x2, y2; // pixel coords of the original image
  final String horizontalPosition; // "left" | "center" | "right"
  final String distanceHint;       // "near" | "medium" | "far"

  Detection({
    required this.label,
    required this.confidence,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.horizontalPosition,
    required this.distanceHint,
  });

  String speak() {
    // The voice layer formats this for the user.
    return '$label, $distanceHint, on the $horizontalPosition';
  }
}

class ObjectDetector {
  static const _modelAsset = 'assets/models/yolo11n_coco_int8.tflite';
  static const _labelsAsset = 'assets/models/coco_labels.txt';
  static const int _inputSize = 640;
  static const double _confThreshold = 0.35;

  final Interpreter _interpreter;
  final List<String> _labels;

  ObjectDetector._(this._interpreter, this._labels);

  static Future<ObjectDetector> load() async {
    final interpreter = await Interpreter.fromAsset(
      _modelAsset,
      options: InterpreterOptions()..threads = Platform.numberOfProcessors,
    );
    final labelsRaw = await rootBundle.loadString(_labelsAsset);
    final labels = labelsRaw
        .split('\n')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
    return ObjectDetector._(interpreter, labels);
  }

  Future<List<Detection>> detect(Uint8List jpegOrPng) async {
    final decoded = img.decodeImage(jpegOrPng);
    if (decoded == null) return const [];

    final resized = img.copyResize(
      decoded,
      width: _inputSize,
      height: _inputSize,
      interpolation: img.Interpolation.linear,
    );

    // INT8 model — feed uint8 [1, 640, 640, 3]
    final input = Uint8List(1 * _inputSize * _inputSize * 3);
    var i = 0;
    for (var y = 0; y < _inputSize; y++) {
      for (var x = 0; x < _inputSize; x++) {
        final p = resized.getPixel(x, y);
        input[i++] = p.r.toInt();
        input[i++] = p.g.toInt();
        input[i++] = p.b.toInt();
      }
    }

    // Output shape depends on whether NMS was embedded at export. With
    // `nms=True` the head is [1, max_det, 6] = (x1, y1, x2, y2, conf, cls).
    final outShape = _interpreter.getOutputTensor(0).shape;
    final maxDet = outShape[1];
    final outBuffer = List.filled(maxDet * 6, 0.0)
        .reshape([1, maxDet, 6]);

    _interpreter.run(input.reshape([1, _inputSize, _inputSize, 3]), outBuffer);

    final results = <Detection>[];
    final scaleX = decoded.width / _inputSize;
    final scaleY = decoded.height / _inputSize;
    for (var d = 0; d < maxDet; d++) {
      final row = outBuffer[0][d] as List;
      final conf = row[4] as double;
      if (conf < _confThreshold) continue;
      final cls = (row[5] as double).toInt();
      if (cls < 0 || cls >= _labels.length) continue;

      final x1 = (row[0] as double) * scaleX;
      final y1 = (row[1] as double) * scaleY;
      final x2 = (row[2] as double) * scaleX;
      final y2 = (row[3] as double) * scaleY;
      results.add(Detection(
        label: _labels[cls],
        confidence: conf,
        x1: x1, y1: y1, x2: x2, y2: y2,
        horizontalPosition: _horizontal(decoded.width, (x1 + x2) / 2),
        distanceHint: _distance(decoded.width * decoded.height, (x2 - x1) * (y2 - y1)),
      ));
    }
    return results;
  }

  void close() => _interpreter.close();

  static String _horizontal(int imgW, double cx) {
    if (cx < imgW / 3) return 'left';
    if (cx > 2 * imgW / 3) return 'right';
    return 'center';
  }

  static String _distance(int imgArea, double boxArea) {
    final r = boxArea / imgArea;
    if (r > 0.40) return 'near';
    if (r > 0.10) return 'medium';
    return 'far';
  }
}
