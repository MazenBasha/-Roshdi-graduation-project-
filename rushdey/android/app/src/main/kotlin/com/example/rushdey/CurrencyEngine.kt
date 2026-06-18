package com.example.rushdey

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import kotlin.math.exp

/**
 * On-device Egyptian currency recognition.
 *
 * The YOLO detector is tried first for scene/multi-note images. If it finds no
 * note, the full-note classifier is used as a fallback. The exported detector
 * graph was traced at 320x320; feeding 640x640 causes the anchor mismatch crash.
 */
class CurrencyEngine(private val context: Context) {

    private var detectorModel: Module? = null
    private var classifierModel: Module? = null
    private var isReady = false

    private val detectorLabels = listOf("1_EGP", "5_EGP", "10_EGP", "20_EGP", "50_EGP", "100_EGP", "200_EGP")
    private val classifierLabels = listOf(
        "1_EGP", "5_EGP", "10_EGP", "10_EGP_NEW", "20_EGP", "20_EGP_NEW", "50_EGP", "100_EGP", "200_EGP"
    )
    private val values = mapOf(
        "1_EGP" to 1,
        "5_EGP" to 5,
        "10_EGP" to 10,
        "10_EGP_NEW" to 10,
        "20_EGP" to 20,
        "20_EGP_NEW" to 20,
        "50_EGP" to 50,
        "100_EGP" to 100,
        "200_EGP" to 200
    )

    private val detectorInputSize = 320
    private val classifierInputSize = 224
    private val detectorConfidenceThreshold = 0.35f
    private val classifierConfidenceThreshold = 0.60f
    private val iouThreshold = 0.5f

    data class Detection(
        val className: String,
        val confidence: Float,
        val value: Int,
        val box: FloatArray
    )

    data class CurrencyResult(
        val detections: List<Detection>,
        val total: Int,
        val arabicName: String
    )

    fun initialize(): Boolean {
        detectorModel = try {
            Module.load(copyAssetToFile("best.ptl").absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load currency detector", e)
            null
        }

        classifierModel = try {
            Module.load(copyAssetToFile("currency_model.ptl").absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load currency classifier", e)
            null
        }

        isReady = detectorModel != null || classifierModel != null
        Log.i(TAG, "CurrencyEngine loaded detector=${detectorModel != null} classifier=${classifierModel != null}")
        return isReady
    }

    fun detect(imageBytes: ByteArray): CurrencyResult {
        if (!isReady) {
            return CurrencyResult(emptyList(), 0, "\u0645\u062d\u0631\u0643 \u0627\u0644\u0639\u0645\u0644\u0629 \u063a\u064a\u0631 \u062c\u0627\u0647\u0632")
        }

        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            ?: return CurrencyResult(emptyList(), 0, "\u062e\u0637\u0623 \u0641\u064a \u0642\u0631\u0627\u0621\u0629 \u0627\u0644\u0635\u0648\u0631\u0629")

        return try {
            val detectorDetections = try {
                runDetector(bitmap)
            } catch (e: Exception) {
                Log.e(TAG, "Currency detector failed", e)
                emptyList()
            }

            if (detectorDetections.isNotEmpty()) {
                return resultFromDetections(detectorDetections, "detector")
            }

            val classifierDetection = try {
                runClassifier(bitmap)
            } catch (e: Exception) {
                Log.e(TAG, "Currency classifier failed", e)
                null
            }

            if (classifierDetection != null) {
                resultFromDetections(listOf(classifierDetection), "classifier")
            } else {
                CurrencyResult(emptyList(), 0, noClearCurrencyText())
            }
        } finally {
            bitmap.recycle()
        }
    }

    private fun runDetector(bitmap: Bitmap): List<Detection> {
        val localModel = detectorModel ?: return emptyList()
        val resized = Bitmap.createScaledBitmap(bitmap, detectorInputSize, detectorInputSize, true)
        try {
            val outputTensor = localModel.forward(IValue.from(bitmapToDetectorTensor(resized))).toTensor()
            val outputData = outputTensor.dataAsFloatArray
            val outputShape = outputTensor.shape()
            if (outputShape.size < 3 || outputShape[1].toInt() < 4 + detectorLabels.size) {
                Log.w(TAG, "Unexpected detector output shape=${outputShape.joinToString()}")
                return emptyList()
            }

            val numClasses = detectorLabels.size
            val numAnchors = outputShape[2].toInt()
            val detections = mutableListOf<Detection>()

            for (i in 0 until numAnchors) {
                val cx = outputData[i]
                val cy = outputData[numAnchors + i]
                val w = outputData[2 * numAnchors + i]
                val h = outputData[3 * numAnchors + i]

                var bestClassIdx = 0
                var bestScore = 0f
                for (c in 0 until numClasses) {
                    val score = outputData[(4 + c) * numAnchors + i]
                    if (score > bestScore) {
                        bestScore = score
                        bestClassIdx = c
                    }
                }

                if (bestScore >= detectorConfidenceThreshold) {
                    val x1 = ((cx - w / 2f) / detectorInputSize).coerceIn(0f, 1f)
                    val y1 = ((cy - h / 2f) / detectorInputSize).coerceIn(0f, 1f)
                    val x2 = ((cx + w / 2f) / detectorInputSize).coerceIn(0f, 1f)
                    val y2 = ((cy + h / 2f) / detectorInputSize).coerceIn(0f, 1f)
                    val label = detectorLabels[bestClassIdx]
                    detections.add(
                        Detection(
                            className = label,
                            confidence = bestScore,
                            value = values[label] ?: 0,
                            box = floatArrayOf(x1, y1, x2, y2)
                        )
                    )
                }
            }

            return nms(detections, iouThreshold)
        } finally {
            resized.recycle()
        }
    }

    private fun runClassifier(bitmap: Bitmap): Detection? {
        val localModel = classifierModel ?: return null
        val resized = Bitmap.createScaledBitmap(bitmap, classifierInputSize, classifierInputSize, true)
        try {
            val outputTensor = localModel.forward(IValue.from(bitmapToClassifierTensor(resized))).toTensor()
            val logits = outputTensor.dataAsFloatArray
            if (logits.size < classifierLabels.size) return null

            val (bestIndex, confidence) = bestSoftmaxClass(logits, classifierLabels.size)
            val label = classifierLabels[bestIndex]
            val value = values[label] ?: 0
            Log.i(TAG, "Classifier currency=$label confidence=$confidence")

            if (confidence < classifierConfidenceThreshold || value <= 0) return null

            return Detection(
                className = label,
                confidence = confidence,
                value = value,
                box = floatArrayOf(0f, 0f, 1f, 1f)
            )
        } finally {
            resized.recycle()
        }
    }

    private fun bestSoftmaxClass(logits: FloatArray, count: Int): Pair<Int, Float> {
        var maxLogit = Float.NEGATIVE_INFINITY
        for (i in 0 until count) maxLogit = maxOf(maxLogit, logits[i])

        var bestIndex = 0
        var bestExp = 0.0
        var sumExp = 0.0
        for (i in 0 until count) {
            val value = exp((logits[i] - maxLogit).toDouble())
            sumExp += value
            if (value > bestExp) {
                bestExp = value
                bestIndex = i
            }
        }

        return bestIndex to (bestExp / sumExp).toFloat()
    }

    private fun resultFromDetections(detections: List<Detection>, source: String): CurrencyResult {
        val total = detections.sumOf { it.value }
        Log.i(TAG, "Detected ${detections.size} currencies from $source, total=$total EGP")
        return CurrencyResult(detections, total, if (total > 0) arabicCurrencyName(total) else noClearCurrencyText())
    }

    private fun bitmapToDetectorTensor(bitmap: Bitmap): Tensor {
        val pixels = IntArray(detectorInputSize * detectorInputSize)
        bitmap.getPixels(pixels, 0, detectorInputSize, 0, 0, detectorInputSize, detectorInputSize)

        val floatArray = FloatArray(3 * detectorInputSize * detectorInputSize)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            floatArray[i] = ((pixel shr 16) and 0xFF) / 255.0f
            floatArray[detectorInputSize * detectorInputSize + i] = ((pixel shr 8) and 0xFF) / 255.0f
            floatArray[2 * detectorInputSize * detectorInputSize + i] = (pixel and 0xFF) / 255.0f
        }

        return Tensor.fromBlob(
            floatArray,
            longArrayOf(1, 3, detectorInputSize.toLong(), detectorInputSize.toLong())
        )
    }

    private fun bitmapToClassifierTensor(bitmap: Bitmap): Tensor {
        val pixels = IntArray(classifierInputSize * classifierInputSize)
        bitmap.getPixels(pixels, 0, classifierInputSize, 0, 0, classifierInputSize, classifierInputSize)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val channelSize = classifierInputSize * classifierInputSize
        val floatArray = FloatArray(3 * channelSize)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            floatArray[i] = (r - mean[0]) / std[0]
            floatArray[channelSize + i] = (g - mean[1]) / std[1]
            floatArray[2 * channelSize + i] = (b - mean[2]) / std[2]
        }

        return Tensor.fromBlob(
            floatArray,
            longArrayOf(1, 3, classifierInputSize.toLong(), classifierInputSize.toLong())
        )
    }

    private fun nms(detections: List<Detection>, threshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val result = mutableListOf<Detection>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            result.add(best)
            sorted.removeAll { iou(best.box, it.box) > threshold }
        }

        return result
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val x1 = maxOf(a[0], b[0])
        val y1 = maxOf(a[1], b[1])
        val x2 = minOf(a[2], b[2])
        val y2 = minOf(a[3], b[3])

        val intersection = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val areaA = (a[2] - a[0]) * (a[3] - a[1])
        val areaB = (b[2] - b[0]) * (b[3] - b[1])
        val union = areaA + areaB - intersection

        return if (union > 0f) intersection / union else 0f
    }

    private fun copyAssetToFile(assetName: String): java.io.File {
        val outFile = java.io.File(context.filesDir, assetName)
        if (outFile.exists() && outFile.length() > 0) return outFile

        val assetPath = "flutter_assets/assets/models/$assetName"
        context.assets.open(assetPath).use { input ->
            java.io.FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
        return outFile
    }

    private fun arabicCurrencyName(total: Int): String = "$total \u062c\u0646\u064a\u0647"

    private fun noClearCurrencyText(): String =
        "\u0645\u0634 \u0634\u0627\u064a\u0641 \u0639\u0645\u0644\u0629 \u0648\u0627\u0636\u062d\u0629"

    fun release() {
        detectorModel?.destroy()
        classifierModel?.destroy()
        detectorModel = null
        classifierModel = null
        isReady = false
    }

    companion object {
        private const val TAG = "CurrencyEngine"
    }
}
