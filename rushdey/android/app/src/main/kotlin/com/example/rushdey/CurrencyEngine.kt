package com.example.rushdey

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor

/**
 * CurrencyEngine — YOLOv8 Egyptian Currency Detection using native PyTorch Android Lite.
 *
 * Bypasses the pytorch_lite Flutter plugin to feed the model the exact 640×640
 * input it was traced with, avoiding the anchor grid size mismatch.
 */
class CurrencyEngine(private val context: Context) {

    private var model: Module? = null
    private var isReady = false

    // Must match the order in labels.txt
    private val labels = listOf("1_EGP", "5_EGP", "10_EGP", "20_EGP", "50_EGP", "100_EGP", "200_EGP")
    private val values = mapOf(
        "1_EGP" to 1, "5_EGP" to 5, "10_EGP" to 10,
        "20_EGP" to 20, "50_EGP" to 50, "100_EGP" to 100, "200_EGP" to 200
    )

    private val inputSize = 640
    private val confThreshold = 0.35f
    private val iouThreshold = 0.5f

    data class Detection(
        val className: String,
        val confidence: Float,
        val value: Int,
        val box: FloatArray // x1, y1, x2, y2 normalized
    )

    data class CurrencyResult(
        val detections: List<Detection>,
        val total: Int,
        val arabicName: String
    )

    fun initialize(): Boolean {
        return try {
            val modelFile = copyAssetToFile("best.ptl")
            model = Module.load(modelFile.absolutePath)
            isReady = true
            Log.i(TAG, "CurrencyEngine loaded successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load currency model", e)
            false
        }
    }

    fun detect(imageBytes: ByteArray): CurrencyResult {
        if (!isReady || model == null) {
            return CurrencyResult(emptyList(), 0, "محرك العملة غير جاهز")
        }

        try {
            // Decode and resize to 640×640
            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                ?: return CurrencyResult(emptyList(), 0, "خطأ في قراءة الصورة")

            val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            if (resized !== bitmap) bitmap.recycle()

            // Convert to float tensor [1, 3, 640, 640] normalized to [0, 1]
            val inputTensor = bitmapToTensor(resized)
            resized.recycle()

            // Run inference
            val localModel = model ?: return CurrencyResult(emptyList(), 0, "محرك العملة غير جاهز")
            val output = localModel.forward(IValue.from(inputTensor))
            val outputTensor = output.toTensor()
            val outputData = outputTensor.dataAsFloatArray
            val outputShape = outputTensor.shape() // Expected: [1, numClasses+4, numAnchors]

            // Parse YOLOv8 output
            val numClasses = labels.size
            val numAnchors = outputShape[2].toInt() // 8400 for 640×640
            // val numFeatures = outputShape[1].toInt() // 4 + numClasses

            val detections = mutableListOf<Detection>()

            for (i in 0 until numAnchors) {
                // YOLOv8 output format: [1, 4+numClasses, numAnchors]
                // First 4 values: cx, cy, w, h
                val cx = outputData[0 * numAnchors + i]
                val cy = outputData[1 * numAnchors + i]
                val w  = outputData[2 * numAnchors + i]
                val h  = outputData[3 * numAnchors + i]

                // Find best class
                var bestClassIdx = 0
                var bestScore = 0f
                for (c in 0 until numClasses) {
                    val score = outputData[(4 + c) * numAnchors + i]
                    if (score > bestScore) {
                        bestScore = score
                        bestClassIdx = c
                    }
                }

                if (bestScore >= confThreshold) {
                    val x1 = ((cx - w / 2f) / inputSize).coerceIn(0f, 1f)
                    val y1 = ((cy - h / 2f) / inputSize).coerceIn(0f, 1f)
                    val x2 = ((cx + w / 2f) / inputSize).coerceIn(0f, 1f)
                    val y2 = ((cy + h / 2f) / inputSize).coerceIn(0f, 1f)

                    val label = labels[bestClassIdx]
                    detections.add(Detection(
                        className = label,
                        confidence = bestScore,
                        value = values[label] ?: 0,
                        box = floatArrayOf(x1, y1, x2, y2)
                    ))
                }
            }

            // NMS
            val nmsDetections = nms(detections, iouThreshold)
            val total = nmsDetections.sumOf { it.value }

            val arabicName = if (total > 0) {
                "$total جنيه"
            } else {
                "مش شايف عملة واضحة"
            }

            Log.i(TAG, "Detected ${nmsDetections.size} currencies, total=$total EGP")
            return CurrencyResult(nmsDetections, total, arabicName)

        } catch (e: Exception) {
            Log.e(TAG, "Currency detection failed", e)
            return CurrencyResult(emptyList(), 0, "حصل خطأ في نموذج العملة")
        }
    }

    private fun bitmapToTensor(bitmap: Bitmap): Tensor {
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val floatArray = FloatArray(3 * inputSize * inputSize)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            // RGB channels normalized to [0, 1]
            floatArray[i] = ((pixel shr 16) and 0xFF) / 255.0f                         // R
            floatArray[inputSize * inputSize + i] = ((pixel shr 8) and 0xFF) / 255.0f   // G
            floatArray[2 * inputSize * inputSize + i] = (pixel and 0xFF) / 255.0f        // B
        }

        return Tensor.fromBlob(floatArray, longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()))
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
        if (outFile.exists()) return outFile

        val assetPath = "flutter_assets/assets/models/$assetName"
        context.assets.open(assetPath).use { input ->
            java.io.FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
        return outFile
    }

    fun release() {
        model?.destroy()
        model = null
        isReady = false
    }

    companion object {
        private const val TAG = "CurrencyEngine"
    }
}
