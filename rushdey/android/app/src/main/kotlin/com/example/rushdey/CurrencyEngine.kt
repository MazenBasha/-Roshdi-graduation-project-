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
 * The optional "YOLOv8" asset from the Updates branch is handled defensively:
 * some exports return detector boxes, while the checked-in mobile export returns
 * 9-class currency logits. Detector outputs are summed note-by-note; classifier
 * outputs return the single best note value and still fall back safely.
 */
class CurrencyEngine(private val context: Context) {

    private var advancedModel: Module? = null
    private var classifierModel: Module? = null
    private var isReady = false

    enum class Mode {
        CLASSIC,
        YOLO_V8
    }

    private val detectorLabels = listOf("1_EGP", "5_EGP", "10_EGP", "20_EGP", "50_EGP", "100_EGP", "200_EGP")
    private val advancedLabels = listOf(
        "1_EGP", "5_EGP", "10_EGP", "10_EGP_NEW", "20_EGP", "20_EGP_NEW", "50_EGP", "100_EGP", "200_EGP"
    )
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

    private val advancedClassifierInputSize = 224
    private val detectorInputSize = 320
    private val classifierInputSize = 224
    private val detectorConfidenceThreshold = 0.35f
    private val advancedClassifierConfidenceThreshold = 0.35f
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

    private data class RawModelOutput(
        val data: FloatArray,
        val shape: LongArray
    )

    fun initialize(): Boolean {
        advancedModel = try {
            Module.load(copyAssetToFile("best.ptl").absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load advanced currency model", e)
            null
        }

        classifierModel = try {
            Module.load(copyAssetToFile("currency_model.ptl").absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load currency classifier", e)
            null
        }

        isReady = advancedModel != null || classifierModel != null
        Log.i(TAG, "CurrencyEngine loaded advanced=${advancedModel != null} classifier=${classifierModel != null}")
        return isReady
    }

    fun detect(imageBytes: ByteArray, mode: Mode = Mode.CLASSIC): CurrencyResult {
        if (!isReady || !isModeReady(mode)) {
            return CurrencyResult(emptyList(), 0, "\u0645\u062d\u0631\u0643 \u0627\u0644\u0639\u0645\u0644\u0629 \u063a\u064a\u0631 \u062c\u0627\u0647\u0632")
        }

        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            ?: return CurrencyResult(emptyList(), 0, "\u062e\u0637\u0623 \u0641\u064a \u0642\u0631\u0627\u0621\u0629 \u0627\u0644\u0635\u0648\u0631\u0629")

        return try {
            if (mode == Mode.YOLO_V8) {
                val detectorDetections = try {
                    runDetector(bitmap)
                } catch (e: Exception) {
                    Log.e(TAG, "Currency YOLOv8 detector failed", e)
                    emptyList()
                }

                if (detectorDetections.isNotEmpty()) {
                    return resultFromDetections(detectorDetections, "yolo_v8")
                }

                Log.i(TAG, "Advanced currency model found no notes; falling back to classic classifier")
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

    private fun isModeReady(mode: Mode): Boolean {
        return when (mode) {
            Mode.CLASSIC -> classifierModel != null
            Mode.YOLO_V8 -> advancedModel != null || classifierModel != null
        }
    }

    private fun runDetector(bitmap: Bitmap): List<Detection> {
        val localModel = advancedModel ?: return emptyList()

        runAdvancedModelAtSize(localModel, bitmap, advancedClassifierInputSize, normalize = true)?.let { output ->
            decodeAdvancedOutput(output, advancedClassifierInputSize)?.let { detections ->
                if (detections.isNotEmpty()) return detections
            }
        }

        runAdvancedModelAtSize(localModel, bitmap, detectorInputSize, normalize = false)?.let { output ->
            decodeAdvancedOutput(output, detectorInputSize)?.let { detections ->
                if (detections.isNotEmpty()) return detections
            }
        }

        return emptyList()
    }

    private fun runAdvancedModelAtSize(
        model: Module,
        bitmap: Bitmap,
        inputSize: Int,
        normalize: Boolean
    ): RawModelOutput? {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        try {
            val inputTensor = if (normalize) {
                bitmapToNormalizedTensor(resized, inputSize)
            } else {
                bitmapToUnitTensor(resized, inputSize)
            }
            val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
            return RawModelOutput(outputTensor.dataAsFloatArray, outputTensor.shape())
        } catch (e: Exception) {
            Log.w(TAG, "Advanced currency model failed at ${inputSize}x$inputSize", e)
            return null
        } finally {
            resized.recycle()
        }
    }

    private fun decodeAdvancedOutput(output: RawModelOutput, inputSize: Int): List<Detection>? {
        Log.i(TAG, "Advanced currency output shape=${output.shape.joinToString()} input=$inputSize")

        decodeClassifierOutput(output.data, output.shape, advancedLabels, advancedClassifierConfidenceThreshold)?.let {
            return it
        }
        decodeYoloChannelFirstOutput(output.data, output.shape, inputSize)?.let {
            return it
        }
        decodeYoloRowOutput(output.data, output.shape, inputSize)?.let {
            return it
        }

        Log.w(TAG, "Unsupported advanced currency output shape=${output.shape.joinToString()}")
        return null
    }

    private fun decodeClassifierOutput(
        outputData: FloatArray,
        outputShape: LongArray,
        labels: List<String>,
        threshold: Float
    ): List<Detection>? {
        val isFlatLogits = outputShape.size == 1 && outputData.size >= labels.size
        val isBatchedLogits = outputShape.size == 2 &&
            outputShape[0].toInt() == 1 &&
            outputShape[1].toInt() >= labels.size
        if (!isFlatLogits && !isBatchedLogits) return null

        val (bestIndex, confidence) = bestSoftmaxClass(outputData, labels.size)
        val label = labels[bestIndex]
        val value = values[label] ?: 0
        Log.i(TAG, "Advanced classifier currency=$label confidence=$confidence")

        if (confidence < threshold || value <= 0) return emptyList()

        return listOf(
            Detection(
                className = label,
                confidence = confidence,
                value = value,
                box = floatArrayOf(0f, 0f, 1f, 1f)
            )
        )
    }

    private fun decodeYoloChannelFirstOutput(
        outputData: FloatArray,
        outputShape: LongArray,
        inputSize: Int
    ): List<Detection>? {
        if (outputShape.size < 3) return null

        val channels = outputShape[1].toInt()
        val numAnchors = outputShape[2].toInt()
        val labels = labelsForYoloChannels(channels) ?: return null
        val detections = mutableListOf<Detection>()

        for (i in 0 until numAnchors) {
            val cx = outputData[i]
            val cy = outputData[numAnchors + i]
            val w = outputData[2 * numAnchors + i]
            val h = outputData[3 * numAnchors + i]

            var bestClassIdx = 0
            var bestScore = 0f
            for (c in labels.indices) {
                val score = outputData[(4 + c) * numAnchors + i]
                if (score > bestScore) {
                    bestScore = score
                    bestClassIdx = c
                }
            }

            if (bestScore >= detectorConfidenceThreshold) {
                val label = labels[bestClassIdx]
                detections.add(
                    Detection(
                        className = label,
                        confidence = bestScore,
                        value = values[label] ?: 0,
                        box = centerBoxToNormalizedBox(cx, cy, w, h, inputSize)
                    )
                )
            }
        }

        return classAwareNms(detections, iouThreshold)
    }

    private fun decodeYoloRowOutput(
        outputData: FloatArray,
        outputShape: LongArray,
        inputSize: Int
    ): List<Detection>? {
        val rows: Int
        val cols: Int

        when {
            outputShape.size == 3 && outputShape[0].toInt() == 1 -> {
                rows = outputShape[1].toInt()
                cols = outputShape[2].toInt()
            }
            outputShape.size == 2 -> {
                rows = outputShape[0].toInt()
                cols = outputShape[1].toInt()
            }
            else -> return null
        }

        if (cols < 6 || rows <= 0) return null

        val rowFormat = rowFormatForColumnCount(cols) ?: return null
        val labels = rowFormat.labels
        val detections = mutableListOf<Detection>()

        for (row in 0 until rows) {
            val offset = row * cols

            val (classIndex, score) = if (rowFormat.classIndexColumn) {
                outputData[offset + 5].toInt()
                    .let { it to outputData[offset + 4] }
            } else {
                var bestClassIdx = 0
                var bestScore = 0f
                for (c in labels.indices) {
                    val classScore = outputData[offset + rowFormat.classStart + c]
                    if (classScore > bestScore) {
                        bestScore = classScore
                        bestClassIdx = c
                    }
                }
                val confidence = rowFormat.objectnessIndex
                    ?.let { outputData[offset + it] * bestScore }
                    ?: bestScore
                bestClassIdx to confidence
            }

            if (score < detectorConfidenceThreshold) continue
            if (classIndex !in labels.indices) continue

            val label = labels[classIndex]
            detections.add(
                Detection(
                    className = label,
                    confidence = score,
                    value = values[label] ?: 0,
                    box = cornerBoxToNormalizedBox(
                        outputData[offset],
                        outputData[offset + 1],
                        outputData[offset + 2],
                        outputData[offset + 3],
                        inputSize
                    )
                )
            )
        }

        return classAwareNms(detections, iouThreshold)
    }

    private data class RowFormat(
        val labels: List<String>,
        val classStart: Int,
        val objectnessIndex: Int?,
        val classIndexColumn: Boolean = false
    )

    private fun rowFormatForColumnCount(cols: Int): RowFormat? {
        return when (cols) {
            6 -> RowFormat(detectorLabels, classStart = 5, objectnessIndex = 4, classIndexColumn = true)
            4 + advancedLabels.size -> RowFormat(advancedLabels, classStart = 4, objectnessIndex = null)
            5 + advancedLabels.size -> RowFormat(advancedLabels, classStart = 5, objectnessIndex = 4)
            4 + detectorLabels.size -> RowFormat(detectorLabels, classStart = 4, objectnessIndex = null)
            5 + detectorLabels.size -> RowFormat(detectorLabels, classStart = 5, objectnessIndex = 4)
            else -> null
        }
    }

    private fun labelsForYoloChannels(channels: Int): List<String>? {
        return when (channels - 4) {
            advancedLabels.size -> advancedLabels
            detectorLabels.size -> detectorLabels
            else -> null
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

    private fun bitmapToNormalizedTensor(bitmap: Bitmap, inputSize: Int): Tensor {
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val channelSize = inputSize * inputSize
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
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )
    }

    private fun bitmapToClassifierTensor(bitmap: Bitmap): Tensor {
        return bitmapToNormalizedTensor(bitmap, classifierInputSize)
    }

    private fun bitmapToUnitTensor(bitmap: Bitmap, inputSize: Int): Tensor {
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val channelSize = inputSize * inputSize
        val floatArray = FloatArray(3 * channelSize)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            floatArray[i] = ((pixel shr 16) and 0xFF) / 255.0f
            floatArray[channelSize + i] = ((pixel shr 8) and 0xFF) / 255.0f
            floatArray[2 * channelSize + i] = (pixel and 0xFF) / 255.0f
        }

        return Tensor.fromBlob(
            floatArray,
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )
    }

    private fun classAwareNms(detections: List<Detection>, threshold: Float): List<Detection> {
        return detections
            .groupBy { it.className }
            .values
            .flatMap { nms(it, threshold) }
            .sortedByDescending { it.confidence }
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

    private fun centerBoxToNormalizedBox(cx: Float, cy: Float, w: Float, h: Float, inputSize: Int): FloatArray {
        val scale = coordinateScale(inputSize, cx, cy, w, h)
        val nx = cx / scale
        val ny = cy / scale
        val nw = w / scale
        val nh = h / scale
        return floatArrayOf(
            (nx - nw / 2f).coerceIn(0f, 1f),
            (ny - nh / 2f).coerceIn(0f, 1f),
            (nx + nw / 2f).coerceIn(0f, 1f),
            (ny + nh / 2f).coerceIn(0f, 1f)
        )
    }

    private fun cornerBoxToNormalizedBox(x1: Float, y1: Float, x2: Float, y2: Float, inputSize: Int): FloatArray {
        val scale = coordinateScale(inputSize, x1, y1, x2, y2)
        return floatArrayOf(
            (x1 / scale).coerceIn(0f, 1f),
            (y1 / scale).coerceIn(0f, 1f),
            (x2 / scale).coerceIn(0f, 1f),
            (y2 / scale).coerceIn(0f, 1f)
        )
    }

    private fun coordinateScale(inputSize: Int, vararg values: Float): Float {
        return if (values.maxOrNull() ?: 0f <= 1.5f) 1f else inputSize.toFloat()
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
        advancedModel?.destroy()
        classifierModel?.destroy()
        advancedModel = null
        classifierModel = null
        isReady = false
    }

    companion object {
        private const val TAG = "CurrencyEngine"
    }
}
