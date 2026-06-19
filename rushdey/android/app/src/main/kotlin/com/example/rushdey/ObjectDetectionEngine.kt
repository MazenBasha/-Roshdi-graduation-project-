package com.example.rushdey

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.util.ArrayDeque
import java.util.Locale
import kotlin.math.exp

/**
 * Live object detection + visual-proximity warnings for walking assistance.
 *
 * The engine expects a YOLO-style TorchScript Lite model named object_detector.ptl.
 * It tries both Flutter assets and native Android assets so the app can be built
 * before the model is dropped in:
 *
 * - flutter_assets/assets/models/object_detector.ptl
 * - object_detector.ptl
 *
 * Distance is not measured in meters. It follows the local Python prototype and
 * estimates visual proximity from normalized box area/height in the frame.
 */
class ObjectDetectionEngine(private val context: Context) {
    private var model: Module? = null
    private var labels: List<String> = COCO_LABELS
    private var isReady = false

    private val inputSize = 640
    private val confidenceThreshold = 0.40f
    private val iouThreshold = 0.45f
    private val cooldownMs = 2200L
    private val historySize = 3
    private val requiredHits = 2
    private val history = ArrayDeque<Set<String>>()
    private val lastAlertTime = mutableMapOf<String, Long>()

    data class Detection(
        val className: String,
        val confidence: Float,
        val box: FloatArray
    )

    data class AnalyzedDetection(
        val className: String,
        val confidence: Float,
        val distanceHint: String,
        val horizontalPosition: String,
        val areaRatio: Float,
        val heightRatio: Float,
        val box: FloatArray
    )

    data class ObjectResult(
        val ready: Boolean,
        val shouldSpeak: Boolean,
        val messageAr: String,
        val messageEn: String,
        val mainObject: AnalyzedDetection?,
        val allDetections: List<AnalyzedDetection>
    )

    private data class RawModelOutput(
        val data: FloatArray,
        val shape: LongArray
    )

    private data class YoloLayout(
        val objectnessIndex: Int?,
        val classStart: Int,
        val classCount: Int
    )

    private data class ThresholdRule(
        val area: Float?,
        val height: Float?,
        val mode: String = "or"
    )

    private data class ThresholdPair(
        val near: ThresholdRule,
        val veryNear: ThresholdRule
    )

    fun initialize(): Boolean {
        labels = loadLabels()
        model = try {
            val file = copyFirstExistingAssetToFile(
                listOf(
                    "flutter_assets/assets/models/object_detector.ptl",
                    "object_detector.ptl",
                    "flutter_assets/assets/models/yolo11n.ptl",
                    "yolo11n.ptl",
                    "flutter_assets/assets/models/yolov8n.ptl",
                    "yolov8n.ptl"
                ),
                "object_detector.ptl"
            )
            if (file == null) {
                Log.w(TAG, "No object detector model asset found")
                null
            } else {
                Module.load(file.absolutePath)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load object detector", e)
            null
        }

        isReady = model != null
        Log.i(TAG, "ObjectDetectionEngine ready=$isReady labels=${labels.size}")
        return isReady
    }

    fun detect(imageBytes: ByteArray, nowMs: Long = System.currentTimeMillis()): ObjectResult {
        if (!isReady) {
            return ObjectResult(
                ready = false,
                shouldSpeak = false,
                messageAr = "نموذج كشف العوائق غير جاهز",
                messageEn = "Object detector is not ready.",
                mainObject = null,
                allDetections = emptyList()
            )
        }

        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            ?: return ObjectResult(
                ready = true,
                shouldSpeak = false,
                messageAr = "تعذر قراءة صورة الكاميرا",
                messageEn = "Could not read camera image.",
                mainObject = null,
                allDetections = emptyList()
            )

        return try {
            detect(bitmap, nowMs)
        } finally {
            bitmap.recycle()
        }
    }

    private fun detect(bitmap: Bitmap, nowMs: Long): ObjectResult {
        val localModel = model ?: return ObjectResult(false, false, "", "", null, emptyList())
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        try {
            val inputTensor = bitmapToUnitTensor(resized, inputSize)
            val outputTensor = localModel.forward(IValue.from(inputTensor)).toTensor()
            val raw = RawModelOutput(outputTensor.dataAsFloatArray, outputTensor.shape())
            val detections = decodeOutput(raw).sortedByDescending { it.confidence }
            return analyze(detections, nowMs)
        } catch (e: Exception) {
            Log.e(TAG, "Object detector inference failed", e)
            return ObjectResult(
                ready = true,
                shouldSpeak = false,
                messageAr = "تعذر تشغيل كشف العوائق",
                messageEn = "Object detection failed.",
                mainObject = null,
                allDetections = emptyList()
            )
        } finally {
            resized.recycle()
        }
    }

    private fun decodeOutput(output: RawModelOutput): List<Detection> {
        decodeYoloChannelFirstOutput(output)?.let { return it }
        decodeYoloRowOutput(output)?.let { return it }
        Log.w(TAG, "Unsupported object detector output shape=${output.shape.joinToString()}")
        return emptyList()
    }

    private fun decodeYoloChannelFirstOutput(output: RawModelOutput): List<Detection>? {
        val channels: Int
        val anchors: Int
        when {
            output.shape.size == 3 && output.shape[0].toInt() == 1 -> {
                channels = output.shape[1].toInt()
                anchors = output.shape[2].toInt()
            }
            output.shape.size == 2 -> {
                channels = output.shape[0].toInt()
                anchors = output.shape[1].toInt()
            }
            else -> return null
        }
        if (channels < 6 || anchors <= 0) return null

        val layout = yoloLayout(channels) ?: return null

        val localLabels = labelsForClassCount(layout.classCount)
        val detections = mutableListOf<Detection>()
        for (i in 0 until anchors) {
            val cx = output.data[i]
            val cy = output.data[anchors + i]
            val w = output.data[2 * anchors + i]
            val h = output.data[3 * anchors + i]

            var bestClassIdx = 0
            var bestClassScore = 0f
            for (c in 0 until layout.classCount) {
                val score = normalizeScore(output.data[(layout.classStart + c) * anchors + i])
                if (score > bestClassScore) {
                    bestClassScore = score
                    bestClassIdx = c
                }
            }
            val objectness = layout.objectnessIndex?.let { normalizeScore(output.data[it * anchors + i]) } ?: 1f
            val confidence = objectness * bestClassScore
            if (confidence < confidenceThreshold) continue

            detections.add(
                Detection(
                    className = localLabels[bestClassIdx],
                    confidence = confidence,
                    box = centerBoxToNormalizedBox(cx, cy, w, h)
                )
            )
        }

        return classAwareNms(detections, iouThreshold)
    }

    private fun decodeYoloRowOutput(output: RawModelOutput): List<Detection>? {
        val rows: Int
        val cols: Int
        when {
            output.shape.size == 3 && output.shape[0].toInt() == 1 -> {
                rows = output.shape[1].toInt()
                cols = output.shape[2].toInt()
            }
            output.shape.size == 2 -> {
                rows = output.shape[0].toInt()
                cols = output.shape[1].toInt()
            }
            else -> return null
        }
        if (rows <= 0 || cols < 6) return null

        val detections = mutableListOf<Detection>()
        if (cols == 6) {
            val localLabels = labels
            for (row in 0 until rows) {
                val offset = row * cols
                val confidence = normalizeScore(output.data[offset + 4])
                val classIndex = output.data[offset + 5].toInt()
                if (confidence < confidenceThreshold || classIndex !in localLabels.indices) continue
                detections.add(
                    Detection(
                        className = localLabels[classIndex],
                        confidence = confidence,
                        box = cornerBoxToNormalizedBox(
                            output.data[offset],
                            output.data[offset + 1],
                            output.data[offset + 2],
                            output.data[offset + 3]
                        )
                    )
                )
            }
            return classAwareNms(detections, iouThreshold)
        }

        val layout = yoloLayout(cols) ?: return null

        val localLabels = labelsForClassCount(layout.classCount)
        for (row in 0 until rows) {
            val offset = row * cols
            var bestClassIdx = 0
            var bestClassScore = 0f
            for (c in 0 until layout.classCount) {
                val classScore = normalizeScore(output.data[offset + layout.classStart + c])
                if (classScore > bestClassScore) {
                    bestClassScore = classScore
                    bestClassIdx = c
                }
            }
            val objectness = layout.objectnessIndex?.let { normalizeScore(output.data[offset + it]) } ?: 1f
            val confidence = objectness * bestClassScore
            if (confidence < confidenceThreshold) continue
            detections.add(
                Detection(
                    className = localLabels[bestClassIdx],
                    confidence = confidence,
                    box = centerBoxToNormalizedBox(
                        output.data[offset],
                        output.data[offset + 1],
                        output.data[offset + 2],
                        output.data[offset + 3]
                    )
                )
            )
        }

        return classAwareNms(detections, iouThreshold)
    }

    private fun analyze(detections: List<Detection>, nowMs: Long): ObjectResult {
        val analyzed = detections.mapNotNull { analyzeDetection(it) }
        val candidates = analyzed.filter { it.distanceHint == "near" || it.distanceHint == "very_near" }

        val frameKeys = candidates.map { "${it.className}|${it.horizontalPosition}|${it.distanceHint}" }.toSet()
        history.addLast(frameKeys)
        while (history.size > historySize) history.removeFirst()

        val stable = candidates.filter { isStable(it) }
        val main = selectPriorityObject(stable)
            ?: return ObjectResult(true, false, "", "", null, analyzed)

        val key = "${main.className}|${main.horizontalPosition}"
        if (!cooldownPassed(key, nowMs)) {
            return ObjectResult(true, false, "", "", main, analyzed)
        }

        lastAlertTime[key] = nowMs
        val messages = generateMessages(main)
        return ObjectResult(true, true, messages.first, messages.second, main, analyzed)
    }

    private fun analyzeDetection(det: Detection): AnalyzedDetection? {
        val className = det.className.lowercase(Locale.ROOT).trim()
        if (className.isBlank() || det.box.size != 4) return null

        val x1 = det.box[0].coerceIn(0f, 1f)
        val y1 = det.box[1].coerceIn(0f, 1f)
        val x2 = det.box[2].coerceIn(0f, 1f)
        val y2 = det.box[3].coerceIn(0f, 1f)
        if (x2 <= x1 || y2 <= y1) return null

        val width = x2 - x1
        val height = y2 - y1
        val areaRatio = width * height
        val centerX = (x1 + x2) / 2f
        val position = horizontalPosition(centerX)
        val distanceHint = distanceHint(className, areaRatio, height, position)
        return AnalyzedDetection(
            className = className,
            confidence = det.confidence,
            distanceHint = distanceHint,
            horizontalPosition = position,
            areaRatio = areaRatio,
            heightRatio = height,
            box = floatArrayOf(x1, y1, x2, y2)
        )
    }

    private fun horizontalPosition(centerX: Float): String {
        return when {
            centerX < 0.33f -> "left"
            centerX > 0.66f -> "right"
            else -> "center"
        }
    }

    private fun distanceHint(
        className: String,
        areaRatio: Float,
        heightRatio: Float,
        horizontalPosition: String
    ): String {
        val thresholds = thresholdsForClass(className)
        if (conditionMet(thresholds.veryNear, areaRatio, heightRatio, horizontalPosition)) return "very_near"
        if (conditionMet(thresholds.near, areaRatio, heightRatio, horizontalPosition)) return "near"
        return "far"
    }

    private fun thresholdsForClass(className: String): ThresholdPair {
        return when {
            className in LARGE_DANGER_CLASSES -> ThresholdPair(
                near = ThresholdRule(area = 0.18f, height = 0.45f),
                veryNear = ThresholdRule(area = 0.35f, height = 0.65f)
            )
            className in MEDIUM_OBSTACLE_CLASSES -> ThresholdPair(
                near = ThresholdRule(area = 0.12f, height = 0.35f),
                veryNear = ThresholdRule(area = 0.25f, height = 0.55f)
            )
            className in SMALL_OBJECT_CLASSES -> ThresholdPair(
                near = ThresholdRule(area = 0.08f, height = null, mode = "center_area_only"),
                veryNear = ThresholdRule(area = 0.16f, height = null, mode = "center_area_only")
            )
            else -> ThresholdPair(
                near = ThresholdRule(area = 0.15f, height = 0.40f),
                veryNear = ThresholdRule(area = 0.30f, height = 0.60f)
            )
        }
    }

    private fun conditionMet(
        rule: ThresholdRule,
        areaRatio: Float,
        heightRatio: Float,
        horizontalPosition: String
    ): Boolean {
        val areaOk = rule.area?.let { areaRatio > it } ?: false
        val heightOk = rule.height?.let { heightRatio > it } ?: false
        return when (rule.mode) {
            "center_area_only" -> horizontalPosition == "center" && areaOk
            "and" -> areaOk && heightOk
            else -> areaOk || heightOk
        }
    }

    private fun isStable(det: AnalyzedDetection): Boolean {
        var count = 0
        for (frameKeys in history) {
            val stableInFrame = frameKeys.any { key ->
                val parts = key.split("|")
                parts.size == 3 &&
                    parts[0] == det.className &&
                    parts[1] == det.horizontalPosition &&
                    (parts[2] == "near" || parts[2] == "very_near")
            }
            if (stableInFrame) count++
        }
        return count >= requiredHits
    }

    private fun selectPriorityObject(candidates: List<AnalyzedDetection>): AnalyzedDetection? {
        return candidates.minWithOrNull(
            compareBy<AnalyzedDetection> { det ->
                when {
                    det.distanceHint == "very_near" -> 0
                    det.className in DANGER_CLASSES -> 1
                    det.className in MEDIUM_OBSTACLE_CLASSES -> 2
                    det.className == "person" -> 3
                    else -> 4
                }
            }.thenByDescending { it.areaRatio }
        )
    }

    private fun cooldownPassed(key: String, nowMs: Long): Boolean {
        val last = lastAlertTime[key] ?: return true
        return nowMs - last >= cooldownMs
    }

    private fun generateMessages(det: AnalyzedDetection): Pair<String, String> {
        val objectEn = article(det.className)
        val positionEn = POSITION_EN[det.horizontalPosition] ?: "near you"
        val objectAr = ARABIC_CLASS_NAMES[det.className] ?: "جسم"
        val positionAr = POSITION_AR[det.horizontalPosition] ?: "قريب منك"

        return if (det.distanceHint == "very_near") {
            "خطر، $objectAr قريب جدا $positionAr" to
                "Danger, $objectEn is very close $positionEn."
        } else if (det.className in DANGER_CLASSES) {
            "تحذير، $objectAr قريب $positionAr" to
                "Warning, $objectEn is close $positionEn."
        } else {
            "خلي بالك، $objectAr قريب $positionAr" to
                "Be careful, $objectEn is close $positionEn."
        }
    }

    private fun article(name: String): String {
        if (name.isBlank()) return "an object"
        return if ("aeiou".contains(name.first().lowercaseChar())) "an $name" else "a $name"
    }

    private fun bitmapToUnitTensor(bitmap: Bitmap, size: Int): Tensor {
        val pixels = IntArray(size * size)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)

        val channelSize = size * size
        val floatArray = FloatArray(3 * channelSize)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            floatArray[i] = ((pixel shr 16) and 0xFF) / 255.0f
            floatArray[channelSize + i] = ((pixel shr 8) and 0xFF) / 255.0f
            floatArray[2 * channelSize + i] = (pixel and 0xFF) / 255.0f
        }

        return Tensor.fromBlob(floatArray, longArrayOf(1, 3, size.toLong(), size.toLong()))
    }

    private fun classAwareNms(detections: List<Detection>, threshold: Float): List<Detection> {
        return detections
            .groupBy { it.className }
            .values
            .flatMap { nms(it, threshold) }
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
        val areaA = maxOf(0f, a[2] - a[0]) * maxOf(0f, a[3] - a[1])
        val areaB = maxOf(0f, b[2] - b[0]) * maxOf(0f, b[3] - b[1])
        val union = areaA + areaB - intersection
        return if (union > 0f) intersection / union else 0f
    }

    private fun centerBoxToNormalizedBox(cx: Float, cy: Float, w: Float, h: Float): FloatArray {
        val scale = coordinateScale(cx, cy, w, h)
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

    private fun cornerBoxToNormalizedBox(x1: Float, y1: Float, x2: Float, y2: Float): FloatArray {
        val scale = coordinateScale(x1, y1, x2, y2)
        return floatArrayOf(
            (x1 / scale).coerceIn(0f, 1f),
            (y1 / scale).coerceIn(0f, 1f),
            (x2 / scale).coerceIn(0f, 1f),
            (y2 / scale).coerceIn(0f, 1f)
        )
    }

    private fun coordinateScale(vararg values: Float): Float {
        return if ((values.maxOrNull() ?: 0f) <= 1.5f) 1f else inputSize.toFloat()
    }

    private fun normalizeScore(value: Float): Float {
        return if (value in 0f..1f) value else (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()
    }

    private fun yoloLayout(totalChannelsOrColumns: Int): YoloLayout? {
        val noObjectnessClassCount = totalChannelsOrColumns - 4
        val objectnessClassCount = totalChannelsOrColumns - 5
        return when {
            noObjectnessClassCount > 0 && noObjectnessClassCount == labels.size ->
                YoloLayout(objectnessIndex = null, classStart = 4, classCount = noObjectnessClassCount)
            objectnessClassCount > 0 && objectnessClassCount == labels.size ->
                YoloLayout(objectnessIndex = 4, classStart = 5, classCount = objectnessClassCount)
            noObjectnessClassCount == COCO_LABELS.size ->
                YoloLayout(objectnessIndex = null, classStart = 4, classCount = noObjectnessClassCount)
            objectnessClassCount == COCO_LABELS.size ->
                YoloLayout(objectnessIndex = 4, classStart = 5, classCount = objectnessClassCount)
            noObjectnessClassCount in 1..200 && totalChannelsOrColumns <= 84 ->
                YoloLayout(objectnessIndex = null, classStart = 4, classCount = noObjectnessClassCount)
            objectnessClassCount in 1..200 ->
                YoloLayout(objectnessIndex = 4, classStart = 5, classCount = objectnessClassCount)
            else -> null
        }
    }

    private fun labelsForClassCount(classCount: Int): List<String> {
        return when {
            labels.size == classCount -> labels
            classCount <= labels.size -> labels.take(classCount)
            else -> List(classCount) { "class_$it" }
        }
    }

    private fun loadLabels(): List<String> {
        val text = readFirstExistingAsset(
            listOf(
                "flutter_assets/assets/models/object_labels.txt",
                "object_labels.txt",
                "flutter_assets/assets/models/coco_labels.txt",
                "coco_labels.txt"
            )
        ) ?: return COCO_LABELS
        val parsed = text.lines()
            .map { it.trim() }
            .filter { it.isNotBlank() && !it.startsWith("#") }
        return parsed.ifEmpty { COCO_LABELS }
    }

    private fun readFirstExistingAsset(paths: List<String>): String? {
        for (path in paths) {
            try {
                context.assets.open(path).use { input ->
                    return input.bufferedReader(Charsets.UTF_8).readText()
                }
            } catch (_: Exception) {
                // Try next path.
            }
        }
        return null
    }

    private fun copyFirstExistingAssetToFile(paths: List<String>, outputName: String): File? {
        val outFile = File(context.filesDir, outputName)
        for (assetPath in paths) {
            try {
                context.assets.open(assetPath).use { input ->
                    FileOutputStream(outFile).use { output -> input.copyTo(output) }
                }
                if (outFile.length() > 0) return outFile
            } catch (_: Exception) {
                // Try next path.
            }
        }
        return null
    }

    fun release() {
        model?.destroy()
        model = null
        isReady = false
        history.clear()
        lastAlertTime.clear()
    }

    companion object {
        private const val TAG = "ObjectDetectionEngine"

        private val LARGE_DANGER_CLASSES = setOf("person", "car", "bus", "truck", "motorcycle", "bicycle")
        private val DANGER_CLASSES = setOf("car", "bus", "truck", "motorcycle", "bicycle")
        private val MEDIUM_OBSTACLE_CLASSES = setOf(
            "chair", "bench", "couch", "sofa", "dining table", "table", "suitcase", "backpack",
            "potted plant", "bed", "toilet", "door", "wall", "stairs", "obstacle"
        )
        private val SMALL_OBJECT_CLASSES = setOf("bottle", "cup", "cell phone", "book", "remote", "mouse", "keyboard")

        private val POSITION_EN = mapOf(
            "left" to "on your left",
            "center" to "in front of you",
            "right" to "on your right"
        )
        private val POSITION_AR = mapOf(
            "left" to "على الشمال",
            "center" to "قدامك",
            "right" to "على اليمين"
        )

        private val ARABIC_CLASS_NAMES = mapOf(
            "person" to "شخص",
            "bicycle" to "عجلة",
            "car" to "عربية",
            "motorcycle" to "موتوسيكل",
            "airplane" to "طائرة",
            "bus" to "أتوبيس",
            "train" to "قطار",
            "truck" to "شاحنة",
            "boat" to "مركب",
            "traffic light" to "إشارة",
            "fire hydrant" to "صنبور حريق",
            "stop sign" to "علامة توقف",
            "parking meter" to "عداد",
            "bench" to "دكة",
            "bird" to "طائر",
            "cat" to "قطة",
            "dog" to "كلب",
            "horse" to "حصان",
            "sheep" to "خروف",
            "cow" to "بقرة",
            "elephant" to "فيل",
            "bear" to "دب",
            "zebra" to "حمار وحشي",
            "giraffe" to "زرافة",
            "backpack" to "شنطة ظهر",
            "umbrella" to "شمسية",
            "handbag" to "شنطة",
            "tie" to "كرافتة",
            "suitcase" to "شنطة",
            "frisbee" to "طبق",
            "skis" to "زلاجات",
            "snowboard" to "لوح تزحلق",
            "sports ball" to "كرة",
            "kite" to "طائرة ورقية",
            "baseball bat" to "مضرب",
            "baseball glove" to "قفاز",
            "skateboard" to "لوح تزحلق",
            "surfboard" to "لوح",
            "tennis racket" to "مضرب",
            "bottle" to "زجاجة",
            "wine glass" to "كأس",
            "cup" to "كوباية",
            "fork" to "شوكة",
            "knife" to "سكينة",
            "spoon" to "معلقة",
            "bowl" to "طبق",
            "banana" to "موزة",
            "apple" to "تفاحة",
            "sandwich" to "ساندوتش",
            "orange" to "برتقالة",
            "broccoli" to "بروكلي",
            "carrot" to "جزرة",
            "hot dog" to "هوت دوج",
            "pizza" to "بيتزا",
            "donut" to "دونات",
            "cake" to "كيكة",
            "chair" to "كرسي",
            "couch" to "كنبة",
            "sofa" to "كنبة",
            "potted plant" to "زرع",
            "bed" to "سرير",
            "dining table" to "ترابيزة",
            "table" to "ترابيزة",
            "toilet" to "حمام",
            "tv" to "تلفزيون",
            "laptop" to "لاب توب",
            "mouse" to "ماوس",
            "remote" to "ريموت",
            "keyboard" to "كيبورد",
            "cell phone" to "موبايل",
            "microwave" to "ميكروويف",
            "oven" to "فرن",
            "toaster" to "توستر",
            "sink" to "حوض",
            "refrigerator" to "ثلاجة",
            "book" to "كتاب",
            "clock" to "ساعة",
            "vase" to "فازة",
            "scissors" to "مقص",
            "teddy bear" to "دبدوب",
            "hair drier" to "مجفف شعر",
            "toothbrush" to "فرشة أسنان",
            "door" to "باب",
            "wall" to "حيطة",
            "stairs" to "سلم",
            "obstacle" to "عائق"
        )

        private val COCO_LABELS = listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        )
    }
}
