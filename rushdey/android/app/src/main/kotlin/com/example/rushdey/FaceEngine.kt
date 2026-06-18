package com.example.rushdey

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.tensorflow.lite.Interpreter
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * FaceEngine — On-device face recognition using TFLite.
 *
 * Pipeline:
 *  1. Load face_model.tflite (MobileFaceNet backbone)
 *  2. Load/persist templates.json (person → 128-dim embedding)
 *  3. preprocess: resize 112×112, normalize [-1,1]
 *  4. Run inference → 128-dim L2-normalised embedding
 *  5. Cosine similarity vs all templates → return best match / "Unknown"
 *  6. Enroll new person: compute embedding → append to templates.json
 */
class FaceEngine(private val context: Context) {

    private var interpreter: Interpreter? = null
    private val inputSize = 112
    private val embeddingSize = 128
    private val recognitionThreshold = 0.985f
    private val modelAssetName = "assets/models/face_model.tflite"
    private val templatesFileName = "face_templates.json"
    private val faceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .build()
    )

    /** In-memory template DB: personName -> embedding FloatArray */
    private val templates = mutableMapOf<String, FloatArray>()

    data class FaceResult(
        val name: String,
        val similarity: Float,
        val recognized: Boolean
    )

    // ───────────────────────────── Initialization ─────────────────────────────

    fun initialize(): Boolean {
        return try {
            val modelBuffer = loadModelFromAssets()
            val options = Interpreter.Options().apply {
                numThreads = 2
            }
            interpreter = Interpreter(modelBuffer, options)
            loadTemplatesFromDisk()
            Log.i(TAG, "FaceEngine initialized. Templates: ${templates.size}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "FaceEngine init failed", e)
            false
        }
    }

    private fun loadModelFromAssets(): ByteBuffer {
        // First try direct asset key via Flutter loader, then fallback
        val file = copyAssetToInternalIfNeeded(modelAssetName)
        val fis = FileInputStream(file)
        val channel = fis.channel
        val buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
        fis.close()
        return buffer
    }

    private fun copyAssetToInternalIfNeeded(assetName: String): File {
        val dest = File(context.filesDir, assetName)
        if (dest.exists() && dest.length() > 0) return dest
        dest.parentFile?.mkdirs()
        try {
            val key = io.flutter.FlutterInjector.instance()
                .flutterLoader().getLookupKeyForAsset(assetName)
            context.assets.open(key)
        } catch (_: Exception) {
            context.assets.open(assetName)
        }.use { input ->
            FileOutputStream(dest).use { output ->
                input.copyTo(output)
            }
        }
        return dest
    }

    // ────────────────────────── Template DB ───────────────────────────────────

    private fun templatesFile(): File = File(context.filesDir, templatesFileName)

    fun loadTemplatesFromDisk() {
        val file = templatesFile()
        if (!file.exists()) {
            // Also try to copy the seeded templates from assets
            try {
                val key = io.flutter.FlutterInjector.instance()
                    .flutterLoader().getLookupKeyForAsset("assets/models/face_templates.json")
                context.assets.open(key).use { it.copyTo(FileOutputStream(file)) }
            } catch (_: Exception) { /* no seeded templates */ }
        }
        if (!file.exists()) return
        try {
            val json = JSONObject(file.readText())
            val tObj = json.optJSONObject("templates") ?: return
            templates.clear()
            tObj.keys().forEach { name ->
                val arr = tObj.getJSONArray(name)
                val emb = FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
                templates[name] = emb
            }
            Log.i(TAG, "Loaded ${templates.size} face templates from disk")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load templates", e)
        }
    }

    private fun saveTemplatesToDisk() {
        try {
            val tObj = JSONObject()
            templates.forEach { (name, emb) ->
                val arr = org.json.JSONArray()
                emb.forEach { arr.put(it.toDouble()) }
                tObj.put(name, arr)
            }
            val root = JSONObject()
            root.put("version", "1.0")
            root.put("templates", tObj)
            templatesFile().writeText(root.toString(2))
            Log.i(TAG, "Saved ${templates.size} templates to disk")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save templates", e)
        }
    }

    fun listEnrolledPersons(): List<String> = templates.keys.toList().sorted()

    fun deleteEnrolledPerson(name: String): Boolean {
        val removed = templates.remove(name) != null
        if (removed) saveTemplatesToDisk()
        return removed
    }

    // ─────────────────────────── Inference ───────────────────────────────────

    /**
     * Extract 128-dim embedding from a JPEG/PNG byte array (camera photo).
     */
    fun extractEmbedding(imageBytes: ByteArray): FloatArray? {
        return try {
            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                ?: run {
                    Log.w(TAG, "extractEmbedding: could not decode image")
                    return null
                }
            val faceCrop = detectLargestFace(bitmap)
            if (faceCrop == null) {
                Log.w(TAG, "extractEmbedding: no face detected")
                bitmap.recycle()
                return null
            }
            val resized = Bitmap.createScaledBitmap(faceCrop, inputSize, inputSize, true)
            val inputBuffer = bitmapToInputBuffer(resized)

            // Output: [1, 128]
            val outputBuffer = Array(1) { FloatArray(embeddingSize) }
            interpreter?.run(inputBuffer, outputBuffer)
            if (resized !== faceCrop) resized.recycle()
            faceCrop.recycle()
            bitmap.recycle()

            val embedding = outputBuffer[0]
            l2Normalize(embedding)
            embedding
        } catch (e: Exception) {
            Log.e(TAG, "extractEmbedding failed", e)
            null
        }
    }

    /**
     * Recognize face in given imageBytes against enrolled templates.
     */
    fun recognize(imageBytes: ByteArray): FaceResult {
        val embedding = extractEmbedding(imageBytes)
            ?: return FaceResult(NO_FACE_LABEL, 0f, false)

        if (templates.isEmpty()) {
            return FaceResult("لا يوجد أشخاص مسجلين", 0f, false)
        }

        var bestName = "غير معروف"
        var bestScore = 0f
        templates.forEach { (name, tmpl) ->
            val score = cosineSimilarity(embedding, tmpl)
            if (score > bestScore) {
                bestScore = score
                bestName = name
            }
        }

        val recognized = bestScore >= recognitionThreshold
        return FaceResult(
            name = if (recognized) bestName else "شخص غير مسجل",
            similarity = bestScore,
            recognized = recognized
        )
    }

    /**
     * Enroll a person from multiple images (computes mean embedding).
     */
    fun enrollPerson(name: String, imageBytesList: List<ByteArray>): Boolean {
        val embeddings = imageBytesList.mapNotNull { extractEmbedding(it) }
        return enrollPersonFromEmbeddings(name, embeddings)
    }

    fun enrollPersonFromEmbeddings(name: String, embeddings: List<FloatArray>): Boolean {
        if (embeddings.isEmpty()) return false

        // Mean embedding
        val mean = FloatArray(embeddingSize)
        embeddings.forEach { emb -> emb.forEachIndexed { i, v -> mean[i] += v } }
        mean.forEachIndexed { i, _ -> mean[i] /= embeddings.size.toFloat() }
        l2Normalize(mean)

        templates[name] = mean
        saveTemplatesToDisk()
        Log.i(TAG, "Enrolled $name from ${embeddings.size} images")
        return true
    }

    // ─────────────────────────── Preprocessing ───────────────────────────────

    private fun bitmapToInputBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr 8) and 0xFF).toFloat()
            val b = (pixel and 0xFF).toFloat()
            // Normalize to [-1, 1]
            buffer.putFloat(r / 127.5f - 1f)
            buffer.putFloat(g / 127.5f - 1f)
            buffer.putFloat(b / 127.5f - 1f)
        }
        buffer.rewind()
        return buffer
    }

    private fun l2Normalize(vec: FloatArray) {
        var norm = 0f
        vec.forEach { norm += it * it }
        norm = kotlin.math.sqrt(norm.toDouble()).toFloat().coerceAtLeast(1e-6f)
        vec.forEachIndexed { i, v -> vec[i] = v / norm }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        a.forEachIndexed { i, v -> dot += v * b[i] }
        return dot.coerceIn(-1f, 1f)
    }

    fun release() {
        interpreter?.close()
        interpreter = null
        faceDetector.close()
    }

    private fun detectLargestFace(bitmap: Bitmap): Bitmap? {
        return try {
            val image = InputImage.fromBitmap(bitmap, 0)
            val faces = Tasks.await(faceDetector.process(image))
            val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() } ?: return null
            val padded = padRect(face.boundingBox, bitmap.width, bitmap.height, 0.2f)
            Bitmap.createBitmap(bitmap, padded.left, padded.top, padded.width(), padded.height())
        } catch (e: Exception) {
            Log.w(TAG, "Face detection failed", e)
            null
        }
    }

    private fun padRect(rect: Rect, maxW: Int, maxH: Int, padRatio: Float): Rect {
        val padX = (rect.width() * padRatio).toInt()
        val padY = (rect.height() * padRatio).toInt()
        val left = (rect.left - padX).coerceAtLeast(0)
        val top = (rect.top - padY).coerceAtLeast(0)
        val right = (rect.right + padX).coerceAtMost(maxW)
        val bottom = (rect.bottom + padY).coerceAtMost(maxH)
        return Rect(left, top, right, bottom)
    }

    companion object {
        private const val TAG = "FaceEngine"
        const val NO_FACE_LABEL = "__NO_FACE__"
    }
}
