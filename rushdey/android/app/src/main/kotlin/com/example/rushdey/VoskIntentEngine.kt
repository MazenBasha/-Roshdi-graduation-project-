package com.example.rushdey

import android.content.Context
import android.util.Log
import io.flutter.plugin.common.EventChannel
import kotlinx.coroutines.*
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.URL

/**
 * VoskIntentEngine — Arabic intent detection using Vosk on Android.
 *
 * Ports the logic from vosk_vosk.py to Kotlin/Android.
 *
 * After the wake word is detected:
 *  1. Starts Vosk ASR with Arabic model
 *  2. Listens for up to [timeoutMs] ms
 *  3. Detects intent: face_who_is_in_front | currency_count | ocr_read_text | unknown
 *  4. Sends intent via EventChannel to Flutter
 *
 * The Arabic Vosk model (~50 MB) is downloaded on first run and cached.
 */
class VoskIntentEngine(
    private val context: Context,
    private val eventSink: () -> EventChannel.EventSink?
) {
    private var model: Model? get() = sharedModel; set(value) { sharedModel = value }
    private var speechService: SpeechService? = null
    private var isListening = false
    private val modelDirName = "vosk-model-ar"
    private val modelUrl = "https://alphacephei.com/vosk/models/vosk-model-ar-mgb2-0.4.zip"
    private val timeoutMs = 8000L
    private var timeoutJob: Job? = null

    // Grammar phrases (matching vosk_vosk.py)
    private val grammarPhrases = listOf(
        "مين", "مين ده", "مين دي", "مين قدامي", "من قدامي",
        "مين امامي", "من امامي", "ده مين", "هذا مين",
        "مكتوب", "اقرا", "اقرالي", "النص", "المكتوب",
        "كام", "كم", "دول كام", "دول كم", "عدد", "قد ايه"
    )

    // ─────────────────────── Init ─────────────────────────────────────────────

    /**
     * Initialize Vosk model. Downloads if not cached. Returns false if unavailable.
     */
    fun initialize(onReady: (Boolean) -> Unit) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val modelDir = getOrDownloadModel()
                if (modelDir == null) {
                    Log.e(TAG, "Vosk model not available")
                    withContext(Dispatchers.Main) { onReady(false) }
                    return@launch
                }
                if (sharedModel == null) {
                    sharedModel = Model(modelDir.absolutePath)
                }

                Log.i(TAG, "Vosk model loaded from: ${modelDir.absolutePath}")
                withContext(Dispatchers.Main) { onReady(true) }
            } catch (e: Exception) {
                Log.e(TAG, "Vosk init failed", e)
                withContext(Dispatchers.Main) { onReady(false) }
            }
        }
    }

    /**
     * Returns model directory (downloads if needed). Returns null if no internet.
     */
    private fun getOrDownloadModel(): File? {
        // Check if bundled in assets (for offline APK)
        val bundledDir = copyBundledModelIfNeeded()
        if (bundledDir != null) return bundledDir

        // Check cache
        val cacheDir = File(context.filesDir, modelDirName)
        if (cacheDir.exists() && cacheDir.list()?.isNotEmpty() == true) {
            return cacheDir
        }

        // Try to download
        return try {
            Log.i(TAG, "Downloading Vosk Arabic model...")
            sendEvent(mapOf("type" to "vosk_status", "status" to "downloading_model"))
            downloadAndExtractModel(cacheDir)
            sendEvent(mapOf("type" to "vosk_status", "status" to "model_ready"))
            cacheDir
        } catch (e: Exception) {
            Log.e(TAG, "Failed to download Vosk model: $e")
            sendEvent(mapOf("type" to "vosk_status", "status" to "model_unavailable"))
            null
        }
    }

    private fun copyBundledModelIfNeeded(): File? {
        return try {
            val destDir = File(context.filesDir, modelDirName)
            if (isUsableVoskDir(destDir)) return destDir

            destDir.deleteRecursively()
            destDir.mkdirs()
            copyAssetTree(modelDirName, destDir)
            if (isUsableVoskDir(destDir)) destDir else null
        } catch (e: Exception) {
            Log.w(TAG, "No bundled Vosk model found in Android assets", e)
            null
        }
    }

    private fun isUsableVoskDir(dir: File): Boolean {
        return File(dir, "am/final.mdl").exists() &&
            File(dir, "graph/HCLr.fst").exists() &&
            File(dir, "graph/Gr.fst").exists() &&
            File(dir, "conf/model.conf").exists()
    }

    private fun copyAssetTree(assetPath: String, targetDir: File) {
        val children = context.assets.list(assetPath).orEmpty()
        if (children.isEmpty()) {
            targetDir.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                FileOutputStream(targetDir).use { output -> input.copyTo(output) }
            }
            return
        }

        targetDir.mkdirs()
        children.forEach { child ->
            copyAssetTree("$assetPath/$child", File(targetDir, child))
        }
    }

    private fun downloadAndExtractModel(destDir: File) {
        val zipFile = File(context.cacheDir, "vosk_model.zip")
        URL(modelUrl).openStream().use { input ->
            FileOutputStream(zipFile).use { output -> input.copyTo(output) }
        }
        destDir.mkdirs()
        extractZip(zipFile, destDir)
        zipFile.delete()
    }

    private fun extractZip(zipFile: File, destDir: File) {
        java.util.zip.ZipInputStream(zipFile.inputStream()).use { zip ->
            var entry = zip.nextEntry
            while (entry != null) {
                val name = entry.name.substringAfter("/") // strip top-level folder
                if (name.isNotBlank()) {
                    val target = File(destDir, name)
                    if (entry.isDirectory) {
                        target.mkdirs()
                    } else {
                        target.parentFile?.mkdirs()
                        FileOutputStream(target).use { zip.copyTo(it) }
                    }
                }
                zip.closeEntry()
                entry = zip.nextEntry
            }
        }
    }

    // ─────────────────────── Listening ───────────────────────────────────────

    fun startListening() {
        if (model == null) {
            sendEvent(mapOf("type" to "intent_error", "message" to "Vosk model not loaded"))
            return
        }
        if (isListening) return

        try {
            // Use standard recognizer (runtime graphs not supported by this model)
            val recognizer = Recognizer(model, 16000.0f)

            speechService = SpeechService(recognizer, 16000.0f).apply {
                startListening(object : RecognitionListener {
                    override fun onPartialResult(hypothesis: String?) {
                        if (hypothesis.isNullOrBlank()) return
                        try {
                            val partial = JSONObject(hypothesis).optString("partial", "")
                            if (partial.isNotBlank()) {
                                val intent = detectIntent(partial)
                                if (intent != "none" && intent != "unknown") {
                                    // Got a confident early result
                                    dispatchIntent(intent, partial, isFinal = false)
                                }
                            }
                        } catch (_: Exception) {}
                    }

                    override fun onResult(hypothesis: String?) {
                        if (hypothesis.isNullOrBlank()) return
                        try {
                            val text = JSONObject(hypothesis).optString("text", "")
                            if (text.isNotBlank()) {
                                val intent = detectIntent(text)
                                dispatchIntent(intent, text, isFinal = true)
                            }
                        } catch (_: Exception) {}
                    }

                    override fun onFinalResult(hypothesis: String?) {
                        onResult(hypothesis)
                    }

                    override fun onError(exception: Exception?) {
                        Log.e(TAG, "Vosk error", exception)
                        sendEvent(mapOf("type" to "intent_error", "message" to (exception?.message ?: "Vosk error")))
                        stopListening()
                    }

                    override fun onTimeout() {
                        sendEvent(mapOf("type" to "intent_timeout"))
                        stopListening()
                    }
                })
            }

            isListening = true
            sendEvent(mapOf("type" to "vosk_status", "status" to "listening_intent"))

            // Timeout safeguard
            timeoutJob = CoroutineScope(Dispatchers.Main).launch {
                delay(timeoutMs)
                if (isListening) {
                    sendEvent(mapOf("type" to "intent_timeout"))
                    stopListening()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "startListening failed", e)
            sendEvent(mapOf("type" to "intent_error", "message" to (e.message ?: "Unknown error")))
        }
    }

    fun stopListening() {
        timeoutJob?.cancel()
        speechService?.apply {
            stop()
            shutdown()
        }
        speechService = null
        isListening = false
    }

    // ─────────────────────── Intent Detection ────────────────────────────────

    /**
     * Ports detect_intent() from vosk_vosk.py
     */
    fun detectIntent(text: String): String {
        val t = normalizeArabic(text)
        if (t.isBlank()) return "none"

        // OCR: priority first
        if (t.contains("مكتوب") || t.contains("اقرا") || t.contains("اقرالي") ||
            t.contains("النص") || t.contains("المكتوب")) {
            return "ocr_read_text"
        }

        // Currency
        if (t.contains("كام") || t.contains("كم") || t.contains("عدد") || t.contains("قد ايه")) {
            return "currency_count"
        }

        // Face
        if (t.contains("مين") || t.contains("من")) {
            if (t.contains("كام") || t.contains("كم")) return "currency_count"
            return "face_who_is_in_front"
        }

        // Fuzzy fallback
        return fuzzyDetect(t)
    }

    private fun normalizeArabic(s: String): String {
        return s.trim()
            .replace(Regex("[^\\u0600-\\u06FF\\s]"), " ")
            .replace(Regex("\\s+"), " ")
            .replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            .replace("ة", "ه").replace("ى", "ي")
            .replace("ؤ", "و").replace("ئ", "ي")
    }

    private fun fuzzyDetect(t: String): String {
        val facePhrases = listOf("مين", "مين ده", "مين قدامي", "من قدامي")
        val currencyPhrases = listOf("كام", "كم", "دول كام", "عدد", "قد ايه")
        val ocrPhrases = listOf("مكتوب", "اقرا", "اقرالي", "النص")

        var bestLabel = "unknown"
        var bestScore = 0.55 // minimum threshold

        fun scorePhrases(phrases: List<String>): Double =
            phrases.maxOfOrNull { sequenceSimilarity(t, normalizeArabic(it)) } ?: 0.0

        val faceScore = scorePhrases(facePhrases)
        val currScore = scorePhrases(currencyPhrases)
        val ocrScore  = scorePhrases(ocrPhrases)

        if (faceScore > bestScore) { bestScore = faceScore; bestLabel = "face_who_is_in_front" }
        if (currScore > bestScore) { bestScore = currScore; bestLabel = "currency_count" }
        if (ocrScore  > bestScore) { bestLabel = "ocr_read_text" }

        return bestLabel
    }

    /** Simple character-level overlap ratio (like Python's SequenceMatcher) */
    private fun sequenceSimilarity(a: String, b: String): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0
        val longer = if (a.length >= b.length) a else b
        val shorter = if (a.length < b.length) a else b
        var matches = 0
        shorter.forEach { c -> if (longer.contains(c)) matches++ }
        return 2.0 * matches / (a.length + b.length)
    }

    private fun buildGrammarJson(): String {
        val arr = org.json.JSONArray()
        grammarPhrases.forEach { arr.put(it) }
        return arr.toString()
    }

    private fun dispatchIntent(intent: String, text: String, isFinal: Boolean) {
        if (intent == "none" || intent == "unknown") return
        Log.i(TAG, "Intent detected: $intent (text='$text', final=$isFinal)")
        stopListening()
        sendEvent(mapOf(
            "type" to "intent_detected",
            "intent" to intent,
            "text" to text,
            "final" to isFinal
        ))
    }

    private fun sendEvent(event: Map<String, Any>) {
        CoroutineScope(Dispatchers.Main).launch {
            eventSink()?.success(event)
        }
    }

    fun release() {
        stopListening()
        /* Do not close shared model here */
    }

    companion object {
        var sharedModel: Model? = null

        private const val TAG = "VoskIntentEngine"
    }
}
