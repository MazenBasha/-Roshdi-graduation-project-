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
 *  3. Detects intent: face_who_is_in_front | currency_count | ocr_read_text | object_obstacle_detection | unknown
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
    private var pendingPartialIntent: String? = null
    private var pendingPartialText: String = ""

    private val faceUtterances = listOf(
        "مين ده", "مين دي", "ده مين", "دي مين",
        "مين اللي قدامي", "مين اللي ادامي", "مين قدامي", "مين ادامي",
        "من اللي قدامي", "من اللي ادامي", "من قدامي", "من ادامي",
        "مين واقف قدامي", "مين واقف ادامي",
        "مين موجود قدامي", "مين موجود ادامي",
        "في حد قدامي", "في حد ادامي",
        "الشخص ده مين", "الشخص دي مين",
        "الراجل ده مين", "الست دي مين", "البنت دي مين", "الولد ده مين",
        "اللي قدامي ده مين", "اللي ادامي ده مين",
        "اعرف مين ده", "اعرف مين دي",
        "عايز اعرف مين ده", "عايز اعرف مين دي",
        "قولي مين ده", "قولي مين دي",
        "قول لي مين ده", "قول لي مين دي",
        "تعرف مين ده", "تعرف مين دي",
        "هو ده مين", "هي دي مين",
        "مين الشخص ده", "مين الشخص اللي قدامي",
        "مين اللي واقف", "مين اللي واقف قدامي",
        "قدامي مين", "ادامي مين",
        "شايف مين", "انت شايف مين",
        "فيه مين قدامي", "فيه مين ادامي"
    ).distinct()

    private val currencyUtterances = listOf(
        "دول كام", "دول كم", "دول بكام",
        "الفلوس دي كام", "الفلوس دي كم",
        "معايا كام", "معايا كم",
        "كام جنيه دول", "كم جنيه دول",
        "المبلغ ده كام", "المبلغ ده كم",
        "قيمة الفلوس كام", "قيمة الفلوس كم",
        "دول قيمتهم كام", "دول قيمتهم كم",
        "احسب الفلوس", "عد الفلوس",
        "عدهم", "احسبهم",
        "شوف دول كام", "شوف دول كم",
        "قولي دول كام", "قولي دول كم",
        "قول لي المبلغ", "قولي المبلغ",
        "معايا قد ايه", "معايا اد ايه",
        "دول قد ايه", "دول اد ايه",
        "كام ورقة", "كم ورقة",
        "عدد الفلوس", "عدد الورق",
        "اعرف معايا كام", "اعرف معايا كم",
        "الفلوس اللي في ايدي كام",
        "الفلوس اللي في ايدي كم",
        "احسبلي الفلوس", "اعدلي الفلوس"
    ).distinct()

    private val ocrUtterances = listOf(
        "اقرا", "اقرالي", "اقرا لي",
        "اقرا المكتوب", "اقرالي المكتوب",
        "اقرا اللي مكتوب", "اقرالي اللي مكتوب",
        "مكتوب ايه", "ايه المكتوب",
        "الكلام ده ايه", "النص ده ايه",
        "اقرا النص", "اقرالي النص",
        "اقرا الكلام", "اقرالي الكلام",
        "قولي مكتوب ايه", "قول لي مكتوب ايه",
        "قولي الكلام", "قول لي الكلام",
        "قولي النص", "قول لي النص",
        "شوف مكتوب ايه", "شوف الكلام ده",
        "اقرا الورقة", "اقرالي الورقة",
        "اقرا اللافتة", "اقرالي اللافتة",
        "اقرا الشاشة", "اقرالي الشاشة",
        "عايز اعرف مكتوب ايه",
        "عايز اقرا اللي قدامي",
        "عايز اقرا اللي ادامي",
        "الكلام اللي قدامي ايه",
        "الكلام اللي ادامي ايه",
        "اقرا ده", "اقرالي ده",
        "ايه اللي مكتوب هنا",
        "مكتوب هنا ايه"
    ).distinct()

    private val objectUtterances = listOf(
        "ايه اللي قدامي", "ايه الي قدامي", "ايه اللي ادامي", "ايه الي ادامي",
        "فيه ايه قدامي", "في ايه قدامي", "فيه ايه ادامي", "في ايه ادامي",
        "فيه ايه امامي", "في ايه امامي",
        "ايه قدامي", "ايه ادامي", "ايه امامي",
        "شايف ايه قدامي", "شايف ايه ادامي", "انت شايف ايه قدامي",
        "في عائق قدامي", "فيه عائق قدامي", "في عائق ادامي", "فيه عائق ادامي",
        "في حاجه قدامي", "فيه حاجه قدامي", "في حاجة قدامي", "فيه حاجة قدامي",
        "هل الطريق فاضي", "الطريق فاضي", "الطريق قدامي فاضي",
        "افتح كشف العوائق", "شغل كشف العوائق", "راقب الطريق",
        "حذرني من العوائق", "نبهني لو في عائق", "نبهني من العوائق",
        "خلي بالك من الطريق", "ابدأ تحذير العوائق"
    ).distinct()

    private val grammarPhrases = (faceUtterances + currencyUtterances + ocrUtterances + objectUtterances).distinct()
    private val normalizedFaceUtterances = normalizePhrases(faceUtterances)
    private val normalizedCurrencyUtterances = normalizePhrases(currencyUtterances)
    private val normalizedOcrUtterances = normalizePhrases(ocrUtterances)
    private val normalizedObjectUtterances = normalizePhrases(objectUtterances)
    private val ocrKeywords = normalizePhrases(
        listOf("اقرا", "اقرالي", "مكتوب", "المكتوب", "النص", "الكلام", "الورقه", "اللافته", "الشاشه")
    )
    private val currencyKeywords = normalizePhrases(
        listOf("فلوس", "جنيه", "مبلغ", "قيمه", "كام", "كم", "عدد", "عد", "احسب", "ورقه", "الورق", "قد ايه", "اد ايه", "بكام")
    )
    private val faceKeywords = normalizePhrases(
        listOf("مين", "من", "شخص", "الشخص", "راجل", "الراجل", "ست", "الست", "بنت", "البنت", "ولد", "الولد", "واقف", "موجود", "حد", "شايف")
    )
    private val objectKeywords = normalizePhrases(
        listOf("عائق", "عوائق", "حاجه", "حاجة", "الطريق", "فاضي", "راقب", "تحذير العوائق", "كشف العوائق", "نبهني", "حذرني")
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
        pendingPartialIntent = null
        pendingPartialText = ""

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
                                    pendingPartialIntent = intent
                                    pendingPartialText = partial
                                    if (isStrongPartialCommand(partial, intent)) {
                                        dispatchIntent(intent, partial, isFinal = false)
                                    }
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
                        handleIntentTimeout()
                    }
                })
            }

            isListening = true
            sendEvent(mapOf("type" to "vosk_status", "status" to "listening_intent"))

            // Timeout safeguard
            timeoutJob = CoroutineScope(Dispatchers.Main).launch {
                delay(timeoutMs)
                if (isListening) {
                    handleIntentTimeout()
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
        pendingPartialIntent = null
        pendingPartialText = ""
    }

    private fun handleIntentTimeout() {
        if (!isListening) return

        val intent = pendingPartialIntent
        val text = pendingPartialText
        if (isKnownIntent(intent) && text.isNotBlank()) {
            dispatchIntent(intent!!, text, isFinal = false)
            return
        }

        sendEvent(mapOf("type" to "intent_timeout"))
        stopListening()
    }

    private fun isKnownIntent(intent: String?): Boolean {
        return intent == "face_who_is_in_front" ||
            intent == "currency_count" ||
            intent == "ocr_read_text" ||
            intent == "object_obstacle_detection"
    }

    private fun isStrongPartialCommand(text: String, intent: String): Boolean {
        val t = normalizeArabic(text)
        val wordCount = t.split(" ").count { it.isNotBlank() }
        if (wordCount < 2) return false

        return when (intent) {
            "ocr_read_text" ->
                containsAnyPhrase(t, normalizedOcrUtterances) ||
                    containsAnyPhrase(t, ocrKeywords)
            "currency_count" ->
                containsAnyPhrase(t, normalizedCurrencyUtterances) ||
                    (wordCount >= 3 && containsAnyPhrase(t, currencyKeywords))
            "face_who_is_in_front" ->
                containsAnyPhrase(t, normalizedFaceUtterances)
            "object_obstacle_detection" ->
                containsAnyPhrase(t, normalizedObjectUtterances) ||
                    (wordCount >= 3 && containsAnyPhrase(t, objectKeywords))
            else -> false
        }
    }

    // ─────────────────────── Intent Detection ────────────────────────────────

    /**
     * Ports detect_intent() from vosk_vosk.py
     */
    fun detectIntent(text: String): String {
        val t = normalizeArabic(text)
        if (t.isBlank()) return "none"

        // Direct utterance match, with OCR > currency > face priority.
        if (containsAnyPhrase(t, normalizedOcrUtterances)) {
            return "ocr_read_text"
        }
        if (containsAnyPhrase(t, normalizedCurrencyUtterances)) {
            return "currency_count"
        }
        if (containsAnyPhrase(t, normalizedFaceUtterances)) {
            return "face_who_is_in_front"
        }
        if (containsAnyPhrase(t, normalizedObjectUtterances)) {
            return "object_obstacle_detection"
        }

        // Keyword fallback, keeping the same priority order.
        if (containsAnyPhrase(t, ocrKeywords)) {
            return "ocr_read_text"
        }
        if (containsAnyPhrase(t, currencyKeywords)) {
            return "currency_count"
        }
        if (containsAnyPhrase(t, faceKeywords)) {
            return "face_who_is_in_front"
        }
        if (containsAnyPhrase(t, objectKeywords)) {
            return "object_obstacle_detection"
        }

        return fuzzyDetect(t)
    }

    private fun normalizeArabic(s: String): String {
        return s.trim()
            .lowercase()
            .replace(Regex("[^\\u0600-\\u06FF\\s]"), " ")
            .replace(Regex("\\s+"), " ")
            .replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            .replace("ة", "ه").replace("ى", "ي")
            .replace("ؤ", "و").replace("ئ", "ي")
            .replace("ق", "ا")
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    private fun normalizePhrases(phrases: List<String>): List<String> {
        return phrases.map { normalizeArabic(it) }
            .filter { it.isNotBlank() }
            .distinct()
    }

    private fun containsAnyPhrase(text: String, phrases: List<String>): Boolean {
        val normalizedText = normalizeArabic(text)
        val wordCount = normalizedText.split(" ").count { it.isNotBlank() }
        return phrases.any { phrase ->
            val p = normalizeArabic(phrase)
            p.isNotBlank() && (
                normalizedText.contains(p) ||
                    (wordCount > 1 && p.contains(normalizedText))
            )
        }
    }

    private fun fuzzyDetect(t: String): String {
        val minimumScore = 0.64

        fun scorePhrases(phrases: List<String>): Double =
            phrases.maxOfOrNull { sequenceSimilarity(t, normalizeArabic(it)) } ?: 0.0

        val ocrScore = scorePhrases(normalizedOcrUtterances)
        val currScore = scorePhrases(normalizedCurrencyUtterances)
        val faceScore = scorePhrases(normalizedFaceUtterances)
        val objectScore = scorePhrases(normalizedObjectUtterances)

        return when {
            ocrScore >= minimumScore && ocrScore >= currScore && ocrScore >= faceScore && ocrScore >= objectScore -> "ocr_read_text"
            currScore >= minimumScore && currScore >= faceScore && currScore >= objectScore -> "currency_count"
            faceScore >= minimumScore && faceScore >= objectScore -> "face_who_is_in_front"
            objectScore >= minimumScore -> "object_obstacle_detection"
            else -> "unknown"
        }
    }

    /** Normalized Levenshtein similarity for small ASR mistakes. */
    private fun sequenceSimilarity(a: String, b: String): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0
        if (a == b) return 1.0

        val maxLength = maxOf(a.length, b.length)
        val distance = levenshteinDistance(a, b)
        return (1.0 - (distance.toDouble() / maxLength)).coerceIn(0.0, 1.0)
    }

    private fun levenshteinDistance(a: String, b: String): Int {
        val previous = IntArray(b.length + 1) { it }
        val current = IntArray(b.length + 1)

        for (i in 1..a.length) {
            current[0] = i
            for (j in 1..b.length) {
                val substitutionCost = if (a[i - 1] == b[j - 1]) 0 else 1
                current[j] = minOf(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitutionCost
                )
            }
            for (j in previous.indices) {
                previous[j] = current[j]
            }
        }

        return previous[b.length]
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
