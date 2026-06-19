package com.example.rushdey

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.provider.Settings
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import android.util.Log
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Locale
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import java.util.concurrent.Executors

class MainActivity : FlutterActivity(), TextToSpeech.OnInitListener {
    // Channel names
    private val wakeWordMethodChannel = "com.example.rushdey/wakeword"
    private val wakeWordEventChannel  = "com.example.rushdey/wakeword_events"
    private val modelsMethodChannel   = "com.example.rushdey/models"
    private val modelsEventChannel    = "com.example.rushdey/model_events"

    // Event sinks
    private var wakeWordEventSink: EventChannel.EventSink? = null
    private var modelEventSink: EventChannel.EventSink? = null

    // Pending results
    private var pendingStartResult: MethodChannel.Result? = null
    private var pendingCameraResult: MethodChannel.Result? = null
    private var pendingCameraIntent: String = ""   // "face" or "currency"
    private var pendingEnrollName: String = ""
    private val enrollEmbeddings: MutableList<FloatArray> = mutableListOf()
    private val enrollPhotosNeeded = 3
    private val enrollMaxAttempts = 8
    private val enrollCaptureDelayMs = 1200L
    private var enrollCaptureAttempts = 0

    // ML Engines
    private lateinit var faceEngine: FaceEngine
    private lateinit var currencyEngine: CurrencyEngine
    private lateinit var ocrEngine: OCREngine
    private lateinit var objectDetectionEngine: ObjectDetectionEngine
    private var voskEngine: VoskIntentEngine? = null
    private var voskReady = false
    private var voskInitializing = false
    private var startIntentAfterVoskReady = false
    private var faceReady = false
    private var currencyReady = false
    private var ocrReady = false
    private var objectDetectionReady = false
    private var useYoloCurrencyModel = false

    // TTS
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var preferredTtsVoiceName: String? = null
    private val ttsEnginePackage = "com.google.android.tts"
    private var audioManager: AudioManager? = null
    private var audioFocusRequest: AudioFocusRequest? = null

    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var cameraLensFacing: Int = CameraSelector.LENS_FACING_BACK
    private var cameraMirror: Boolean = false
    private var cameraPreviewActive: Boolean = false
    private var cameraPreviewFrozen: Boolean = false
    private var cameraCaptureInProgress: Boolean = false
    private var lastPreviewJpegBytes: ByteArray? = null
    private var lastPreviewFrameAt: Long = 0L
    private val previewFrameIntervalMs = 220L
    private var objectDetectionActive: Boolean = false
    private var objectDetectionInProgress: Boolean = false
    private var lastObjectDetectionAt: Long = 0L
    private val objectDetectionIntervalMs = 260L
    private var cameraInitInProgress: Boolean = false
    private var cameraRebindRequested: Boolean = false
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    private val mainHandler = Handler(Looper.getMainLooper())

    // ─────────────────────── Flutter Engine Setup ─────────────────────────────

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // ── Wake Word channels (existing) ────────────────────────────────────
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, wakeWordMethodChannel)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "startListening" -> { val engine = call.argument<String>("engine") ?: "pytorch"; startListeningWithPermission(result, engine) }
                    "stopListening"  -> { stopWakeWordService(); result.success(true) }
                    "getStatus"      -> {
                        val listening = WakeWordService.getInstance() != null
                        result.success(mapOf(
                            "isListening" to listening,
                            "status"      to if (listening) "listening" else "stopped",
                            "lastError"   to WakeWordService.getLastError()
                        ))
                    }
                    else -> result.notImplemented()
                }
            }

        EventChannel(flutterEngine.dartExecutor.binaryMessenger, wakeWordEventChannel)
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    wakeWordEventSink = events
                    WakeWordService.setPendingEventSink(events)
                    WakeWordService.getInstance()?.setEventSink(events)
                }
                override fun onCancel(arguments: Any?) {
                    wakeWordEventSink = null
                    WakeWordService.setPendingEventSink(null)
                    WakeWordService.getInstance()?.setEventSink(null)
                }
            })

        // ── Models channel (NEW) ─────────────────────────────────────────────
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, modelsMethodChannel)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "initModels"            -> handleInitModels(result)
                    "startIntentListening"  -> handleStartIntentListening(result)
                    "stopIntentListening"   -> handleStopIntentListening(result)
                    "setCameraConfig"       -> handleSetCameraConfig(call.arguments, result)
                    "getCameraConfig"       -> handleGetCameraConfig(result)
                    "setCurrencyModelMode"  -> handleSetCurrencyModelMode(call.arguments, result)
                    "getCurrencyModelMode"  -> handleGetCurrencyModelMode(result)
                    "startCameraPreview"    -> handleStartCameraPreview(result)
                    "stopCameraPreview"     -> handleStopCameraPreview(result)
                    "startObjectDetection"  -> handleStartObjectDetection(result)
                    "stopObjectDetection"   -> handleStopObjectDetection(result)
                    "captureAndRecognizeFace"     -> handleCaptureFace(result)
                    "captureAndDetectCurrency"    -> handleCaptureCurrency(result)
                    "captureAndReadText"          -> handleCaptureOCR(result)
                    "enrollPerson"          -> handleEnrollPerson(call.arguments, result)
                    "deleteEnrolledPerson"  -> handleDeletePerson(call.arguments, result)
                    "listEnrolledPersons"   -> handleListPersons(result)
                    "listTtsVoices"         -> handleListTtsVoices(result)
                    "setTtsVoice"           -> handleSetTtsVoice(call.arguments, result)
                    "getTtsVoice"           -> handleGetTtsVoice(result)
                    "openTtsSettings"       -> handleOpenTtsSettings(result)
                    "openTtsInstallData"    -> handleOpenTtsInstallData(result)
                    "speakText"             -> { speak(call.arguments as? String ?: ""); result.success(null) }
                    else -> result.notImplemented()
                }
            }

        EventChannel(flutterEngine.dartExecutor.binaryMessenger, modelsEventChannel)
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    modelEventSink = events
                    if (cameraPreviewActive) {
                        sendModelEvent(mapOf("type" to if (cameraPreviewFrozen) "camera_frozen" else "camera_live"))
                        lastPreviewJpegBytes?.let { bytes ->
                            sendModelEvent(mapOf("type" to "camera_preview", "imageBytes" to bytes))
                        }
                    }
                }
                override fun onCancel(arguments: Any?) {
                    modelEventSink = null
                }
            })

        // Init TTS
        audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
        tts = TextToSpeech(this, this, ttsEnginePackage)

        // Init ML engines early
        initEngines()
    }

    private fun initCamera() {
        if (isFinishing || isDestroyed) return
        if (cameraInitInProgress) {
            cameraRebindRequested = true
            return
        }
        cameraInitInProgress = true
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                val previousAnalysis = imageAnalysis
                if (!cameraPreviewActive && pendingCameraIntent.isBlank()) {
                    previousAnalysis?.clearAnalyzer()
                    cameraProvider.unbindAll()
                    imageCapture = null
                    imageAnalysis = null
                    return@addListener
                }
                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()
                imageAnalysis = if (cameraPreviewActive) {
                    ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also { analysis ->
                            analysis.setAnalyzer(cameraExecutor) { image ->
                                handlePreviewFrame(image)
                            }
                        }
                } else {
                    null
                }
                val cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(cameraLensFacing)
                    .build()
                previousAnalysis?.clearAnalyzer()
                cameraProvider.unbindAll()
                val capture = imageCapture
                val analysis = imageAnalysis
                if (cameraPreviewActive && analysis != null && capture != null) {
                    cameraProvider.bindToLifecycle(this, cameraSelector, capture, analysis)
                } else if (capture != null) {
                    cameraProvider.bindToLifecycle(this, cameraSelector, capture)
                }
            } catch (e: Exception) {
                Log.e(TAG, "CameraX init failed", e)
            } finally {
                cameraInitInProgress = false
                if (cameraRebindRequested) {
                    cameraRebindRequested = false
                    initCamera()
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ─────────────────────── TTS ──────────────────────────────────────────────

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val locale = Locale("ar", "EG")
            
            applyPreferredTtsVoice()
            
            ttsReady = true
            // A slightly lower pitch helps avoid a feminine-sounding fallback voice.
            tts?.setPitch(0.82f)
            tts?.setSpeechRate(0.98f)
            tts?.setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {}
                override fun onDone(utteranceId: String?) { abandonAudioFocus() }
                override fun onError(utteranceId: String?) { abandonAudioFocus() }
            })
            
            Log.i(TAG, "TTS init: ready=$ttsReady")
        }
    }

    private fun selectPreferredArabicMaleVoice(voices: Set<Voice>?): Voice? {
        val arabicVoices = voices.orEmpty()
            .filter { it.locale.language.equals("ar", ignoreCase = true) }
        if (arabicVoices.isEmpty()) return null

        val maleVoices = arabicVoices.filter { isLikelyMaleVoice(it.name) }

        return maleVoices.find { isEgyptianArabicVoice(it) && !it.isNetworkConnectionRequired }
            ?: maleVoices.find { !it.isNetworkConnectionRequired }
            ?: maleVoices.find { isEgyptianArabicVoice(it) }
            ?: maleVoices.firstOrNull()
            ?: arabicVoices.find { isEgyptianArabicVoice(it) && !it.isNetworkConnectionRequired }
            ?: arabicVoices.find { !it.isNetworkConnectionRequired }
            ?: arabicVoices.find { isEgyptianArabicVoice(it) }
            ?: arabicVoices.firstOrNull()
    }

    private fun isEgyptianArabicVoice(voice: Voice): Boolean {
        val name = voice.name.lowercase(Locale.ROOT)
        return voice.locale.country.equals("EG", ignoreCase = true) ||
            name.contains("eg") ||
            name.contains("egypt") ||
            name.contains("arz")
    }

    private fun isLikelyMaleVoice(name: String): Boolean {
        val n = name.lowercase(Locale.ROOT)
        return listOf(
            "male",
            "man",
            "masculine",
            "m1",
            "m2",
            "m3",
            "ar-xa-x-arm",
            "ar-xa-x-arc",
            "ar-xa-x-ard"
        ).any { n.contains(it) }
    }

    private fun applyPreferredTtsVoice(): Voice? {
        val locale = Locale("ar", "EG")
        val voices = tts?.voices.orEmpty()
        val preferred = preferredTtsVoiceName
            ?.let { voiceName -> voices.firstOrNull { it.name == voiceName } }
        val selected = preferred ?: selectPreferredArabicMaleVoice(voices)

        if (selected != null) {
            tts?.voice = selected
            Log.i(TAG, "Selected Arabic TTS voice: ${selected.name}")
        } else {
            tts?.setLanguage(locale)
            Log.w(TAG, "No Arabic TTS voice found; falling back to locale $locale")
        }

        return selected
    }

    private fun handleListTtsVoices(result: MethodChannel.Result) {
        val voices = tts?.voices.orEmpty()
            .sortedWith(compareByDescending<Voice> { it.locale.language.equals("ar", ignoreCase = true) }
                .thenByDescending { isEgyptianArabicVoice(it) }
                .thenByDescending { isLikelyMaleVoice(it.name) }
                .thenBy { it.isNetworkConnectionRequired }
                .thenBy { it.locale.toLanguageTag() }
                .thenBy { it.name })
            .map { voiceToMap(it) }

        result.success(mapOf(
            "ttsReady" to ttsReady,
            "preferredVoiceName" to preferredTtsVoiceName,
            "selectedVoiceName" to tts?.voice?.name,
            "voices" to voices
        ))
    }

    private fun handleSetTtsVoice(args: Any?, result: MethodChannel.Result) {
        val map = args as? Map<*, *>
        val voiceName = (map?.get("voiceName") as? String)
            ?: (map?.get("name") as? String)
        preferredTtsVoiceName = voiceName?.takeIf { it.isNotBlank() }
        val selected = if (ttsReady) applyPreferredTtsVoice() else null

        result.success(mapOf(
            "ttsReady" to ttsReady,
            "preferredVoiceName" to preferredTtsVoiceName,
            "selectedVoice" to selected?.let { voiceToMap(it) }
        ))
    }

    private fun handleGetTtsVoice(result: MethodChannel.Result) {
        result.success(mapOf(
            "ttsReady" to ttsReady,
            "preferredVoiceName" to preferredTtsVoiceName,
            "selectedVoice" to tts?.voice?.let { voiceToMap(it) }
        ))
    }

    private fun handleOpenTtsSettings(result: MethodChannel.Result) {
        openTtsIntent(Intent(ACTION_TEXT_TO_SPEECH_SETTINGS), result)
    }

    private fun handleOpenTtsInstallData(result: MethodChannel.Result) {
        try {
            startActivity(Intent(TextToSpeech.Engine.ACTION_INSTALL_TTS_DATA).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            result.success(true)
        } catch (e: Exception) {
            Log.w(TAG, "TTS install data screen unavailable; opening TTS settings", e)
            openTtsIntent(Intent(ACTION_TEXT_TO_SPEECH_SETTINGS), result)
        }
    }

    private fun openTtsIntent(intent: Intent, result: MethodChannel.Result) {
        try {
            startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            result.success(true)
        } catch (e: Exception) {
            try {
                startActivity(Intent(Settings.ACTION_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                result.success(true)
            } catch (fallback: Exception) {
                Log.e(TAG, "Failed to open TTS screen", fallback)
                result.error("TTS_SETTINGS_UNAVAILABLE", fallback.message ?: "TTS settings unavailable", null)
            }
        }
    }

    private fun voiceToMap(voice: Voice): Map<String, Any?> {
        return mapOf(
            "name" to voice.name,
            "locale" to voice.locale.toLanguageTag(),
            "language" to voice.locale.language,
            "country" to voice.locale.country,
            "displayLocale" to voice.locale.getDisplayName(Locale("ar", "EG")),
            "isArabic" to voice.locale.language.equals("ar", ignoreCase = true),
            "isEgyptian" to isEgyptianArabicVoice(voice),
            "isLikelyMale" to isLikelyMaleVoice(voice.name),
            "isNetworkRequired" to voice.isNetworkConnectionRequired,
            "quality" to voice.quality,
            "latency" to voice.latency,
            "features" to voice.features.orEmpty().toList().sorted()
        )
    }

    private var lastSpokenText: String? = null
    private var lastSpokenTime: Long = 0

    private fun speak(text: String) {
        if (!ttsReady || text.isBlank()) return
        if (!requestAudioFocus()) {
            Log.w(TAG, "Audio focus not granted; skipping TTS")
            return
        }
        
        val currentTime = System.currentTimeMillis()
        // Debounce: Avoid repeating the exact same text within 3 seconds
        if (text == lastSpokenText && (currentTime - lastSpokenTime) < 3000) {
            return
        }

        lastSpokenText = text
        lastSpokenTime = currentTime
        
        Log.d(TAG, "Speaking: $text")
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "rushdie_tts")
    }

    private fun requestAudioFocus(): Boolean {
        val manager = audioManager ?: return true
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val attributes = AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()
            val request = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                .setAudioAttributes(attributes)
                .setOnAudioFocusChangeListener { }
                .build()
            audioFocusRequest = request
            manager.requestAudioFocus(request) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
        } else {
            @Suppress("DEPRECATION")
            manager.requestAudioFocus(null, AudioManager.STREAM_MUSIC, AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK) ==
                AudioManager.AUDIOFOCUS_REQUEST_GRANTED
        }
    }

    private fun abandonAudioFocus() {
        val manager = audioManager ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            audioFocusRequest?.let { manager.abandonAudioFocusRequest(it) }
        } else {
            @Suppress("DEPRECATION")
            manager.abandonAudioFocus(null)
        }
    }

    // ─────────────────────── Engine Init ─────────────────────────────────────

    private fun initEngines() {
        CoroutineScope(Dispatchers.IO).launch {
            faceEngine = FaceEngine(this@MainActivity)
            faceReady = faceEngine.initialize()
            Log.i(TAG, "FaceEngine ready: $faceReady")

            currencyEngine = CurrencyEngine(this@MainActivity)
            currencyReady = currencyEngine.initialize()
            Log.i(TAG, "CurrencyEngine ready: $currencyReady")
            
            ocrEngine = OCREngine(this@MainActivity)
            ocrReady = ocrEngine.initialize()
            Log.i(TAG, "OCREngine ready: $ocrReady")

            objectDetectionEngine = ObjectDetectionEngine(this@MainActivity)
            objectDetectionReady = objectDetectionEngine.initialize()
            Log.i(TAG, "ObjectDetectionEngine ready: $objectDetectionReady")

            withContext(Dispatchers.Main) {
                sendModelEvent(mapOf(
                    "type" to "engines_ready",
                    "face" to faceReady,
                    "currency" to currencyReady,
                    "ocr" to ocrReady,
                    "objectDetection" to objectDetectionReady
                ))
            }
        }
    }

    private fun handleInitModels(result: MethodChannel.Result) {
        ensureVoskInitialized(startAfterReady = false)
        result.success(mapOf(
            "face" to faceReady,
            "currency" to currencyReady,
            "ocr" to ocrReady,
            "objectDetection" to objectDetectionReady
        ))
    }

    // ─────────────────────── Intent Listening ────────────────────────────────

    private fun handleStartIntentListening(result: MethodChannel.Result) {
        if (voskReady) {
            voskEngine?.startListening()
        } else {
            sendModelEvent(mapOf("type" to "vosk_status", "status" to "preparing_model"))
            ensureVoskInitialized(startAfterReady = true)
        }
        result.success(true)
    }

    private fun handleStopIntentListening(result: MethodChannel.Result) {
        startIntentAfterVoskReady = false
        voskEngine?.stopListening()
        result.success(true)
    }

    private fun ensureVoskInitialized(startAfterReady: Boolean) {
        if (voskReady) {
            if (startAfterReady) voskEngine?.startListening()
            return
        }

        if (startAfterReady) startIntentAfterVoskReady = true
        if (voskInitializing) return

        voskInitializing = true
        if (voskEngine == null) {
            voskEngine = VoskIntentEngine(this) { modelEventSink }
        }
        voskEngine?.initialize { ready ->
            voskInitializing = false
            voskReady = ready
            sendModelEvent(mapOf("type" to "vosk_ready", "ready" to ready))
            if (ready && startIntentAfterVoskReady) {
                startIntentAfterVoskReady = false
                voskEngine?.startListening()
            } else if (!ready) {
                startIntentAfterVoskReady = false
                sendModelEvent(mapOf("type" to "intent_error", "message" to "Vosk model not available"))
            }
        }
    }

    private fun handleSetCameraConfig(args: Any?, result: MethodChannel.Result) {
        val map = args as? Map<*, *>
        val lens = (map?.get("lensFacing") as? String)?.lowercase()
        val mirror = map?.get("mirror") as? Boolean

        cameraLensFacing = if (lens == "front") {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        if (mirror != null) cameraMirror = mirror

        if (hasCameraPermission()) {
            initCamera()
        }

        result.success(mapOf(
            "lensFacing" to if (cameraLensFacing == CameraSelector.LENS_FACING_FRONT) "front" else "back",
            "mirror" to cameraMirror
        ))
    }

    private fun handleGetCameraConfig(result: MethodChannel.Result) {
        result.success(mapOf(
            "lensFacing" to if (cameraLensFacing == CameraSelector.LENS_FACING_FRONT) "front" else "back",
            "mirror" to cameraMirror
        ))
    }

    // ─────────────────────── Camera + Inference ───────────────────────────────

    private fun handleSetCurrencyModelMode(args: Any?, result: MethodChannel.Result) {
        val map = args as? Map<*, *>
        val mode = (map?.get("mode") as? String)?.lowercase()
        val useYolo = map?.get("useYolo") as? Boolean
        useYoloCurrencyModel = useYolo ?: (mode == "yolo_v8" || mode == "yolov8" || mode == "yolo")

        Log.i(TAG, "Currency model mode set to ${currencyModelModeName()}")
        result.success(mapOf("mode" to currencyModelModeName()))
    }

    private fun handleGetCurrencyModelMode(result: MethodChannel.Result) {
        result.success(mapOf("mode" to currencyModelModeName()))
    }

    private fun selectedCurrencyMode(): CurrencyEngine.Mode {
        return if (useYoloCurrencyModel) CurrencyEngine.Mode.YOLO_V8 else CurrencyEngine.Mode.CLASSIC
    }

    private fun currencyModelModeName(): String {
        return if (useYoloCurrencyModel) "yolo_v8" else "classic"
    }

    private fun currencyModelModeName(mode: CurrencyEngine.Mode): String {
        return if (mode == CurrencyEngine.Mode.YOLO_V8) "yolo_v8" else "classic"
    }

    private fun handleStartCameraPreview(result: MethodChannel.Result) {
        if (!hasCameraPermission()) {
            requestCameraPermission(result, "preview")
            return
        }
        startPreviewStream()
        result.success(true)
    }

    private fun startPreviewStream() {
        cameraPreviewActive = true
        cameraPreviewFrozen = false
        lastPreviewFrameAt = 0L
        if (imageCapture == null || imageAnalysis == null) {
            initCamera()
        }
        sendModelEvent(mapOf("type" to "camera_live"))
        lastPreviewJpegBytes?.let { bytes ->
            sendModelEvent(mapOf("type" to "camera_preview", "imageBytes" to bytes))
        }
    }

    private fun handleStopCameraPreview(result: MethodChannel.Result) {
        objectDetectionActive = false
        objectDetectionInProgress = false
        cameraPreviewActive = false
        cameraPreviewFrozen = false
        if (pendingCameraIntent.isBlank()) {
            shutdownCamera()
        }
        result.success(true)
    }

    private fun handleStartObjectDetection(result: MethodChannel.Result) {
        if (!hasCameraPermission()) {
            requestCameraPermission(result, "object_live")
            return
        }
        startLiveObjectDetection()
        result.success(mapOf("active" to objectDetectionActive, "ready" to objectDetectionReady))
    }

    private fun handleStopObjectDetection(result: MethodChannel.Result) {
        stopLiveObjectDetection(speakStop = true)
        result.success(true)
    }

    private fun startLiveObjectDetection() {
        if (!objectDetectionReady || !::objectDetectionEngine.isInitialized) {
            objectDetectionActive = false
            val msg = "نموذج كشف العوائق غير جاهز"
            sendModelEvent(mapOf(
                "type" to "object_detection_status",
                "active" to false,
                "ready" to false,
                "message" to msg
            ))
            speak(msg)
            return
        }

        objectDetectionActive = true
        objectDetectionInProgress = false
        lastObjectDetectionAt = 0L
        startPreviewStream()
        val msg = "تم تشغيل تحذير العوائق"
        sendModelEvent(mapOf(
            "type" to "object_detection_status",
            "active" to true,
            "ready" to true,
            "message" to msg
        ))
        speak(msg)
    }

    private fun stopLiveObjectDetection(speakStop: Boolean = false) {
        val wasActive = objectDetectionActive
        objectDetectionActive = false
        objectDetectionInProgress = false
        sendModelEvent(mapOf(
            "type" to "object_detection_status",
            "active" to false,
            "ready" to objectDetectionReady,
            "message" to "تم إيقاف تحذير العوائق"
        ))
        if (speakStop && wasActive) speak("تم إيقاف تحذير العوائق")
    }

    private fun handlePreviewFrame(image: ImageProxy) {
        try {
            if (!cameraPreviewActive || cameraPreviewFrozen) return
            val now = System.currentTimeMillis()
            if (now - lastPreviewFrameAt < previewFrameIntervalMs) return
            lastPreviewFrameAt = now
            val bytes = imageProxyToPreviewJpegBytes(image, cameraMirror) ?: return
            lastPreviewJpegBytes = bytes
            sendModelEvent(mapOf(
                "type" to "camera_preview",
                "imageBytes" to bytes
            ))
            maybeRunObjectDetection(bytes, now)
        } catch (e: Exception) {
            Log.w(TAG, "Preview frame failed", e)
        } finally {
            image.close()
        }
    }

    private fun maybeRunObjectDetection(frameBytes: ByteArray, now: Long) {
        if (!objectDetectionActive || !objectDetectionReady || !::objectDetectionEngine.isInitialized) return
        if (objectDetectionInProgress) return
        if (now - lastObjectDetectionAt < objectDetectionIntervalMs) return

        lastObjectDetectionAt = now
        objectDetectionInProgress = true
        val bytesForInference = frameBytes.copyOf()

        CoroutineScope(Dispatchers.IO).launch {
            val detectionResult = objectDetectionEngine.detect(bytesForInference, now)
            withContext(Dispatchers.Main) {
                objectDetectionInProgress = false
                if (!objectDetectionActive) return@withContext

                val message = detectionResult.messageAr
                sendModelEvent(mapOf(
                    "type" to "object_detection_result",
                    "active" to true,
                    "ready" to detectionResult.ready,
                    "shouldSpeak" to detectionResult.shouldSpeak,
                    "messageAr" to message,
                    "messageEn" to detectionResult.messageEn,
                    "mainObject" to detectionResult.mainObject?.let { objectDetectionToMap(it) },
                    "detections" to detectionResult.allDetections.map { objectDetectionToMap(it) }
                ))

                if (detectionResult.shouldSpeak && message.isNotBlank()) {
                    speak(message)
                }
            }
        }
    }

    private fun objectDetectionToMap(det: ObjectDetectionEngine.AnalyzedDetection): Map<String, Any> {
        return mapOf(
            "className" to det.className,
            "confidence" to det.confidence,
            "distanceHint" to det.distanceHint,
            "horizontalPosition" to det.horizontalPosition,
            "areaRatio" to det.areaRatio,
            "heightRatio" to det.heightRatio,
            "bbox" to det.box.map { it.toDouble() }
        )
    }

    private fun handleCaptureFace(result: MethodChannel.Result) {
        if (!hasCameraPermission()) {
            requestCameraPermission(result, "face")
            return
        }
        pendingCameraResult = result
        pendingCameraIntent = "face"
        startPreviewStream()
        launchCamera()
    }

    private fun handleCaptureCurrency(result: MethodChannel.Result) {
        if (!hasCameraPermission()) {
            requestCameraPermission(result, "currency")
            return
        }
        pendingCameraResult = result
        pendingCameraIntent = "currency"
        startPreviewStream()
        launchCamera()
    }

    private fun launchCamera() {
        if (cameraCaptureInProgress) {
            Log.w(TAG, "Photo capture requested while another capture is still running")
            mainHandler.postDelayed({
                if (pendingCameraIntent.isNotBlank()) launchCamera()
            }, 350L)
            return
        }
        if (cameraPreviewActive && imageAnalysis == null) {
            initCamera()
        }
        if (imageCapture == null) {
            initCamera()
            mainHandler.postDelayed({ launchCamera() }, 500)
            return
        }

        cameraCaptureInProgress = true
        imageCapture?.takePicture(ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val bytes = try {
                    imageProxyToJpegBytes(image, cameraMirror)
                } catch (e: Exception) {
                    Log.e(TAG, "Image conversion failed", e)
                    null
                } finally {
                    image.close()
                    cameraCaptureInProgress = false
                } ?: run {
                    val fallback = lastPreviewJpegBytes
                    if (fallback == null) {
                        handleCameraCaptureFailure("تعذر التقاط الصورة")
                        return
                    }
                    fallback
                }

                cameraPreviewFrozen = cameraPreviewActive
                lastPreviewJpegBytes = bytes
                sendModelEvent(mapOf(
                    "type" to "camera_frozen",
                    "imageBytes" to bytes
                ))
                if (!cameraPreviewActive) {
                    shutdownCamera()
                }
                
                when {
                    pendingCameraIntent == "face"     -> runFaceInference(bytes)
                    pendingCameraIntent == "currency" -> runCurrencyInference(bytes)
                    pendingCameraIntent == "ocr"      -> runOCRInference(bytes)
                    pendingCameraIntent == "enroll"   -> runEnrollCapture(bytes)
                }
            }

            override fun onError(exception: ImageCaptureException) {
                cameraCaptureInProgress = false
                Log.e(TAG, "Photo capture failed: ${exception.message}", exception)
                handleCameraCaptureFailure(exception.message ?: "Photo capture failed")
            }
        })
    }

    private fun handleCameraCaptureFailure(message: String) {
        if (!cameraPreviewActive) {
            shutdownCamera()
        } else {
            cameraPreviewFrozen = false
            sendModelEvent(mapOf("type" to "camera_live"))
            lastPreviewJpegBytes?.let { bytes ->
                sendModelEvent(mapOf("type" to "camera_preview", "imageBytes" to bytes))
            }
        }
        val result = pendingCameraResult
        pendingCameraResult = null
        pendingCameraIntent = ""
        result?.error("CAMERA_ERROR", message, null)
    }

    private fun shutdownCamera() {
        try {
            val cameraProvider = ProcessCameraProvider.getInstance(this).get()
            imageAnalysis?.clearAnalyzer()
            cameraProvider.unbindAll()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to shutdown camera", e)
        } finally {
            imageCapture = null
            imageAnalysis = null
            cameraCaptureInProgress = false
            cameraInitInProgress = false
            cameraRebindRequested = false
        }
    }

    private fun imageProxyToJpegBytes(image: ImageProxy, mirror: Boolean): ByteArray? {
        val bitmap = imageProxyToBitmap(image) ?: return null
        val finalBitmap = if (mirror) mirrorBitmap(bitmap) else bitmap
        return try {
            val out = ByteArrayOutputStream()
            finalBitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
            out.toByteArray()
        } finally {
            if (finalBitmap !== bitmap) finalBitmap.recycle()
            bitmap.recycle()
        }
    }

    private fun imageProxyToPreviewJpegBytes(image: ImageProxy, mirror: Boolean): ByteArray? {
        val bitmap = imageProxyToBitmap(image) ?: return null
        val mirrored = if (mirror) mirrorBitmap(bitmap) else bitmap
        val maxWidth = 640
        val scale = if (mirrored.width > maxWidth) maxWidth.toFloat() / mirrored.width.toFloat() else 1f
        val previewBitmap = if (scale < 1f) {
            Bitmap.createScaledBitmap(
                mirrored,
                maxOf(1, (mirrored.width * scale).toInt()),
                maxOf(1, (mirrored.height * scale).toInt()),
                true
            )
        } else {
            mirrored
        }
        return try {
            val out = ByteArrayOutputStream()
            previewBitmap.compress(Bitmap.CompressFormat.JPEG, 65, out)
            out.toByteArray()
        } finally {
            if (previewBitmap !== mirrored) previewBitmap.recycle()
            if (mirrored !== bitmap) mirrored.recycle()
            bitmap.recycle()
        }
    }

    private fun releaseCameraFreeze(delayMs: Long = 1800L) {
        if (!cameraPreviewActive) return
        mainHandler.postDelayed({
            if (cameraPreviewActive) {
                cameraPreviewFrozen = false
                sendModelEvent(mapOf("type" to "camera_live"))
                lastPreviewJpegBytes?.let { bytes ->
                    sendModelEvent(mapOf("type" to "camera_preview", "imageBytes" to bytes))
                }
            }
        }, delayMs)
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        if (image.planes.size < 3) {
            val buffer = image.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
            val rotated = rotateBitmapIfNeeded(bmp, image.imageInfo.rotationDegrees)
            if (rotated !== bmp) bmp.recycle()
            return rotated
        }

        val nv21 = yuv420ToNv21(image)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        if (!yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)) {
            return null
        }
        val jpegBytes = out.toByteArray()
        val bmp = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size) ?: return null
        val rotated = rotateBitmapIfNeeded(bmp, image.imageInfo.rotationDegrees)
        if (rotated !== bmp) bmp.recycle()
        return rotated
    }

    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val nv21 = ByteArray(ySize + 2 * (width / 2) * (height / 2))

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        var outputOffset = 0
        for (row in 0 until height) {
            val rowOffset = row * yPlane.rowStride
            for (col in 0 until width) {
                nv21[outputOffset++] = yBuffer.get(rowOffset + col * yPlane.pixelStride)
            }
        }

        val chromaWidth = width / 2
        val chromaHeight = height / 2
        var chromaOffset = ySize
        for (row in 0 until chromaHeight) {
            val uRowOffset = row * uPlane.rowStride
            val vRowOffset = row * vPlane.rowStride
            for (col in 0 until chromaWidth) {
                nv21[chromaOffset++] = vBuffer.get(vRowOffset + col * vPlane.pixelStride)
                nv21[chromaOffset++] = uBuffer.get(uRowOffset + col * uPlane.pixelStride)
            }
        }
        return nv21
    }

    private fun mirrorBitmap(bitmap: Bitmap): Bitmap {
        val matrix = Matrix().apply { preScale(-1f, 1f) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun rotateBitmapIfNeeded(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    private fun runCurrencyInference(imageBytes: ByteArray) {
        val result = pendingCameraResult
        pendingCameraResult = null
        pendingCameraIntent = ""

        CoroutineScope(Dispatchers.IO).launch {
            val currencyMode = selectedCurrencyMode()
            val currencyModeName = currencyModelModeName(currencyMode)
            val currResult = if (currencyReady) {
                currencyEngine.detect(imageBytes, currencyMode)
            } else {
                CurrencyEngine.CurrencyResult(emptyList(), 0, "\u0645\u062d\u0631\u0643 \u0627\u0644\u0639\u0645\u0644\u0629 \u063a\u064a\u0631 \u062c\u0627\u0647\u0632")
            }

            withContext(Dispatchers.Main) {
                speak(currResult.arabicName)
                sendModelEvent(mapOf(
                    "type"       to "currency_result",
                    "arabicName" to currResult.arabicName,
                    "total"      to currResult.total,
                    "ttsText"    to currResult.arabicName,
                    "model"      to currencyModeName,
                    "imageBytes" to imageBytes
                ))
                result?.success(mapOf(
                    "arabicName" to currResult.arabicName,
                    "total"      to currResult.total,
                    "ttsText"    to currResult.arabicName,
                    "model"      to currencyModeName
                ))
                releaseCameraFreeze()
            }
        }
    }

    private fun runOCRInference(imageBytes: ByteArray) {
        val result = pendingCameraResult
        pendingCameraResult = null
        pendingCameraIntent = ""

        CoroutineScope(Dispatchers.IO).launch {
            val text = if (ocrReady) {
                ocrEngine.recognizeText(imageBytes)
            } else "محرك القراءة غير جاهز"

            withContext(Dispatchers.Main) {
                speak(text)
                sendModelEvent(mapOf(
                    "type"    to "ocr_result",
                    "text"    to text,
                    "ttsText" to text,
                    "imageBytes" to imageBytes
                ))
                result?.success(mapOf("text" to text))
                releaseCameraFreeze()
            }
        }
    }

    private fun handleCaptureOCR(result: MethodChannel.Result) {
        if (!hasCameraPermission()) {
            requestCameraPermission(result, "ocr")
            return
        }
        pendingCameraResult = result
        pendingCameraIntent = "ocr"
        startPreviewStream()
        launchCamera()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        // Kept for other intents if needed
    }

    private fun runFaceInference(imageBytes: ByteArray) {
        val result = pendingCameraResult
        pendingCameraResult = null
        pendingCameraIntent = ""

        CoroutineScope(Dispatchers.IO).launch {
            val faceResult = if (faceReady) {
                faceEngine.recognize(imageBytes)
            } else null

            withContext(Dispatchers.Main) {
                if (faceResult == null) {
                    val msg = "نموذج التعرف على الوجوه غير جاهز"
                    speak(msg)
                    result?.success(mapOf("error" to msg))
                    releaseCameraFreeze()
                    return@withContext
                }

                val ttsText = when {
                    faceResult.name == FaceEngine.NO_FACE_LABEL -> "تعذر الكشف عن أشخاص"
                    faceResult.recognized -> "ده ${faceResult.name}"
                    faceResult.name == "لا يوجد أشخاص مسجلين" -> "لا يوجد أشخاص مسجلين"
                    else -> "شخص غير مسجل"
                }
                speak(ttsText)

                val displayName = if (faceResult.name == FaceEngine.NO_FACE_LABEL) "غير محدد" else faceResult.name

                sendModelEvent(mapOf(
                    "type"       to "face_result",
                    "name"       to displayName,
                    "similarity" to faceResult.similarity,
                    "recognized" to faceResult.recognized,
                    "ttsText"    to ttsText,
                    "imageBytes" to imageBytes
                ))
                result?.success(mapOf(
                    "name"       to displayName,
                    "similarity" to faceResult.similarity,
                    "recognized" to faceResult.recognized,
                    "ttsText"    to ttsText
                ))
                releaseCameraFreeze()
            }
        }
    }


    private fun sendCurrencyCapture(imageBytes: ByteArray) {
        val result = pendingCameraResult
        pendingCameraResult = null
        sendModelEvent(mapOf(
            "type" to "currency_image",
            "imageBytes" to imageBytes
        ))
        result?.success(true)
    }

    // ─────────────────────── Enrollment ──────────────────────────────────────

    private fun handleEnrollPerson(args: Any?, result: MethodChannel.Result) {
        val map = args as? Map<*, *>
        val name = map?.get("name") as? String
        if (name.isNullOrBlank()) {
            result.error("INVALID_ARGS", "Name is required", null)
            return
        }
        if (!faceReady) {
            val msg = "\u0646\u0645\u0648\u0630\u062c \u0627\u0644\u062a\u0639\u0631\u0641 \u0639\u0644\u0649 \u0627\u0644\u0648\u062c\u0648\u0647 \u063a\u064a\u0631 \u062c\u0627\u0647\u0632"
            speak(msg)
            result.error("FACE_NOT_READY", msg, null)
            return
        }
        if (!hasCameraPermission()) {
            requestCameraPermission(result, "enroll:$name")
            return
        }
        pendingEnrollName = name
        enrollEmbeddings.clear()
        enrollCaptureAttempts = 0
        pendingCameraResult = result
        pendingCameraIntent = "enroll"
        startPreviewStream()
        sendModelEvent(mapOf("type" to "enroll_start", "name" to name, "photosNeeded" to enrollPhotosNeeded))
        speak("\u062b\u0628\u062a \u0648\u0634 \u0627\u0644\u0634\u062e\u0635 \u0642\u062f\u0627\u0645 \u0627\u0644\u0643\u0627\u0645\u064a\u0631\u0627")
        launchCamera()
    }

    private fun runEnrollCapture(imageBytes: ByteArray) {
        enrollCaptureAttempts++
        CoroutineScope(Dispatchers.IO).launch {
            val embedding = faceEngine.extractEmbedding(imageBytes)
            withContext(Dispatchers.Main) {
                if (embedding != null) {
                    enrollEmbeddings.add(embedding)
                    val taken = enrollEmbeddings.size
                    sendModelEvent(mapOf(
                        "type"     to "enroll_progress",
                        "taken"    to taken,
                        "needed"   to enrollPhotosNeeded,
                        "attempts" to enrollCaptureAttempts,
                        "clear"    to true
                    ))
                    speak("\u062a\u0645 \u0627\u0644\u062a\u0642\u0627\u0637 \u0635\u0648\u0631\u0629 \u0648\u0627\u0636\u062d\u0629 $taken \u0645\u0646 $enrollPhotosNeeded")
                } else {
                    sendModelEvent(mapOf(
                        "type"     to "enroll_progress",
                        "taken"    to enrollEmbeddings.size,
                        "needed"   to enrollPhotosNeeded,
                        "attempts" to enrollCaptureAttempts,
                        "clear"    to false
                    ))
                    if (enrollCaptureAttempts == 1 || enrollCaptureAttempts % 2 == 0) {
                        speak("\u0645\u0634 \u0634\u0627\u064a\u0641 \u0648\u0634 \u0648\u0627\u0636\u062d. \u0642\u0631\u0628 \u0627\u0644\u0648\u0634 \u0648\u062b\u0628\u062a\u0647 \u0642\u062f\u0627\u0645 \u0627\u0644\u0643\u0627\u0645\u064a\u0631\u0627")
                    }
                }

                when {
                    enrollEmbeddings.size >= enrollPhotosNeeded -> finishEnrollment(success = true)
                    enrollCaptureAttempts >= enrollMaxAttempts -> finishEnrollment(
                        success = false,
                        userMessage = "\u0645\u0634 \u0634\u0627\u064a\u0641 \u0648\u0634 \u0648\u0627\u0636\u062d. \u0642\u0631\u0628 \u0627\u0644\u0648\u0634 \u0648\u062e\u0644\u064a\u0647 \u0641\u064a \u0625\u0636\u0627\u0621\u0629 \u0643\u0648\u064a\u0633\u0629.",
                        errorMessage = "Could not capture enough clear face embeddings"
                    )
                    else -> {
                        releaseCameraFreeze(delayMs = 350L)
                        mainHandler.postDelayed({ launchCamera() }, enrollCaptureDelayMs)
                    }
                }
            }
        }
    }

    /*
        enrollImageBytes.add(imageBytes)
        enrollPhotosTaken++
        sendModelEvent(mapOf(
            "type"   to "enroll_progress",
            "taken"  to enrollPhotosTaken,
            "needed" to enrollPhotosNeeded
        ))

        if (enrollPhotosTaken < enrollPhotosNeeded) {
            // Take another photo
            launchCamera()
        } else {
            // Enroll with all collected images
            val result = pendingCameraResult
            pendingCameraResult = null
            val name = pendingEnrollName
            val images = enrollImageBytes.toList()
            enrollImageBytes.clear()

            CoroutineScope(Dispatchers.IO).launch {
                val ok = faceEngine.enrollPerson(name, images)
                withContext(Dispatchers.Main) {
                    if (ok) {
                        speak("تم تسجيل $name بنجاح")
                        sendModelEvent(mapOf("type" to "enroll_done", "name" to name, "success" to true))
                        result?.success(mapOf("success" to true, "name" to name))
                    } else {
                        speak("فشل تسجيل $name")
                        sendModelEvent(mapOf("type" to "enroll_done", "name" to name, "success" to false))
                        result?.error("ENROLL_FAILED", "Could not extract embeddings", null)
                    }
                }
            }
        }
    }

    */

    private fun finishEnrollment(
        success: Boolean,
        userMessage: String? = null,
        errorMessage: String? = null
    ) {
        val result = pendingCameraResult
        pendingCameraResult = null
        pendingCameraIntent = ""
        val name = pendingEnrollName
        val embeddings = enrollEmbeddings.toList()
        enrollEmbeddings.clear()
        enrollCaptureAttempts = 0

        if (!success) {
            val msg = userMessage ?: "\u0641\u0634\u0644 \u062a\u0633\u062c\u064a\u0644 $name"
            speak(msg)
            sendModelEvent(mapOf("type" to "enroll_done", "name" to name, "success" to false, "message" to msg))
            releaseCameraFreeze()
            result?.error("ENROLL_FAILED", errorMessage ?: msg, null)
            return
        }

        CoroutineScope(Dispatchers.IO).launch {
            val ok = faceEngine.enrollPersonFromEmbeddings(name, embeddings)
            withContext(Dispatchers.Main) {
                if (ok) {
                    val msg = "\u062a\u0645 \u062a\u0633\u062c\u064a\u0644 $name \u0628\u0646\u062c\u0627\u062d"
                    speak(msg)
                    sendModelEvent(mapOf("type" to "enroll_done", "name" to name, "success" to true, "message" to msg))
                    releaseCameraFreeze()
                    result?.success(mapOf("success" to true, "name" to name))
                } else {
                    val msg = "\u0641\u0634\u0644 \u062a\u0633\u062c\u064a\u0644 $name"
                    speak(msg)
                    sendModelEvent(mapOf("type" to "enroll_done", "name" to name, "success" to false, "message" to msg))
                    releaseCameraFreeze()
                    result?.error("ENROLL_FAILED", "Could not save enrollment", null)
                }
            }
        }
    }

    private fun handleDeletePerson(args: Any?, result: MethodChannel.Result) {
        val name = args as? String
        if (name.isNullOrBlank()) { result.error("INVALID_ARGS", "Name required", null); return }
        val ok = faceEngine.deleteEnrolledPerson(name)
        result.success(ok)
    }

    private fun handleListPersons(result: MethodChannel.Result) {
        val persons = if (faceReady) faceEngine.listEnrolledPersons() else emptyList()
        result.success(persons)
    }

    // ─────────────────────── Permissions ─────────────────────────────────────

    private fun hasMicrophonePermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private var currentEngine: String = "pytorch"
    private fun startListeningWithPermission(result: MethodChannel.Result, engine: String) {
        this.currentEngine = engine
        if (hasMicrophonePermission()) { startWakeWordService(currentEngine); result.success(true); return }
        if (pendingStartResult != null) {
            result.error("MIC_PERMISSION_PENDING", "Already requesting permission", null); return
        }
        pendingStartResult = result
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
    }

    private fun requestCameraPermission(result: MethodChannel.Result, intent: String) {
        pendingCameraResult = result
        pendingCameraIntent = intent
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA_PERMISSION)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        val granted = grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED

        when (requestCode) {
            REQUEST_RECORD_AUDIO -> {
                val result = pendingStartResult; pendingStartResult = null
                if (granted) { startWakeWordService(currentEngine); result?.success(true) }
                else result?.error("MIC_PERMISSION_DENIED", "Microphone permission denied", null)
            }
            REQUEST_CAMERA_PERMISSION -> {
                if (granted) {
                    initCamera()
                    when {
                        pendingCameraIntent == "preview"  -> {
                            cameraPreviewActive = true
                            cameraPreviewFrozen = false
                            pendingCameraIntent = ""
                            val result = pendingCameraResult
                            pendingCameraResult = null
                            initCamera()
                            sendModelEvent(mapOf("type" to "camera_live"))
                            result?.success(true)
                        }
                        pendingCameraIntent == "face"     -> launchCamera()
                        pendingCameraIntent == "currency" -> launchCamera()
                        pendingCameraIntent == "ocr"      -> launchCamera()
                        pendingCameraIntent == "object_live" -> {
                            pendingCameraIntent = ""
                            val result = pendingCameraResult
                            pendingCameraResult = null
                            startLiveObjectDetection()
                            result?.success(mapOf("active" to objectDetectionActive, "ready" to objectDetectionReady))
                        }
                        pendingCameraIntent.startsWith("enroll:") -> {
                            pendingEnrollName = pendingCameraIntent.substringAfter("enroll:")
                            enrollEmbeddings.clear()
                            enrollCaptureAttempts = 0
                            pendingCameraIntent = "enroll"
                            startPreviewStream()
                            sendModelEvent(mapOf("type" to "enroll_start", "name" to pendingEnrollName, "photosNeeded" to enrollPhotosNeeded))
                            launchCamera()
                        }
                    }
                } else {
                    val result = pendingCameraResult; pendingCameraResult = null
                    pendingCameraIntent = ""
                    result?.error("CAMERA_PERMISSION_DENIED", "Camera permission denied", null)
                }
            }
        }
    }

    // ─────────────────────── Wake Word Service ────────────────────────────────

    private fun startWakeWordService(engine: String = currentEngine) {
        WakeWordService.setPendingEventSink(wakeWordEventSink)
        val intent = Intent(this, WakeWordService::class.java).apply {
            action = WakeWordService.ACTION_START_LISTENING
            putExtra(WakeWordService.EXTRA_ENGINE_TYPE, engine)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
        WakeWordService.getInstance()?.setEventSink(wakeWordEventSink)
    }

    private fun stopWakeWordService() {
        val intent = Intent(this, WakeWordService::class.java).apply {
            action = WakeWordService.ACTION_STOP_LISTENING
        }
        startService(intent)
    }

    // ─────────────────────── Helpers ─────────────────────────────────────────

    private fun sendModelEvent(event: Map<String, Any?>) {
        mainHandler.post {
            try {
                modelEventSink?.success(event)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to send model event ${event["type"]}", e)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        objectDetectionActive = false
        tts?.shutdown()
        if (::faceEngine.isInitialized) faceEngine.release()
        if (::currencyEngine.isInitialized) currencyEngine.release()
        if (::ocrEngine.isInitialized) ocrEngine.release()
        if (::objectDetectionEngine.isInitialized) objectDetectionEngine.release()
        voskEngine?.release()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val ACTION_TEXT_TO_SPEECH_SETTINGS = "com.android.settings.TTS_SETTINGS"
        private const val REQUEST_RECORD_AUDIO      = 1101
        private const val REQUEST_CAMERA            = 1102
        private const val REQUEST_CAMERA_PERMISSION = 1103
    }
}
