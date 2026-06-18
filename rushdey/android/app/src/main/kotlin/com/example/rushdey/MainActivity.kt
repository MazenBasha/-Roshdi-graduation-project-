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
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
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
    private var voskEngine: VoskIntentEngine? = null
    private var voskReady = false
    private var voskInitializing = false
    private var startIntentAfterVoskReady = false
    private var faceReady = false
    private var currencyReady = false
    private var ocrReady = false

    // TTS
    private var tts: TextToSpeech? = null
    private var ttsReady = false
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
                    "startCameraPreview"    -> handleStartCameraPreview(result)
                    "stopCameraPreview"     -> handleStopCameraPreview(result)
                    "captureAndRecognizeFace"     -> handleCaptureFace(result)
                    "captureAndDetectCurrency"    -> handleCaptureCurrency(result)
                    "captureAndReadText"          -> handleCaptureOCR(result)
                    "enrollPerson"          -> handleEnrollPerson(call.arguments, result)
                    "deleteEnrolledPerson"  -> handleDeletePerson(call.arguments, result)
                    "listEnrolledPersons"   -> handleListPersons(result)
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
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
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
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ─────────────────────── TTS ──────────────────────────────────────────────

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val locale = Locale("ar", "EG")
            
            // Prefer stable offline voice for Egyptian Arabic
            val voices = tts?.voices
            val bestVoice = voices?.find {
                it.locale.language == "ar" &&
                (it.locale.country == "EG" || it.name.contains("eg", ignoreCase = true)) &&
                !it.isNetworkConnectionRequired
            } ?: voices?.find {
                it.locale.language == "ar" && !it.isNetworkConnectionRequired
            } ?: voices?.find { it.locale.language == "ar" }

            if (bestVoice != null) {
                tts?.voice = bestVoice
                Log.i(TAG, "Selected offline voice: ${bestVoice.name}")
            } else {
                tts?.setLanguage(locale)
            }
            
            ttsReady = true
            tts?.setPitch(1.0f)
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

            withContext(Dispatchers.Main) {
                sendModelEvent(mapOf(
                    "type" to "engines_ready",
                    "face" to faceReady,
                    "currency" to currencyReady,
                    "ocr" to ocrReady
                ))
            }
        }
    }

    private fun handleInitModels(result: MethodChannel.Result) {
        ensureVoskInitialized(startAfterReady = false)
        result.success(mapOf("face" to faceReady, "currency" to currencyReady, "ocr" to ocrReady))
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
        cameraPreviewActive = false
        cameraPreviewFrozen = false
        if (pendingCameraIntent.isBlank()) {
            shutdownCamera()
        }
        result.success(true)
    }

    private fun handlePreviewFrame(image: ImageProxy) {
        try {
            if (!cameraPreviewActive || cameraPreviewFrozen || modelEventSink == null) return
            val now = System.currentTimeMillis()
            if (now - lastPreviewFrameAt < previewFrameIntervalMs) return
            lastPreviewFrameAt = now
            val bytes = imageProxyToPreviewJpegBytes(image, cameraMirror) ?: return
            lastPreviewJpegBytes = bytes
            sendModelEvent(mapOf(
                "type" to "camera_preview",
                "imageBytes" to bytes
            ))
        } catch (e: Exception) {
            Log.w(TAG, "Preview frame failed", e)
        } finally {
            image.close()
        }
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
            cameraProvider.unbindAll()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to shutdown camera", e)
        } finally {
            imageCapture = null
            imageAnalysis = null
            cameraCaptureInProgress = false
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
            val currResult = if (currencyReady) {
                currencyEngine.detect(imageBytes)
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
                    "imageBytes" to imageBytes
                ))
                result?.success(mapOf(
                    "arabicName" to currResult.arabicName,
                    "total"      to currResult.total,
                    "ttsText"    to currResult.arabicName
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

    private fun sendModelEvent(event: Map<String, Any>) {
        mainHandler.post { modelEventSink?.success(event) }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts?.shutdown()
        if (::faceEngine.isInitialized) faceEngine.release()
        if (::currencyEngine.isInitialized) currencyEngine.release()
        if (::ocrEngine.isInitialized) ocrEngine.release()
        voskEngine?.release()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_RECORD_AUDIO      = 1101
        private const val REQUEST_CAMERA            = 1102
        private const val REQUEST_CAMERA_PERMISSION = 1103
    }
}
