package com.example.rushdey

import android.content.Context
import android.util.Log
import io.flutter.FlutterInjector
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Runtime for the from-scratch wake-word model trained in wake_word_final.ipynb.
 */
class WakeWordModelEngine(
    private val context: Context,
    private val onEvent: (Map<String, Any>) -> Unit,
    private val onDetected: (Float) -> Unit,
) {
    private var module: Module? = null
    private var recorder: AudioRecorder? = null
    private val melSpectrogram = MelSpectrogram()
    private val isProcessing = AtomicBoolean(false)
    private var isRunning = false
    private var consecutiveHits = 0
    private var lastTelemetryMs = 0L
    private var lastConfidenceLogMs = 0L
    private var lastConfidence = 0f

    fun start(): Boolean {
        if (isRunning) return true

        return try {
            ensureModelLoaded()
            consecutiveHits = 0
            lastTelemetryMs = 0L
            lastConfidence = 0f
            recorder = AudioRecorder { audioWindow -> processAudioWindow(audioWindow) }
            val started = recorder?.startRecording() == true
            isRunning = started
            if (!started) {
                onEvent(mapOf("type" to "error", "message" to "تعذر تشغيل الميكروفون"))
            }
            started
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start PyTorch wake-word engine", e)
            onEvent(mapOf("type" to "error", "message" to (e.message ?: "Wake-word model failed")))
            false
        }
    }

    fun stop() {
        recorder?.stopRecording()
        recorder = null
        isRunning = false
        consecutiveHits = 0
        isProcessing.set(false)
    }

    fun release() {
        stop()
        module?.destroy()
        module = null
    }

    private fun ensureModelLoaded() {
        if (module != null) return
        val modelFile = copyFlutterAssetToFile("assets/models/wake_word_trial2.ptl")
        module = Module.load(modelFile.absolutePath)
        Log.i(TAG, "Wake-word model loaded from ${modelFile.absolutePath}")
    }

    private fun processAudioWindow(audioWindow: FloatArray) {
        if (!isRunning || !isProcessing.compareAndSet(false, true)) return

        try {
            val normalized = normalizeAudio(audioWindow)
            val confidence = runInference(normalized)
            val windowRms = rms(audioWindow)
            lastConfidence = confidence

            val now = System.currentTimeMillis()
            if (now - lastTelemetryMs >= 250L) {
                lastTelemetryMs = now
                onEvent(mapOf("type" to "confidence", "confidence" to confidence))
                onEvent(mapOf("type" to "mic_level", "level" to windowRms))
            }
            if (now - lastConfidenceLogMs >= 1000L) {
                lastConfidenceLogMs = now
                Log.d(TAG, "Wake confidence=$confidence rms=$windowRms")
            }

            if (confidence >= DETECTION_THRESHOLD && windowRms >= MIN_DETECTION_RMS) {
                consecutiveHits += 1
            } else {
                consecutiveHits = 0
            }

            if (consecutiveHits >= REQUIRED_CONSECUTIVE_HITS) {
                isRunning = false
                onDetected(confidence)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Wake-word inference failed", e)
            consecutiveHits = 0
        } finally {
            isProcessing.set(false)
        }
    }

    private fun runInference(audioWindow: FloatArray): Float {
        val localModule = module ?: return 0f
        val features = melSpectrogram.compute(audioWindow)
        val input = FloatArray(N_MELS * N_FRAMES)

        for (mel in 0 until N_MELS) {
            val row = features.getOrNull(mel)
            for (frame in 0 until N_FRAMES) {
                input[mel * N_FRAMES + frame] = row?.getOrNull(frame) ?: 0f
            }
        }

        val inputTensor = Tensor.fromBlob(
            input,
            longArrayOf(1, 1, N_MELS.toLong(), N_FRAMES.toLong())
        )
        val output = localModule.forward(IValue.from(inputTensor)).toTensor()
        val logits = output.dataAsFloatArray
        if (logits.size < 2) return 0f
        val probs = softmax2(logits[0], logits[1])
        return probs.second
    }

    private fun normalizeAudio(audio: FloatArray): FloatArray {
        var maxAbs = 0f
        audio.forEach { sample -> maxAbs = maxOf(maxAbs, abs(sample)) }
        if (maxAbs <= 1e-6f) return audio.copyOf()
        return FloatArray(audio.size) { i -> audio[i] / maxAbs }
    }

    private fun softmax2(a: Float, b: Float): Pair<Float, Float> {
        val max = maxOf(a, b)
        val ea = exp((a - max).toDouble()).toFloat()
        val eb = exp((b - max).toDouble()).toFloat()
        val sum = (ea + eb).coerceAtLeast(1e-6f)
        return Pair(ea / sum, eb / sum)
    }

    private fun rms(samples: FloatArray): Float {
        if (samples.isEmpty()) return 0f
        var sum = 0.0
        samples.forEach { sample -> sum += sample * sample }
        return sqrt(sum / samples.size).toFloat()
    }

    private fun copyFlutterAssetToFile(assetName: String): File {
        val outFile = File(context.filesDir, assetName)
        if (outFile.exists() && outFile.length() > 0) return outFile
        outFile.parentFile?.mkdirs()

        val key = try {
            FlutterInjector.instance().flutterLoader().getLookupKeyForAsset(assetName)
        } catch (_: Exception) {
            "flutter_assets/$assetName"
        }

        context.assets.open(key).use { input ->
            FileOutputStream(outFile).use { output -> input.copyTo(output) }
        }
        return outFile
    }

    companion object {
        private const val TAG = "WakeWordModelEngine"
        private const val N_MELS = 64
        private const val N_FRAMES = 32
        private const val DETECTION_THRESHOLD = 0.50f
        private const val MIN_DETECTION_RMS = 0.025f
        private const val REQUIRED_CONSECUTIVE_HITS = 1
    }
}
