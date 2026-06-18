package com.example.rushdey

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.*
import kotlin.math.min

/**
 * AudioRecorder handles microphone capture with sliding window processing
 * Captures 16kHz mono PCM audio and maintains a rolling 1-second buffer
 */
class AudioRecorder(
    private val onAudioReady: (FloatArray) -> Unit
) {
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingJob: Job? = null
    
    // Audio configuration
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    
    // Buffer configuration
    private val windowSizeSeconds = 1.0f
    private val slideIntervalMs = 100L
    private val windowSize = (sampleRate * windowSizeSeconds).toInt()
    
    private val rollingBuffer = FloatArray(windowSize)
    private var bufferPosition = 0
    
    /**
     * Start audio recording
     */
    fun startRecording(): Boolean {
        if (isRecording) {
            Log.w(TAG, "Already recording")
            return false
        }
        
        return try {
            val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
            
            if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
                Log.e(TAG, "Invalid buffer size: $bufferSize")
                return false
            }
            
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                sampleRate,
                channelConfig,
                audioFormat,
                bufferSize * 2
            )
            
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord not initialized")
                return false
            }
            
            audioRecord?.startRecording()
            isRecording = true
            
            // Start recording coroutine
            recordingJob = CoroutineScope(Dispatchers.IO).launch {
                recordAudio()
            }
            
            Log.i(TAG, "Recording started")
            true
            
        } catch (e: SecurityException) {
            Log.e(TAG, "Missing RECORD_AUDIO permission", e)
            false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recording", e)
            false
        }
    }
    
    /**
     * Stop audio recording
     */
    fun stopRecording() {
        if (!isRecording) return
        
        isRecording = false
        recordingJob?.cancel()
        
        try {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping recording", e)
        }
        
        // Clear buffer
        rollingBuffer.fill(0f)
        bufferPosition = 0
        
        Log.i(TAG, "Recording stopped")
    }
    
    /**
     * Main recording loop with sliding window processing
     */
    private suspend fun recordAudio() {
        val readBufferSize = (sampleRate * slideIntervalMs / 1000).toInt()
        val shortBuffer = ShortArray(readBufferSize)
        
        var lastProcessTime = System.currentTimeMillis()
        var lastLevelLogTime = System.currentTimeMillis()
        
        while (isRecording && audioRecord?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
            // Read audio data
            val readCount = audioRecord?.read(shortBuffer, 0, readBufferSize) ?: 0
            
            if (readCount > 0) {
                // Convert short to float and update rolling buffer
                updateRollingBuffer(shortBuffer, readCount)

                val currentTime = System.currentTimeMillis()
                if (currentTime - lastLevelLogTime >= 1000L) {
                    val rms = calculateRms(shortBuffer, readCount)
                    Log.d(TAG, "Mic RMS level: $rms")
                    lastLevelLogTime = currentTime
                }
                
                // Process at regular intervals
                if (currentTime - lastProcessTime >= slideIntervalMs) {
                    // Copy current window in chronological order (oldest -> newest).
                    val windowCopy = getOrderedWindow()
                    
                    // Invoke callback on a different thread to avoid blocking recording
                    withContext(Dispatchers.Default) {
                        onAudioReady(windowCopy)
                    }
                    
                    lastProcessTime = currentTime
                }
            } else if (readCount < 0) {
                Log.e(TAG, "Error reading audio: $readCount")
                delay(10)
            }
            
            // Small delay to prevent tight loop
            delay(1)
        }
    }
    
    /**
     * Update rolling buffer with new audio samples
     */
    private fun updateRollingBuffer(newSamples: ShortArray, count: Int) {
        // Convert short samples to float [-1.0, 1.0]
        val floatSamples = FloatArray(count) { i ->
            newSamples[i] / 32768.0f
        }
        
        // Add to rolling buffer (circular buffer logic)
        var srcPos = 0
        while (srcPos < count) {
            val remaining = count - srcPos
            val spaceInBuffer = windowSize - bufferPosition
            val toCopy = min(remaining, spaceInBuffer)
            
            System.arraycopy(floatSamples, srcPos, rollingBuffer, bufferPosition, toCopy)
            
            bufferPosition = (bufferPosition + toCopy) % windowSize
            srcPos += toCopy
        }
    }

    /**
     * Return the rolling audio window ordered from oldest to newest sample.
     */
    private fun getOrderedWindow(): FloatArray {
        if (bufferPosition == 0) {
            return rollingBuffer.copyOf()
        }

        val ordered = FloatArray(windowSize)
        val tailLength = windowSize - bufferPosition
        System.arraycopy(rollingBuffer, bufferPosition, ordered, 0, tailLength)
        System.arraycopy(rollingBuffer, 0, ordered, tailLength, bufferPosition)
        return ordered
    }

    private fun calculateRms(samples: ShortArray, count: Int): Float {
        if (count <= 0) return 0f
        var sum = 0.0
        for (i in 0 until count) {
            val normalized = samples[i] / 32768.0
            sum += normalized * normalized
        }
        return kotlin.math.sqrt(sum / count).toFloat()
    }
    
    /**
     * Check if currently recording
     */
    fun isRecording(): Boolean = isRecording
    
    companion object {
        private const val TAG = "AudioRecorder"
    }
}
