package com.example.rushdey

import kotlin.math.*

/**
 * MelSpectrogram preprocessor matching training configuration
 * Converts 16kHz PCM audio to mel spectrogram matching config.json
 */
class MelSpectrogram(
    private val sampleRate: Int = 16000,
    private val nFft: Int = 1024,
    private val hopLength: Int = 512,
    private val nMels: Int = 64,
    private val fMin: Float = 20f,
    private val fMax: Float = 8000f
) {
    private val melFilterbank = createMelFilterbank()

    /**
     * Compute mel spectrogram from audio samples
     * @param audioSamples Float array of audio samples (16kHz mono PCM)
     * @return 2D FloatArray [nMels x timeSteps]
     */
    fun compute(audioSamples: FloatArray): Array<FloatArray> {
        // 1. Compute STFT
        val stft = computeStft(audioSamples)
        
        // 2. Compute magnitude spectrogram
        val magnitudeSpec = stft.map { frame ->
            frame.map { complex -> sqrt(complex.first * complex.first + complex.second * complex.second) }.toFloatArray()
        }.toTypedArray()
        
        // 3. Apply mel filterbank
        val melSpec = applyMelFilterbank(magnitudeSpec)
        
        // 4. Convert to log scale (add small epsilon to avoid log(0))
        return melSpec.map { frame ->
            frame.map { value -> ln(value + 1e-10f) }.toFloatArray()
        }.toTypedArray()
    }

    /**
     * Compute Short-Time Fourier Transform
     */
    private fun computeStft(signal: FloatArray): Array<Array<Pair<Float, Float>>> {
        val numFrames = (signal.size - nFft) / hopLength + 1
        val fftSize = nFft / 2 + 1
        val window = hammingWindow(nFft)
        
        val stft = Array(numFrames) { Array(fftSize) { Pair(0f, 0f) } }
        
        for (frameIdx in 0 until numFrames) {
            val start = frameIdx * hopLength
            val frame = FloatArray(nFft) { i ->
                if (start + i < signal.size) signal[start + i] * window[i] else 0f
            }
            
            stft[frameIdx] = rfft(frame)
        }
        
        return stft
    }

    /**
     * Real FFT implementation (returns only positive frequencies)
     */
    private fun rfft(signal: FloatArray): Array<Pair<Float, Float>> {
        val n = signal.size
        val fftSize = n / 2 + 1
        
        // Pad to complex array
        val complexSignal = Array(n) { Pair(signal[it], 0f) }
        
        // Perform FFT
        fft(complexSignal)
        
        // Return only positive frequencies
        return Array(fftSize) { complexSignal[it] }
    }

    /**
     * Cooley-Tukey FFT algorithm (in-place)
     */
    private fun fft(signal: Array<Pair<Float, Float>>) {
        val n = signal.size
        if (n <= 1) return
        
        // Bit reversal
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                val temp = signal[i]
                signal[i] = signal[j]
                signal[j] = temp
            }
            var k = n / 2
            while (k <= j) {
                j -= k
                k /= 2
            }
            j += k
        }
        
        // FFT computation
        var len = 2
        while (len <= n) {
            val angle = -2.0 * PI / len
            val wLen = Pair(cos(angle).toFloat(), sin(angle).toFloat())
            
            var i = 0
            while (i < n) {
                var w = Pair(1f, 0f)
                for (j in 0 until len / 2) {
                    val u = signal[i + j]
                    val v = complexMultiply(w, signal[i + j + len / 2])
                    signal[i + j] = Pair(u.first + v.first, u.second + v.second)
                    signal[i + j + len / 2] = Pair(u.first - v.first, u.second - v.second)
                    w = complexMultiply(w, wLen)
                }
                i += len
            }
            len *= 2
        }
    }

    /**
     * Complex number multiplication
     */
    private fun complexMultiply(a: Pair<Float, Float>, b: Pair<Float, Float>): Pair<Float, Float> {
        return Pair(
            a.first * b.first - a.second * b.second,
            a.first * b.second + a.second * b.first
        )
    }

    /**
     * Hamming window function
     */
    private fun hammingWindow(size: Int): FloatArray {
        return FloatArray(size) { i ->
            (0.54 - 0.46 * cos(2.0 * PI * i / (size - 1))).toFloat()
        }
    }

    /**
     * Create mel filterbank
     */
    private fun createMelFilterbank(): Array<FloatArray> {
        val fftSize = nFft / 2 + 1
        
        // Convert Hz to Mel
        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)
        
        // Create mel points equally spaced
        val melPoints = FloatArray(nMels + 2) { i ->
            melMin + (melMax - melMin) * i / (nMels + 1)
        }
        
        // Convert mel points back to Hz
        val hzPoints = melPoints.map { melToHz(it) }
        
        // Convert Hz to FFT bin
        val binPoints = hzPoints.map { (it / sampleRate * nFft).toInt() }
        
        // Create filterbank
        val filterbank = Array(nMels) { FloatArray(fftSize) }
        
        for (i in 0 until nMels) {
            val left = binPoints[i]
            val center = binPoints[i + 1]
            val right = binPoints[i + 2]
            
            for (j in left until center) {
                if (j < fftSize) {
                    filterbank[i][j] = (j - left).toFloat() / (center - left)
                }
            }
            for (j in center until right) {
                if (j < fftSize) {
                    filterbank[i][j] = (right - j).toFloat() / (right - center)
                }
            }
        }
        
        return filterbank
    }

    /**
     * Apply mel filterbank to magnitude spectrogram
     */
    private fun applyMelFilterbank(magnitudeSpec: Array<FloatArray>): Array<FloatArray> {
        return magnitudeSpec.map { frame ->
            FloatArray(nMels) { melIdx ->
                var sum = 0f
                for (binIdx in melFilterbank[melIdx].indices) {
                    sum += frame[binIdx] * melFilterbank[melIdx][binIdx]
                }
                sum
            }
        }.toTypedArray()
    }

    /**
     * Convert Hz to Mel scale
     */
    private fun hzToMel(hz: Float): Float {
        return 2595f * log10(1f + hz / 700f)
    }

    /**
     * Convert Mel to Hz scale
     */
    private fun melToHz(mel: Float): Float {
        return 700f * (10f.pow(mel / 2595f) - 1f)
    }
}
