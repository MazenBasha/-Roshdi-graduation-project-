package com.example.rushdey

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.googlecode.tesseract.android.TessBaseAPI
import io.flutter.FlutterInjector
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

/**
 * Offline Arabic OCR using Tesseract and bundled ara.traineddata.
 */
class OCREngine(private val context: Context) {

    private var tessBaseApi: TessBaseAPI? = null
    private val dataDir = File(context.filesDir, "tesseract")
    private val tessDataDir = File(dataDir, "tessdata")
    private val trainedDataFile = File(tessDataDir, "ara.traineddata")
    private val noClearText = "\u0645\u0641\u064a\u0634 \u0646\u0635 \u0648\u0627\u0636\u062d \u0642\u062f\u0627\u0645\u064a"
    private val latinRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    fun initialize(): Boolean {
        return try {
            copyTrainedDataIfNeeded()
            val api = TessBaseAPI()
            val ok = api.init(dataDir.absolutePath, "ara")
            if (!ok) {
                api.recycle()
                Log.e(TAG, "Tesseract init failed for Arabic traineddata")
                return false
            }
            api.setPageSegMode(TessBaseAPI.PageSegMode.PSM_AUTO)
            api.setVariable("preserve_interword_spaces", "1")
            tessBaseApi = api
            Log.i(TAG, "OCREngine initialized from ${trainedDataFile.absolutePath}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "OCREngine init failed", e)
            false
        }
    }

    /**
     * Recognize Arabic text from a JPEG byte array captured by CameraX.
     */
    fun recognizeText(imageBytes: ByteArray): String {
        val api = tessBaseApi ?: return "محرك القراءة غير جاهز"
        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            ?: return "خطأ في قراءة الصورة"

        if (!hasEnoughInk(bitmap)) {
            bitmap.recycle()
            return noClearText
        }

        return recognizeTextWithFiltering(api, bitmap)

        /*
        return try {
            val prepared = prepareBitmapForOcr(bitmap)
            synchronized(api) {
                api.setImage(prepared)
                val text = api.getUTF8Text()
                    ?.replace(Regex("[\\r\\n]+"), "\n")
                    ?.trim()
                    .orEmpty()
                api.clear()
                if (text.isBlank()) "مفيش نص واضح قدامي" else text
            }
        } catch (e: Exception) {
            Log.e(TAG, "OCR failed", e)
            "فشل في التعرف على النص"
        } finally {
            bitmap.recycle()
        }
    }

    */
    }

    private fun prepareBitmapForOcr(source: Bitmap, contrast: Float = 1.35f): Bitmap {
        val scaled = scaleForOcr(source)
        val cropped = cropLikelyTextRegion(scaled)
        val grayscale = Bitmap.createBitmap(cropped.width, cropped.height, Bitmap.Config.ARGB_8888)
        val offset = 128f * (1f - contrast)
        val contrastMatrix = ColorMatrix(
            floatArrayOf(
                contrast, 0f, 0f, 0f, offset,
                0f, contrast, 0f, 0f, offset,
                0f, 0f, contrast, 0f, offset,
                0f, 0f, 0f, 1f, 0f
            )
        )
        val matrix = ColorMatrix().apply {
            setSaturation(0f)
            postConcat(contrastMatrix)
        }
        val canvas = Canvas(grayscale)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG).apply {
            colorFilter = ColorMatrixColorFilter(matrix)
        }
        canvas.drawBitmap(cropped, 0f, 0f, paint)
        if (cropped !== scaled) cropped.recycle()
        if (scaled !== source) scaled.recycle()
        return grayscale
    }

    private fun cropLikelyTextRegion(source: Bitmap): Bitmap {
        val width = source.width
        val height = source.height
        val pixels = IntArray(width * height)
        source.getPixels(pixels, 0, width, 0, 0, width, height)

        var minX = width
        var minY = height
        var maxX = -1
        var maxY = -1
        var darkCount = 0
        val darkThreshold = 210

        for (y in 0 until height) {
            val row = y * width
            for (x in 0 until width) {
                val p = pixels[row + x]
                val luminance = (0.299f * Color.red(p) + 0.587f * Color.green(p) + 0.114f * Color.blue(p)).toInt()
                if (luminance < darkThreshold) {
                    darkCount++
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }

        val darkRatio = darkCount.toFloat() / pixels.size.toFloat()
        if (maxX < minX || maxY < minY || darkRatio < 0.001f || darkRatio > 0.35f) {
            return source
        }

        val pad = maxOf(24, (maxOf(maxX - minX, maxY - minY) * 0.16f).toInt())
        val left = (minX - pad).coerceAtLeast(0)
        val top = (minY - pad).coerceAtLeast(0)
        val right = (maxX + pad).coerceAtMost(width - 1)
        val bottom = (maxY + pad).coerceAtMost(height - 1)
        val cropWidth = right - left + 1
        val cropHeight = bottom - top + 1

        if (cropWidth >= width * 0.92f && cropHeight >= height * 0.92f) {
            return source
        }
        if (cropWidth < 40 || cropHeight < 20) {
            return source
        }
        return Bitmap.createBitmap(source, left, top, cropWidth, cropHeight)
    }

    private fun hasEnoughInk(source: Bitmap): Boolean {
        val step = maxOf(1, minOf(source.width, source.height) / 320)
        var sampled = 0
        var dark = 0
        for (y in 0 until source.height step step) {
            for (x in 0 until source.width step step) {
                val p = source.getPixel(x, y)
                val luminance = (0.299f * Color.red(p) + 0.587f * Color.green(p) + 0.114f * Color.blue(p)).toInt()
                sampled++
                if (luminance < 215) dark++
            }
        }
        val ratio = if (sampled == 0) 0f else dark.toFloat() / sampled.toFloat()
        return ratio >= 0.0015f
    }

    private fun prepareBinaryBitmapForOcr(source: Bitmap): Bitmap {
        val gray = prepareBitmapForOcr(source, contrast = 1.65f)
        val width = gray.width
        val height = gray.height
        val pixels = IntArray(width * height)
        gray.getPixels(pixels, 0, width, 0, 0, width, height)

        val histogram = IntArray(256)
        for (pixel in pixels) {
            histogram[Color.red(pixel)]++
        }
        val threshold = otsuThreshold(histogram, pixels.size).coerceIn(90, 210)
        for (i in pixels.indices) {
            val value = if (Color.red(pixels[i]) > threshold) 255 else 0
            pixels[i] = Color.rgb(value, value, value)
        }
        val binary = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        binary.setPixels(pixels, 0, width, 0, 0, width, height)
        gray.recycle()
        return binary
    }

    private fun scaleForOcr(source: Bitmap): Bitmap {
        val maxSide = 1600
        val minReadableSide = 700
        val longest = maxOf(source.width, source.height).toFloat()
        val shortest = minOf(source.width, source.height).toFloat()
        val downScale = if (longest > maxSide) maxSide / longest else 1f
        val upScale = if (shortest < minReadableSide) minReadableSide / shortest else 1f
        val scale = (downScale * upScale).coerceIn(0.25f, 2.0f)
        return if (scale != 1f) {
            Bitmap.createScaledBitmap(
                source,
                (source.width * scale).toInt().coerceAtLeast(1),
                (source.height * scale).toInt().coerceAtLeast(1),
                true
            )
        } else {
            source
        }
    }

    private fun otsuThreshold(histogram: IntArray, total: Int): Int {
        var sum = 0.0
        for (i in histogram.indices) sum += i * histogram[i]

        var sumBackground = 0.0
        var weightBackground = 0
        var maxVariance = 0.0
        var threshold = 128

        for (i in histogram.indices) {
            weightBackground += histogram[i]
            if (weightBackground == 0) continue
            val weightForeground = total - weightBackground
            if (weightForeground == 0) break

            sumBackground += i * histogram[i]
            val meanBackground = sumBackground / weightBackground
            val meanForeground = (sum - sumBackground) / weightForeground
            val variance = weightBackground.toDouble() * weightForeground.toDouble() *
                (meanBackground - meanForeground) * (meanBackground - meanForeground)
            if (variance > maxVariance) {
                maxVariance = variance
                threshold = i
            }
        }
        return threshold
    }

    private fun recognizeTextWithFiltering(api: TessBaseAPI, bitmap: Bitmap): String {
        return try {
            val attempts = mutableListOf<OcrAttempt>()
            runLatinOcrAttempt(bitmap)?.let { attempts.add(it) }
            val variants = listOf(
                OcrVariant("binary_sparse", TessBaseAPI.PageSegMode.PSM_SPARSE_TEXT) { prepareBinaryBitmapForOcr(bitmap) }
            )

            for (variant in variants) {
                val prepared = variant.createBitmap()
                try {
                    attempts.add(runOcrAttempt(api, prepared, variant.name, variant.pageSegMode))
                } finally {
                    prepared.recycle()
                }
            }

            val best = attempts.maxByOrNull { it.quality }
            Log.d(TAG, "OCR attempts=${attempts.joinToString { "${it.name}:conf=${it.confidence},q=${"%.1f".format(it.quality)},readable=${it.readable},clean='${it.cleaned.take(80)}',text='${it.text.take(80)}'" }}")
            when {
                best == null -> noClearText
                best.readable -> best.text
                canShowLowConfidenceGuess(best.cleaned, best.confidence) -> best.cleaned
                else -> noClearText
            }
        } catch (e: Exception) {
            Log.e(TAG, "OCR failed", e)
            "\u0641\u0634\u0644 \u0641\u064a \u0627\u0644\u062a\u0639\u0631\u0641 \u0639\u0644\u0649 \u0627\u0644\u0646\u0635"
        } finally {
            bitmap.recycle()
        }
    }

    private fun runLatinOcrAttempt(bitmap: Bitmap): OcrAttempt? {
        return try {
            val result = Tasks.await(
                latinRecognizer.process(InputImage.fromBitmap(bitmap, 0)),
                1500,
                TimeUnit.MILLISECONDS
            )
            val cleaned = cleanRecognizedText(result.text)
            val readable = isReadableText(cleaned, 80)
            val confidence = if (cleaned.isBlank()) 0 else 80
            val text = if (readable) cleaned else noClearText
            OcrAttempt("mlkit_latin", text, cleaned, confidence, readable, qualityScore(cleaned, confidence, readable))
        } catch (e: Exception) {
            Log.w(TAG, "ML Kit OCR failed", e)
            null
        }
    }

    private fun runOcrAttempt(
        api: TessBaseAPI,
        prepared: Bitmap,
        name: String,
        pageSegMode: Int
    ): OcrAttempt {
        return synchronized(api) {
            api.setPageSegMode(pageSegMode)
            api.setImage(prepared)
            val rawText = api.getUTF8Text().orEmpty()
            val confidence = api.meanConfidence()
            api.clear()
            val cleaned = cleanRecognizedText(rawText)
            val readable = isReadableText(cleaned, confidence)
            val text = if (readable) cleaned else noClearText
            OcrAttempt(name, text, cleaned, confidence, readable, qualityScore(cleaned, confidence, readable))
        }
    }

    private fun cleanRecognizedText(rawText: String): String {
        return rawText
            .replace(Regex("[\\u064B-\\u065F\\u0670]"), "")
            .replace(Regex("[\\r\\n]+"), "\n")
            .replace(Regex("[^\\p{L}\\p{N}\\s\\u060C\\u061B\\u061F.,:;!?%/+-]"), " ")
            .replace(Regex("[ \\t]+"), " ")
            .replace(Regex(" ?\\n ?"), "\n")
            .replace(Regex("\\n{3,}"), "\n\n")
            .trim()
    }

    private fun isReadableText(cleaned: String, confidence: Int): Boolean {
        if (cleaned.isBlank()) return false

        val arabicLetters = Regex("[\\u0621-\\u064A]").findAll(cleaned).count()
        val latinLetters = Regex("[A-Za-z]").findAll(cleaned).count()
        val digits = Regex("\\p{N}").findAll(cleaned).count()
        val letterCount = arabicLetters + latinLetters
        val tokens = Regex("\\S+").findAll(cleaned).map { it.value }.toList()
        val meaningfulTokens = tokens.count { token -> token.count { it.isLetterOrDigit() } >= 2 }
        val singleCharTokens = tokens.count { token -> token.count { it.isLetterOrDigit() } == 1 }
        val singleCharRatio = if (tokens.isEmpty()) 1f else singleCharTokens.toFloat() / tokens.size.toFloat()
        val digitHeavyNoise = digits > maxOf(4, letterCount * 2) && arabicLetters < 8

        if (singleCharRatio > 0.55f || digitHeavyNoise) return false

        return when {
            confidence >= 75 && arabicLetters >= 3 -> true
            confidence >= 65 && arabicLetters >= 6 && meaningfulTokens >= 2 -> true
            confidence >= 55 && arabicLetters >= 12 && meaningfulTokens >= 3 -> true
            confidence >= 55 && latinLetters >= 4 && meaningfulTokens >= 2 -> true
            else -> false
        }
    }

    private fun canShowLowConfidenceGuess(cleaned: String, confidence: Int): Boolean {
        if (cleaned.isBlank() || confidence < 22) return false
        val arabicLetters = Regex("[\\u0621-\\u064A]").findAll(cleaned).count()
        val latinLetters = Regex("[A-Za-z]").findAll(cleaned).count()
        val digits = Regex("\\p{N}").findAll(cleaned).count()
        val letterCount = arabicLetters + latinLetters
        val tokens = Regex("\\S+").findAll(cleaned).map { it.value }.toList()
        val meaningfulTokens = tokens.count { token -> token.count { it.isLetterOrDigit() } >= 2 }
        val singleCharTokens = tokens.count { token -> token.count { it.isLetterOrDigit() } == 1 }
        val singleCharRatio = if (tokens.isEmpty()) 1f else singleCharTokens.toFloat() / tokens.size.toFloat()
        val digitHeavyNoise = digits > maxOf(6, letterCount * 2) && letterCount < 10

        return letterCount >= 4 &&
            meaningfulTokens >= 1 &&
            singleCharRatio <= 0.65f &&
            !digitHeavyNoise
    }

    private fun qualityScore(cleaned: String, confidence: Int, readable: Boolean): Double {
        if (!readable) return -1000.0 + confidence
        val arabicLetters = Regex("[\\u0621-\\u064A]").findAll(cleaned).count()
        val latinLetters = Regex("[A-Za-z]").findAll(cleaned).count()
        val digits = Regex("\\p{N}").findAll(cleaned).count()
        val tokens = Regex("\\S+").findAll(cleaned).count()
        return confidence * 2.0 + (arabicLetters + latinLetters) * 1.6 + digits * 0.4 + tokens * 2.0
    }

    private fun filterRecognizedText(rawText: String, confidence: Int): String {
        val cleaned = cleanRecognizedText(rawText)
        return if (isReadableText(cleaned, confidence)) cleaned else noClearText
    }

    private data class OcrVariant(
        val name: String,
        val pageSegMode: Int,
        val createBitmap: () -> Bitmap
    )

    private data class OcrAttempt(
        val name: String,
        val text: String,
        val cleaned: String,
        val confidence: Int,
        val readable: Boolean,
        val quality: Double
    )

    /*
    private fun recognizeTextWithSinglePass(api: TessBaseAPI, bitmap: Bitmap): String {
        return try {
            val prepared = prepareBitmapForOcr(bitmap)
            try {
                synchronized(api) {
                    api.setImage(prepared)
                    val rawText = api.getUTF8Text().orEmpty()
                    val confidence = api.meanConfidence()
                    api.clear()
                    val filteredText = filterRecognizedText(rawText, confidence)
                    Log.d(TAG, "OCR confidence=$confidence raw='${rawText.take(120)}' filtered='${filteredText.take(120)}'")
                    filteredText
                }
            } finally {
                prepared.recycle()
            }
        } catch (e: Exception) {
            Log.e(TAG, "OCR failed", e)
            "\u0641\u0634\u0644 \u0641\u064a \u0627\u0644\u062a\u0639\u0631\u0641 \u0639\u0644\u0649 \u0627\u0644\u0646\u0635"
        } finally {
            bitmap.recycle()
        }
    }
    */

    private fun copyTrainedDataIfNeeded() {
        if (trainedDataFile.exists() && trainedDataFile.length() > 0) return

        tessDataDir.mkdirs()
        val assetName = "assets/tessdata/ara.traineddata"
        val key = try {
            FlutterInjector.instance().flutterLoader().getLookupKeyForAsset(assetName)
        } catch (_: Exception) {
            "flutter_assets/$assetName"
        }

        context.assets.open(key).use { input ->
            FileOutputStream(trainedDataFile).use { output ->
                input.copyTo(output)
            }
        }
    }

    fun release() {
        tessBaseApi?.recycle()
        tessBaseApi = null
        latinRecognizer.close()
    }

    companion object {
        private const val TAG = "OCREngine"
    }
}
