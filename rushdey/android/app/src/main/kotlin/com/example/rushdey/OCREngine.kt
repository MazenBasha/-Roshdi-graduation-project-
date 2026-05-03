package com.example.rushdey

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.tasks.await

/**
 * OCREngine — On-device text recognition using Google ML Kit.
 */
class OCREngine {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    /**
     * Recognize text from a JPEG/PNG byte array.
     */
    suspend fun recognizeText(imageBytes: ByteArray): String {
        return try {
            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                ?: return "خطأ في قراءة الصورة"
            
            val image = InputImage.fromBitmap(bitmap, 0)
            val text = recognizer.process(image).await().text
            if (text.isBlank()) "مفيش نص واضح قدامي" else text
        } catch (e: Exception) {
            Log.e(TAG, "OCR failed", e)
            "فشل في التعرف على النص"
        }
    }

    fun release() {
        recognizer.close()
    }

    companion object {
        private const val TAG = "OCREngine"
    }
}
