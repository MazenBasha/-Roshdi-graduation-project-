package com.example.rushdey

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

class StaticImageTestReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val pendingResult = goAsync()
        CoroutineScope(Dispatchers.IO).launch {
            try {
                runTests(context.applicationContext)
            } catch (e: Exception) {
                Log.e(TAG, "RUSHDEY_STATIC_TEST ERROR ${e.message}", e)
            } finally {
                pendingResult.finish()
            }
        }
    }

    private fun runTests(context: Context) {
        val internalRoot = File(context.filesDir, "test_images")
        val externalRoot = File(context.getExternalFilesDir(null), "test_images")
        val testRoot = if (internalRoot.exists()) internalRoot else externalRoot
        Log.i(TAG, "RUSHDEY_STATIC_TEST START root=${testRoot.absolutePath}")
        if (!testRoot.exists()) {
            Log.w(TAG, "RUSHDEY_STATIC_TEST NO_ROOT root=${testRoot.absolutePath}")
            return
        }

        runOcrTests(context, File(testRoot, "ocr"))
        runCurrencyTests(context, File(testRoot, "currency"))
        runFaceTests(context, File(testRoot, "face"))

        Log.i(TAG, "RUSHDEY_STATIC_TEST DONE")
    }

    private fun runOcrTests(context: Context, dir: File) {
        val files = imageFiles(dir)
        if (files.isEmpty()) {
            Log.i(TAG, "RUSHDEY_STATIC_TEST OCR no_images dir=${dir.absolutePath}")
            return
        }

        val engine = OCREngine(context)
        val ready = engine.initialize()
        Log.i(TAG, "RUSHDEY_STATIC_TEST OCR ready=$ready count=${files.size}")
        if (!ready) return

        files.forEach { file ->
            val text = engine.recognizeText(file.readBytes()).oneLine()
            Log.i(TAG, "RUSHDEY_STATIC_TEST OCR file=${file.name} text=$text")
        }
        engine.release()
    }

    private fun runCurrencyTests(context: Context, dir: File) {
        val files = imageFiles(dir)
        if (files.isEmpty()) {
            Log.i(TAG, "RUSHDEY_STATIC_TEST CURRENCY no_images dir=${dir.absolutePath}")
            return
        }

        val engine = CurrencyEngine(context)
        val ready = engine.initialize()
        Log.i(TAG, "RUSHDEY_STATIC_TEST CURRENCY ready=$ready count=${files.size}")
        if (!ready) return

        files.forEach { file ->
            val result = engine.detect(file.readBytes())
            val detections = result.detections.joinToString("|") {
                "${it.className}:${"%.3f".format(it.confidence)}"
            }
            Log.i(
                TAG,
                "RUSHDEY_STATIC_TEST CURRENCY file=${file.name} total=${result.total} text=${result.arabicName.oneLine()} detections=$detections"
            )
        }
        engine.release()
    }

    private fun runFaceTests(context: Context, dir: File) {
        val enrollRoot = File(dir, "enroll")
        val queryRoot = File(dir, "query")
        if (!enrollRoot.exists() && !queryRoot.exists()) {
            Log.i(TAG, "RUSHDEY_STATIC_TEST FACE no_images dir=${dir.absolutePath}")
            return
        }

        val engine = FaceEngine(context)
        val ready = engine.initialize()
        Log.i(TAG, "RUSHDEY_STATIC_TEST FACE ready=$ready")
        if (!ready) return

        val staticNames = mutableListOf<String>()
        enrollRoot.listFiles { file -> file.isDirectory }
            ?.sortedBy { it.name.lowercase() }
            ?.forEach { personDir ->
                val images = imageFiles(personDir).map { it.readBytes() }
                val staticName = "__static_${personDir.name}"
                val ok = engine.enrollPerson(staticName, images)
                if (ok) staticNames.add(staticName)
                Log.i(
                    TAG,
                    "RUSHDEY_STATIC_TEST FACE_ENROLL name=${personDir.name} images=${images.size} success=$ok"
                )
            }

        val queryGroups = queryRoot.listFiles { file -> file.isDirectory }
            ?.sortedBy { it.name.lowercase() }
            .orEmpty()
        if (queryGroups.isNotEmpty()) {
            queryGroups.forEach { expectedDir ->
                imageFiles(expectedDir).forEach { file ->
                    val result = engine.recognize(file.readBytes())
                    val name = result.name.removePrefix("__static_")
                    Log.i(
                        TAG,
                        "RUSHDEY_STATIC_TEST FACE_QUERY expected=${expectedDir.name} file=${file.name} name=$name similarity=${"%.3f".format(result.similarity)} recognized=${result.recognized}"
                    )
                }
            }
        } else {
            imageFiles(queryRoot).forEach { file ->
                val result = engine.recognize(file.readBytes())
                val name = result.name.removePrefix("__static_")
                Log.i(
                    TAG,
                    "RUSHDEY_STATIC_TEST FACE_QUERY file=${file.name} name=$name similarity=${"%.3f".format(result.similarity)} recognized=${result.recognized}"
                )
            }
        }

        staticNames.forEach { engine.deleteEnrolledPerson(it) }
        engine.release()
    }

    private fun imageFiles(dir: File): List<File> {
        if (!dir.exists()) return emptyList()
        return dir.listFiles { file ->
            file.isFile && file.extension.lowercase() in SUPPORTED_IMAGE_EXTENSIONS
        }?.sortedBy { it.name.lowercase() }.orEmpty()
    }

    private fun String.oneLine(): String = replace("\r", "\\r").replace("\n", "\\n")

    companion object {
        private const val TAG = "RushdeyStaticTest"
        private val SUPPORTED_IMAGE_EXTENSIONS = setOf("jpg", "jpeg", "png", "bmp", "webp")
    }
}
