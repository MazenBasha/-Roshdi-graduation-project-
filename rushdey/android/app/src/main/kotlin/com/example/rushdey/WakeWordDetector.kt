package com.example.rushdey

import android.util.Log
import kotlin.math.abs

/**
 * Wake-word matcher for the Egyptian Arabic assistant name "رشدي".
 *
 * Vosk often returns small spelling variations, missed spaces, or stretched vowels.
 * This matcher normalizes those differences, checks known aliases directly, then
 * applies constrained fuzzy matching only against wake aliases to avoid random
 * speech triggering the assistant.
 */
object WakeWordDetector {
    private const val TAG = "WakeWordDetector"
    private const val FUZZY_THRESHOLD = 0.75

    val wakeAliases = listOf(
        "رشدي",
        "رشدى",
        "يا رشدي",
        "يا رشدى",
        "يارشدي",
        "يارشدى",
        "يا ا رشدي",
        "ياا رشدي",
        "يااا رشدي",
        "ياااا رشدي",
        "رشدي يا",
        "اهلا رشدي",
        "أهلا رشدي",
        "الو رشدي",
        "ألو رشدي",
        "اسمع يا رشدي",
        "بص يا رشدي",
        "يا رشدي اسمعني",
        "رشدي اسمعني",
        "رشدي افتح",
        "يا رشدي افتح",
        "يا رشدي ساعدني",
        "رجدي",
        "روجدي",
        "روشدي",
        "رشتي",
        "رشديه",
        "رشديي",
        "يا رجدي",
        "يا روجدي",
        "يا روشدي",
        "يا رشتي",
        "يا رشديه",
        "يا رشديي"
    ).distinct()

    private val normalizedAliases = wakeAliases
        .map { alias -> NormalizedAlias(raw = alias, normalized = normalizeWakeWord(alias)) }
        .filter { it.normalized.isNotBlank() }
        .distinctBy { compact(it.normalized) }

    data class WakeWordMatch(
        val alias: String,
        val normalizedAlias: String,
        val normalizedText: String,
        val score: Double,
        val strategy: String
    )

    private data class NormalizedAlias(
        val raw: String,
        val normalized: String
    )

    fun normalizeWakeWord(s: String): String {
        return s.trim()
            .lowercase()
            .replace(Regex("[\\u064B-\\u065F\\u0670]"), "")
            .replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            .replace("ى", "ي")
            .replace("ة", "ه")
            .replace("ؤ", "و").replace("ئ", "ي")
            .replace(Regex("[^\\u0600-\\u06FF\\s]"), " ")
            .replace(Regex("ا{2,}"), "ا")
            .replace(Regex("و{2,}"), "و")
            .replace(Regex("ي{2,}"), "ي")
            .replace(Regex("ي\\s*ا"), "يا")
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    fun isWakeWord(text: String): Boolean {
        return findWakeWordMatch(text) != null
    }

    fun findWakeWordMatch(text: String): WakeWordMatch? {
        val normalizedText = normalizeWakeWord(text)
        if (normalizedText.isBlank()) return null

        findDirectMatch(normalizedText)?.let { match ->
            Log.i(TAG, "Wake word direct match: alias='${match.alias}', text='$normalizedText'")
            return match
        }

        val candidates = buildFuzzyCandidates(normalizedText)
        var bestMatch: WakeWordMatch? = null

        normalizedAliases.forEach { alias ->
            candidates.forEach { candidate ->
                if (!isSafeFuzzyCandidate(candidate, alias.normalized)) return@forEach
                val score = sequenceSimilarity(compact(candidate), compact(alias.normalized))
                if (score >= FUZZY_THRESHOLD && score > (bestMatch?.score ?: 0.0)) {
                    bestMatch = WakeWordMatch(
                        alias = alias.raw,
                        normalizedAlias = alias.normalized,
                        normalizedText = normalizedText,
                        score = score,
                        strategy = "fuzzy"
                    )
                }
            }
        }

        bestMatch?.let {
            Log.i(TAG, "Wake word fuzzy match: alias='${it.alias}', score=${"%.2f".format(it.score)}, text='$normalizedText'")
        }
        return bestMatch
    }

    private fun findDirectMatch(normalizedText: String): WakeWordMatch? {
        val compactText = compact(normalizedText)
        val directCandidates = buildPhraseCandidates(normalizedText)
            .flatMap { candidate -> listOf(candidate, compact(candidate)) }
            .distinct()

        val alias = normalizedAliases.firstOrNull {
            val compactAlias = compact(it.normalized)
            directCandidates.any { candidate ->
                candidate == it.normalized || candidate == compactAlias
            } || (compactAlias.length > 4 && compactText.contains(compactAlias))
        } ?: return null

        return WakeWordMatch(
            alias = alias.raw,
            normalizedAlias = alias.normalized,
            normalizedText = normalizedText,
            score = 1.0,
            strategy = "direct"
        )
    }

    private fun buildFuzzyCandidates(normalizedText: String): List<String> {
        return buildPhraseCandidates(normalizedText)
            .flatMap { candidate -> listOf(candidate, compact(candidate)) }
            .filter { it.isNotBlank() }
            .distinct()
    }

    private fun buildPhraseCandidates(normalizedText: String): List<String> {
        val words = normalizedText.split(" ").filter { it.isNotBlank() }
        val candidates = mutableListOf(normalizedText)

        words.forEach { candidates.add(it) }
        for (windowSize in 2..minOf(4, words.size)) {
            for (start in 0..(words.size - windowSize)) {
                candidates.add(words.subList(start, start + windowSize).joinToString(" "))
            }
        }

        return candidates
            .filter { it.isNotBlank() }
            .distinct()
    }

    private fun isSafeFuzzyCandidate(candidate: String, alias: String): Boolean {
        val c = compact(candidate)
        val a = compact(alias)
        if (c.length < 4 || a.length < 4) return false
        if (abs(c.length - a.length) > maxOf(2, a.length / 2)) return false
        return hasWakePrefix(c) && looksWakeLike(c)
    }

    private fun looksWakeLike(s: String): Boolean {
        return s.contains("ر") &&
            (s.contains("ش") || s.contains("ج")) &&
            (s.contains("د") || s.contains("ت"))
    }

    private fun hasWakePrefix(s: String): Boolean {
        return listOf("ر", "يا", "اهلا", "الو", "اسمع", "بص").any { s.startsWith(it) }
    }

    private fun compact(s: String): String {
        return s.replace(" ", "")
    }

    private fun sequenceSimilarity(a: String, b: String): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0
        val distance = levenshteinDistance(a, b)
        return 1.0 - (distance.toDouble() / maxOf(a.length, b.length).toDouble())
    }

    private fun levenshteinDistance(a: String, b: String): Int {
        val previous = IntArray(b.length + 1) { it }
        val current = IntArray(b.length + 1)

        for (i in 1..a.length) {
            current[0] = i
            for (j in 1..b.length) {
                val substitutionCost = if (a[i - 1] == b[j - 1]) 0 else 1
                current[j] = minOf(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + substitutionCost
                )
            }
            for (j in 0..b.length) {
                previous[j] = current[j]
            }
        }

        return previous[b.length]
    }
}
