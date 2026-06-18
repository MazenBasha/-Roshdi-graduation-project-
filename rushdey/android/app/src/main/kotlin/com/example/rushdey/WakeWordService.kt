package com.example.rushdey

import android.app.*
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import io.flutter.plugin.common.EventChannel
import kotlinx.coroutines.*
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import org.json.JSONObject

class WakeWordService : Service() {
    private var voskSpeechService: SpeechService? = null
    
    private var isListening = false
    private var hasDetectedInCurrentWindow = false
    
    private var eventSink: EventChannel.EventSink? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    
    override fun onCreate() {
        super.onCreate()
        serviceInstance = this
        eventSink = pendingEventSink
        Log.i(TAG, "WakeWordService created")
        createNotificationChannel()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // ALWAYS call startForeground in onStartCommand to satisfy Android 14+ requirements
        startForeground(NOTIFICATION_ID, createNotification())
        
        when (intent?.action) {
            ACTION_START_LISTENING -> {
                startListening()
            }
            ACTION_STOP_LISTENING -> stopListening()
            ACTION_STOP_SERVICE -> {
                stopListening()
                stopSelf()
            }
        }
        
        return START_STICKY
    }
    
    override fun onBind(intent: Intent?): IBinder? = null
    
    private fun startListening() {
        if (isListening) return
        hasDetectedInCurrentWindow = false
        
        // Initialize engines in background to avoid blocking main thread and causing ANRs
        CoroutineScope(Dispatchers.IO).launch {
            initVoskAndStart()
        }
    }
    
    private suspend fun initVoskAndStart() {
        var voskModel = VoskIntentEngine.sharedModel
        if (voskModel == null) {
            // Need to load it
            val loaded = suspendCancellableCoroutine<Boolean> { cont ->
                VoskIntentEngine(this@WakeWordService) { eventSink }.initialize { ready ->
                    cont.resume(ready, null)
                }
            }
            if (!loaded) {
                withContext(Dispatchers.Main) {
                    sendError("Failed to load Vosk model")
                    stopSelf()
                }
                return
            }
            voskModel = VoskIntentEngine.sharedModel
        }
        
        withContext(Dispatchers.Main) {
            if (voskModel == null) {
                sendError("Failed to load Vosk model")
                return@withContext
            }
            
            try {
                // Use standard recognizer (this model does not support runtime graphs/grammar)
                val recognizer = Recognizer(voskModel, 16000.0f)
                voskSpeechService = SpeechService(recognizer, 16000.0f).apply {
                    startListening(object : RecognitionListener {
                        override fun onPartialResult(hypothesis: String?) {
                            checkVoskResult(hypothesis)
                        }
                        override fun onResult(hypothesis: String?) {
                            checkVoskResult(hypothesis)
                        }
                        override fun onFinalResult(hypothesis: String?) {
                            checkVoskResult(hypothesis)
                        }
                        override fun onError(exception: Exception?) {
                            sendError(exception?.message ?: "Vosk Error")
                        }
                        override fun onTimeout() {}
                    })
                }
                isListening = true
                sendEvent(mapOf("type" to "status", "status" to "listening"))
                Log.i(TAG, "Vosk Wake Word listening started")
            } catch (e: Exception) {
                sendError(e.message ?: "Failed to start Vosk")
                stopSelf()
            }
        }
    }
    
    private fun checkVoskResult(hypothesis: String?) {
        if (hypothesis.isNullOrBlank()) return
        try {
            val partial = JSONObject(hypothesis).optString("partial", "")
            val text = JSONObject(hypothesis).optString("text", "")
            val target = if (text.isNotBlank()) text else partial
            
            if (target.contains("رشدي") && !hasDetectedInCurrentWindow) {
                hasDetectedInCurrentWindow = true
                // Stop vosk immediately to prevent multiple triggers
                voskSpeechService?.stop()
                
                Log.i(TAG, "Vosk Wake word detected! '$target'")
                sendEvent(mapOf(
                    "type" to "detected",
                    "wakeWord" to "رشدي",
                    "confidence" to 1.0f
                ))
            }
        } catch (_: Exception) {}
    }
    
    private fun stopListening() {
        if (!isListening) return
        voskSpeechService?.apply {
            stop()
            shutdown()
        }
        voskSpeechService = null
        isListening = false
        
        sendEvent(mapOf("type" to "status", "status" to "stopped"))
        Log.i(TAG, "Listening stopped")
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
    }
    

    private fun sendError(msg: String) {
        lastErrorMessage = msg
        sendEvent(mapOf("type" to "error", "message" to msg))
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
    }

    private fun sendEvent(event: Map<String, Any>) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            eventSink?.success(event)
            return
        }
        mainHandler.post { eventSink?.success(event) }
    }
    
    fun setEventSink(sink: EventChannel.EventSink?) {
        this.eventSink = sink
    }
    
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(CHANNEL_ID, "Wake Word Detection", NotificationManager.IMPORTANCE_LOW).apply {
                description = "Listening for wake word"
                setShowBadge(false)
            }
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }
    
    private fun createNotification(): Notification {
        val notificationIntent = packageManager.getLaunchIntentForPackage(packageName)
        val flags = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT else PendingIntent.FLAG_UPDATE_CURRENT
        val pendingIntent = PendingIntent.getActivity(this, 0, notificationIntent, flags)
        
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) Notification.Builder(this, CHANNEL_ID) else Notification.Builder(this)
        
        return builder
            .setContentTitle("Rushdie - Wake Word")
            .setContentText("Listening for 'رشدي'...")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        stopListening()
        /* isInitialized = false */
        serviceInstance = null
        Log.i(TAG, "WakeWordService destroyed")
    }
    
    companion object {
        private const val TAG = "WakeWordService"
        private const val CHANNEL_ID = "wake_word_channel"
        private const val NOTIFICATION_ID = 1001
        
        const val ACTION_START_LISTENING = "com.example.rushdey.START_LISTENING"
        const val ACTION_STOP_LISTENING = "com.example.rushdey.STOP_LISTENING"
        const val ACTION_STOP_SERVICE = "com.example.rushdey.STOP_SERVICE"
        const val EXTRA_ENGINE_TYPE = "engine_type"
        
        private var serviceInstance: WakeWordService? = null
        @Volatile private var pendingEventSink: EventChannel.EventSink? = null
        @Volatile private var lastErrorMessage: String? = null
        
        fun getInstance(): WakeWordService? = serviceInstance
        fun setPendingEventSink(sink: EventChannel.EventSink?) { pendingEventSink = sink }
        fun getLastError(): String? = lastErrorMessage
    }
}