with open('android/app/src/main/kotlin/com/example/rushdey/MainActivity.kt', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('private fun startListeningWithPermission(result: MethodChannel.Result) {', 'private var currentEngine: String = "pytorch"\n    private fun startListeningWithPermission(result: MethodChannel.Result, engine: String) {\n        this.currentEngine = engine')
text = text.replace('"startListening" -> startListeningWithPermission(result)', '"startListening" -> { val engine = call.argument<String>("engine") ?: "pytorch"; startListeningWithPermission(result, engine) }')
text = text.replace('if (hasMicrophonePermission()) { startWakeWordService();', 'if (hasMicrophonePermission()) { startWakeWordService(currentEngine);')
text = text.replace('if (granted) { startWakeWordService();', 'if (granted) { startWakeWordService(currentEngine);')
text = text.replace('private fun startWakeWordService() {', 'private fun startWakeWordService(engine: String = currentEngine) {')
text = text.replace('action = WakeWordService.ACTION_START_LISTENING', 'action = WakeWordService.ACTION_START_LISTENING\n            putExtra(WakeWordService.EXTRA_ENGINE_TYPE, engine)')

with open('android/app/src/main/kotlin/com/example/rushdey/MainActivity.kt', 'w', encoding='utf-8') as f:
    f.write(text)
