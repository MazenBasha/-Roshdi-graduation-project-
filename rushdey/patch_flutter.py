import sys

with open('lib/main.dart', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('bool isListening = false;', 'bool isListening = false;\n  String selectedEngine = "pytorch";')

text = text.replace("wakeWordChannel.invokeMethod('startListening');", "wakeWordChannel.invokeMethod('startListening', {'engine': selectedEngine});")

# Inside _handleWakeWordEvent, when type == 'detected'
new_detected = '''          case 'detected':
            modelsChannel.invokeMethod('speakText', 'نعم، تحت أمرك');
            setState(() {
              status = "✅ سمعتك! قول الأمر (مثال: مين قدامي)";
              intentListening = true;
            });
'''
text = text.replace('''          case 'detected':
            setState(() {
              status = "✅ سمعتك! قول الأمر (مثال: مين قدامي)";
              intentListening = true;
            });''', new_detected)

# Inside _handleIntent
text = text.replace('''      setState(() => status = "📸 بفتح الكاميرا للتعرف على الوجوه...");
      modelsChannel.invokeMethod('captureAndRecognizeFace');''', '''      setState(() => status = "📸 بفتح الكاميرا للتعرف على الوجوه...");
      modelsChannel.invokeMethod('speakText', 'جاري التعرف على الوجه');
      modelsChannel.invokeMethod('captureAndRecognizeFace');''')

text = text.replace('''      setState(() => status = "📸 بفتح الكاميرا لعد الفلوس...");
      modelsChannel.invokeMethod('captureAndDetectCurrency');''', '''      setState(() => status = "📸 بفتح الكاميرا لعد الفلوس...");
      modelsChannel.invokeMethod('speakText', 'جاري عد الفلوس');
      modelsChannel.invokeMethod('captureAndDetectCurrency');''')

# Add the switch UI before the start button
switch_ui = '''                  SwitchListTile(
                    title: const Text("استخدام محرك Vosk للنداء"),
                    subtitle: const Text("تفعيل للتعرف على (رشدي) عبر Vosk"),
                    value: selectedEngine == "vosk",
                    onChanged: isListening ? null : (bool value) {
                      setState(() {
                        selectedEngine = value ? "vosk" : "pytorch";
                      });
                    },
                  ),
                  const SizedBox(height: 16),
'''
text = text.replace('                  ElevatedButton.icon(', switch_ui + '                  ElevatedButton.icon(')

with open('lib/main.dart', 'w', encoding='utf-8') as f:
    f.write(text)
