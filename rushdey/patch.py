with open('android/app/src/main/kotlin/com/example/rushdey/WakeWordService.kt', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('put("رشدي )', 'put("رشدي")')
text = text.replace('put("اهلا رشدي )', 'put("اهلا رشدي")')
text = text.replace('isInitialized = false', '/* isInitialized = false */')

with open('android/app/src/main/kotlin/com/example/rushdey/WakeWordService.kt', 'w', encoding='utf-8') as f:
    f.write(text)
