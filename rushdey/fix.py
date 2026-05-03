with open('android/app/src/main/kotlin/com/example/rushdey/WakeWordService.kt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'put("رشدي' in line and not 'put("رشدي")' in line:
        lines[i] = '                    put("رشدي")\n'
    if 'put("اهلا رشدي' in line and not 'put("اهلا رشدي")' in line:
        lines[i] = '                    put("اهلا رشدي")\n'

with open('android/app/src/main/kotlin/com/example/rushdey/WakeWordService.kt', 'w', encoding='utf-8') as f:
    f.writelines(lines)
