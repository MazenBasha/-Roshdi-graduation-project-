import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:async';
import 'package:shared_preferences/shared_preferences.dart';

void main() => runApp(const RushdieApp());

class RushdieApp extends StatelessWidget {
  const RushdieApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Rushdie',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(useMaterial3: true, colorSchemeSeed: Colors.indigo),
      home: const MainShell(),
    );
  }
}

enum IntentType { objects, ocr, money, face }

class CameraConfig {
  final bool useFrontCamera;
  final bool mirror;

  const CameraConfig({required this.useFrontCamera, required this.mirror});
}

class AppSettings {
  static const _keyWakeWordEngine = 'wake_word_engine';
  static const _keyCameraLens = 'camera_lens';
  static const _keyCameraMirror = 'camera_mirror';
  static const MethodChannel _modelsChannel = MethodChannel(
    'com.example.rushdey/models',
  );

  static Future<String> getWakeWordEngine() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyWakeWordEngine) ?? 'pytorch';
  }

  static Future<void> setWakeWordEngine(String engine) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyWakeWordEngine, engine);
  }

  static Future<CameraConfig> getCameraConfig() async {
    final prefs = await SharedPreferences.getInstance();
    final lens = prefs.getString(_keyCameraLens) ?? 'back';
    final mirror = prefs.getBool(_keyCameraMirror) ?? false;
    return CameraConfig(useFrontCamera: lens == 'front', mirror: mirror);
  }

  static Future<void> setCameraConfig(CameraConfig config) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
      _keyCameraLens,
      config.useFrontCamera ? 'front' : 'back',
    );
    await prefs.setBool(_keyCameraMirror, config.mirror);
    await syncCameraConfig(config);
  }

  static Future<void> syncCameraConfig(CameraConfig config) async {
    await _modelsChannel.invokeMethod('setCameraConfig', {
      'lensFacing': config.useFrontCamera ? 'front' : 'back',
      'mirror': config.mirror,
    });
  }
}

String intentTitle(IntentType i) {
  switch (i) {
    case IntentType.objects:
      return "ايه اللي قدامي";
    case IntentType.ocr:
      return "اقرا اللي مكتوب";
    case IntentType.money:
      return "دول كام";
    case IntentType.face:
      return "مين ده";
  }
}

IconData intentIcon(IntentType i) {
  switch (i) {
    case IntentType.objects:
      return Icons.remove_red_eye;
    case IntentType.ocr:
      return Icons.text_snippet;
    case IntentType.money:
      return Icons.payments;
    case IntentType.face:
      return Icons.face;
  }
}

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int index = 0;

  final screens = const [
    SafetyScreen(),
    CommandsScreen(),
    PeopleScreen(),
    SettingsScreen(),
  ];

  @override
  void initState() {
    super.initState();
    // Initialize models on startup
    const MethodChannel(
      'com.example.rushdey/models',
    ).invokeMethod('initModels');
    _syncCameraSettings();
  }

  Future<void> _syncCameraSettings() async {
    final config = await AppSettings.getCameraConfig();
    await AppSettings.syncCameraConfig(config);
  }

  @override
  Widget build(BuildContext context) {
    return Directionality(
      textDirection: TextDirection.rtl,
      child: Scaffold(
        body: SafeArea(child: screens[index]),
        bottomNavigationBar: NavigationBar(
          selectedIndex: index,
          onDestinationSelected: (i) => setState(() => index = i),
          destinations: const [
            NavigationDestination(icon: Icon(Icons.shield), label: "أمان"),
            NavigationDestination(icon: Icon(Icons.mic), label: "أوامر"),
            NavigationDestination(icon: Icon(Icons.people), label: "أشخاص"),
            NavigationDestination(icon: Icon(Icons.settings), label: "إعدادات"),
          ],
        ),
      ),
    );
  }
}

// =======================================================
// 1) SAFETY SCREEN (front-end only)
// =======================================================
class SafetyScreen extends StatefulWidget {
  const SafetyScreen({super.key});

  @override
  State<SafetyScreen> createState() => _SafetyScreenState();
}

class _SafetyScreenState extends State<SafetyScreen> {
  bool safetyOn = false;
  String lastWarning = "لا يوجد تحذيرات";

  void toggleSafety(bool v) {
    setState(() {
      safetyOn = v;
      lastWarning = safetyOn
          ? "وضع الأمان شغال (Placeholder)"
          : "وضع الأمان متوقف";
    });
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(title: const Text("وضع الأمان")),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        safetyOn ? Icons.shield : Icons.shield_outlined,
                        size: 28,
                        color: safetyOn ? cs.primary : null,
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          safetyOn ? "الأمان: شغال" : "الأمان: متوقف",
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                      Switch(value: safetyOn, onChanged: toggleSafety),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "آخر تحذير",
                    style: TextStyle(fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: 10),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: cs.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      lastWarning,
                      style: const TextStyle(fontSize: 16),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// =======================================================
// 2) COMMANDS SCREEN (wake word + intents)
// =======================================================
class CommandsScreen extends StatefulWidget {
  const CommandsScreen({super.key});

  @override
  State<CommandsScreen> createState() => _CommandsScreenState();
}

class _CommandsScreenState extends State<CommandsScreen> {
  static const wakeWordChannel = MethodChannel('com.example.rushdey/wakeword');
  static const wakeWordEvents = EventChannel(
    'com.example.rushdey/wakeword_events',
  );
  static const modelsChannel = MethodChannel('com.example.rushdey/models');
  static const modelEvents = EventChannel('com.example.rushdey/model_events');

  bool listening = false;
  String selectedEngine = "pytorch";
  bool intentListening = false;
  bool previewFrozen = false;
  bool modelBusy = false;
  String status = "اضغط 'ابدأ' للاستماع لكلمة التنبيه";
  String lastText = "—";
  String lastIntent = "—";
  String lastModelTitle = "آخر نتيجة";
  String lastModelResult = "—";
  double lastConfidence = 0.0;
  double lastMicLevel = 0.0;
  Uint8List? lastPreviewBytes;

  StreamSubscription? _wwSub;
  StreamSubscription? _modelSub;

  @override
  void initState() {
    super.initState();
    _setupChannels();
    _loadSettings();
    _startCameraPreview();
  }

  Future<void> _loadSettings() async {
    final engine = await AppSettings.getWakeWordEngine();
    if (!mounted) return;
    setState(() => selectedEngine = engine);
  }

  @override
  void dispose() {
    _stopCameraPreview();
    _wwSub?.cancel();
    _modelSub?.cancel();
    super.dispose();
  }

  Future<void> _startCameraPreview() async {
    try {
      await modelsChannel.invokeMethod('startCameraPreview');
    } catch (_) {
      // Capture commands will ask for camera permission if preview cannot start yet.
    }
  }

  Future<void> _stopCameraPreview() async {
    try {
      await modelsChannel.invokeMethod('stopCameraPreview');
    } catch (_) {}
  }

  void _setupChannels() {
    _wwSub = wakeWordEvents.receiveBroadcastStream().listen((event) {
      if (event is Map) {
        final type = event['type'] as String?;
        switch (type) {
          case 'status':
            final eventStatus = event['status'] as String?;
            final engine = event['engine'] as String?;
            setState(() {
              listening = eventStatus == 'listening';
              if (eventStatus == 'fallback_vosk') {
                status = "نموذج التنبيه غير متاح، بستخدم Vosk مؤقتاً...";
              } else {
                final engineName = engine == 'vosk' ? 'Vosk' : 'النموذج';
                status = listening
                    ? "🎤 يستمع بـ $engineName... قل 'رشدي'"
                    : "متوقف";
              }
            });
            break;
          case 'detected':
            modelsChannel.invokeMethod('speakText', 'نعم، تحت أمرك');
            setState(() {
              status = "✅ سمعتك! قول الأمر (مثال: مين قدامي)";
              intentListening = true;
            });

            // Stop wake word temporarily and start intent listening
            wakeWordChannel.invokeMethod('stopListening');
            modelsChannel.invokeMethod('startIntentListening');
            break;
          case 'mic_level':
            setState(
              () => lastMicLevel = (event['level'] as num?)?.toDouble() ?? 0.0,
            );
            break;
          case 'confidence':
            setState(
              () => lastConfidence =
                  (event['confidence'] as num?)?.toDouble() ?? 0.0,
            );
            break;
          case 'error':
            final message = event['message']?.toString() ?? 'خطأ في الاستماع';
            setState(() {
              listening = false;
              status = message;
            });
            break;
        }
      }
    });

    _modelSub = modelEvents.receiveBroadcastStream().listen((event) async {
      if (event is Map) {
        final type = event['type'] as String?;
        switch (type) {
          case 'intent_detected':
            final intentStr = event['intent'] as String?;
            final text = event['text'] as String?;
            setState(() {
              lastText = text ?? "";
              lastIntent = intentStr ?? "";
              intentListening = false;
            });
            _handleIntent(intentStr);
            break;
          case 'camera_preview':
            final img = event['imageBytes'] as Uint8List?;
            if (!previewFrozen && img != null && img.isNotEmpty) {
              setState(() => lastPreviewBytes = img);
            }
            break;
          case 'camera_frozen':
            final img = event['imageBytes'] as Uint8List?;
            setState(() {
              previewFrozen = true;
              modelBusy = true;
              if (img != null && img.isNotEmpty) {
                lastPreviewBytes = img;
              }
            });
            break;
          case 'camera_live':
            setState(() {
              previewFrozen = false;
              modelBusy = false;
            });
            break;
          case 'intent_timeout':
            setState(() {
              status = "❌ لم يتم التعرف على أمر. قل رشدي مرة أخرى.";
              intentListening = false;
            });
            // Restart wake word
            wakeWordChannel.invokeMethod('startListening', {
              'engine': selectedEngine,
            });
            break;
          case 'intent_error':
            setState(() {
              status = "حصل خطأ في التعرف على الأمر. قل رشدي مرة أخرى.";
              intentListening = false;
            });
            modelsChannel.invokeMethod(
              'speakText',
              'مسمعتش الأمر كويس، قول رشدي مرة تانية',
            );
            wakeWordChannel.invokeMethod('startListening', {
              'engine': selectedEngine,
            });
            break;
          case 'vosk_status':
            final voskStatus = event['status']?.toString();
            if (voskStatus == 'listening_intent') {
              setState(() => status = "بسمع الأمر...");
            } else if (voskStatus == 'downloading_model') {
              setState(() => status = "بجهز نموذج التعرف على الكلام...");
            } else if (voskStatus == 'preparing_model') {
              setState(() => status = "بجهز التعرف على الأمر...");
            }
            break;
          case 'face_result':
          case 'currency_result':
          case 'ocr_result':
            final tts = event['ttsText'] as String?;
            final text = event['text'] as String?;
            final name = event['name'] as String?;
            final arabicName = event['arabicName'] as String?;
            final img = event['imageBytes'] as Uint8List?;
            setState(() {
              modelBusy = false;
              status = "النتيجة: $tts";
              if (type == 'ocr_result') {
                lastText = text ?? tts ?? "";
                lastIntent = "OCR";
                lastModelTitle = "النص المقروء";
                lastModelResult = text ?? tts ?? "—";
              } else if (type == 'currency_result') {
                lastModelTitle = "نتيجة العملة";
                lastModelResult = arabicName ?? tts ?? "—";
              } else {
                lastModelTitle = "نتيجة التعرف على الوجه";
                lastModelResult = name ?? tts ?? "—";
              }
              if (img != null && img.isNotEmpty) {
                lastPreviewBytes = img;
              }
            });
            // Restart wake word after result with a small delay for smooth TTS transition
            Future.delayed(const Duration(milliseconds: 1200), () {
              if (mounted) {
                wakeWordChannel.invokeMethod('startListening', {
                  'engine': selectedEngine,
                });
              }
            });
            break;
        }
      }
    });
  }

  void _handleIntent(String? intentStr) {
    if (intentStr == 'face_who_is_in_front') {
      setState(() {
        modelBusy = true;
        status = "📸 بفتح الكاميرا للتعرف على الوجوه...";
      });
      modelsChannel.invokeMethod('speakText', 'جاري التعرف على الوجه');
      _runModelCapture('captureAndRecognizeFace');
    } else if (intentStr == 'currency_count') {
      setState(() {
        modelBusy = true;
        status = "📸 بفتح الكاميرا لعد الفلوس...";
      });
      modelsChannel.invokeMethod('speakText', 'جاري عد الفلوس');
      _runModelCapture('captureAndDetectCurrency');
    } else if (intentStr == 'ocr_read_text') {
      setState(() {
        modelBusy = true;
        status = "📸 بفتح الكاميرا لقراءة النص...";
      });
      modelsChannel.invokeMethod('speakText', 'جاري قراءة النص');
      _runModelCapture('captureAndReadText');
    } else {
      setState(() => status = "الأمر ده لسه مش شغال. جرب حاجة تانية.");
      modelsChannel.invokeMethod('speakText', "الأمر ده لسه مش شغال");
      wakeWordChannel.invokeMethod('startListening', {
        'engine': selectedEngine,
      });
    }
  }

  Future<void> _runModelCapture(String method) async {
    try {
      await _startCameraPreview();
      await modelsChannel.invokeMethod(method);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        modelBusy = false;
        previewFrozen = false;
        status = "حصلت مشكلة في الكاميرا. جرب تاني.";
      });
      modelsChannel.invokeMethod(
        'speakText',
        'حصلت مشكلة في الكاميرا، جرب تاني',
      );
    }
  }

  Future<void> startListening() async {
    try {
      setState(() => status = "ببدأ الاستماع...");
      await wakeWordChannel.invokeMethod('startListening', {
        'engine': selectedEngine,
      });
    } catch (e) {
      setState(() => status = "خطأ: $e");
    }
  }

  Future<void> stopListening() async {
    try {
      await wakeWordChannel.invokeMethod('stopListening');
      if (intentListening) {
        await modelsChannel.invokeMethod('stopIntentListening');
      }
      setState(() {
        listening = false;
        intentListening = false;
        status = "متوقف";
      });
    } catch (e) {
      setState(() => status = "خطأ: $e");
    }
  }

  void runManualIntent(IntentType intent) async {
    // If we're currently listening, stop it first
    if (listening || intentListening) {
      await stopListening();
    }

    setState(() {
      lastText = "أمر يدوي";
      lastIntent = intent.name;
    });

    if (intent == IntentType.face) {
      _handleIntent('face_who_is_in_front');
    } else if (intent == IntentType.money) {
      _handleIntent('currency_count');
    } else if (intent == IntentType.ocr) {
      _handleIntent('ocr_read_text');
    } else {
      _handleIntent('other');
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final intents = [IntentType.ocr, IntentType.money, IntentType.face];

    return Scaffold(
      appBar: AppBar(title: const Text("الأوامر")),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        (listening || intentListening)
                            ? Icons.hearing
                            : Icons.hearing_disabled,
                        color: (listening || intentListening)
                            ? cs.primary
                            : cs.onSurfaceVariant,
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          status,
                          style: const TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 16,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: FilledButton.icon(
                          onPressed: (listening || intentListening)
                              ? null
                              : startListening,
                          icon: const Icon(Icons.record_voice_over),
                          label: const Text("ابدأ الاستماع"),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: (listening || intentListening)
                              ? stopListening
                              : null,
                          icon: const Icon(Icons.stop),
                          label: const Text("إيقاف"),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "النتيجة الأخيرة",
                    style: TextStyle(fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: 10),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: cs.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text("النص: $lastText\nالنية: $lastIntent"),
                        const SizedBox(height: 12),
                        Text(
                          lastModelTitle,
                          style: const TextStyle(fontWeight: FontWeight.w800),
                        ),
                        const SizedBox(height: 6),
                        SelectableText(
                          lastModelResult,
                          textDirection: TextDirection.rtl,
                          style: const TextStyle(
                            fontSize: 18,
                            height: 1.45,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    "معاينة الكاميرا",
                    style: TextStyle(fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: 8),
                  Container(
                    height: 140,
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: cs.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    clipBehavior: Clip.antiAlias,
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        if (lastPreviewBytes == null)
                          const Center(child: Text("لا يوجد معاينة حالياً"))
                        else
                          Image.memory(
                            lastPreviewBytes!,
                            fit: BoxFit.cover,
                            gaplessPlayback: true,
                          ),
                        PositionedDirectional(
                          top: 8,
                          end: 8,
                          child: DecoratedBox(
                            decoration: BoxDecoration(
                              color: Colors.black.withValues(alpha: 0.58),
                              borderRadius: BorderRadius.circular(999),
                            ),
                            child: Padding(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 10,
                                vertical: 4,
                              ),
                              child: Text(
                                previewFrozen || modelBusy ? "CAPTURED" : "LIVE",
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 11,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          GridView.count(
            crossAxisCount: 2,
            mainAxisSpacing: 12,
            crossAxisSpacing: 12,
            childAspectRatio: 1.25,
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            children: intents.map((i) {
              return FilledButton.tonal(
                onPressed: modelBusy ? null : () => runManualIntent(i),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(intentIcon(i), size: 34),
                    const SizedBox(height: 10),
                    Text(intentTitle(i), textAlign: TextAlign.center),
                  ],
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }
}

// =======================================================
// 3) PEOPLE SCREEN (Face Enrollment)
// =======================================================
class PeopleScreen extends StatefulWidget {
  const PeopleScreen({super.key});

  @override
  State<PeopleScreen> createState() => _PeopleScreenState();
}

class _PeopleScreenState extends State<PeopleScreen> {
  static const modelsChannel = MethodChannel('com.example.rushdey/models');
  static const modelEvents = EventChannel('com.example.rushdey/model_events');

  List<String> people = [];
  bool isLoading = true;
  bool isEnrolling = false;
  bool enrollPreviewFrozen = false;
  String enrollStatus = "";
  int enrollTaken = 0;
  int enrollNeeded = 3;
  Uint8List? enrollPreviewBytes;
  StreamSubscription? _modelSub;

  @override
  void initState() {
    super.initState();
    _loadPersons();
    _modelSub = modelEvents.receiveBroadcastStream().listen((event) {
      if (event is Map) {
        final type = event['type'] as String?;
        switch (type) {
          case 'camera_preview':
            final img = event['imageBytes'] as Uint8List?;
            if (isEnrolling && !enrollPreviewFrozen && img != null && img.isNotEmpty) {
              setState(() => enrollPreviewBytes = img);
            }
            break;
          case 'camera_frozen':
            final img = event['imageBytes'] as Uint8List?;
            if (isEnrolling) {
              setState(() {
                enrollPreviewFrozen = true;
                enrollStatus = "جاري فحص الصورة...";
                if (img != null && img.isNotEmpty) {
                  enrollPreviewBytes = img;
                }
              });
            }
            break;
          case 'camera_live':
            if (isEnrolling) {
              setState(() {
                enrollPreviewFrozen = false;
                if (enrollStatus.isEmpty) {
                  enrollStatus = "ثبت الوش قدام الكاميرا";
                }
              });
            }
            break;
          case 'enroll_start':
            setState(() {
              isEnrolling = true;
              enrollPreviewFrozen = false;
              enrollTaken = 0;
              enrollNeeded = (event['photosNeeded'] as num?)?.toInt() ?? 3;
              enrollStatus = "ثبت الوش قدام الكاميرا";
            });
            break;
          case 'enroll_progress':
            final clear = event['clear'] as bool? ?? false;
            setState(() {
              enrollTaken = (event['taken'] as num?)?.toInt() ?? enrollTaken;
              enrollNeeded = (event['needed'] as num?)?.toInt() ?? enrollNeeded;
              enrollStatus = clear
                  ? "تم التقاط صورة واضحة $enrollTaken من $enrollNeeded"
                  : "مش شايف وش واضح. قرب الوش وثبته";
            });
            break;
          case 'enroll_done':
            setState(() {
              isEnrolling = false;
              enrollPreviewFrozen = false;
              enrollStatus = event['message']?.toString() ?? "";
            });
            _stopCameraPreview();
            _loadPersons();
            break;
        }
      }
    });
  }

  @override
  void dispose() {
    _stopCameraPreview();
    _modelSub?.cancel();
    super.dispose();
  }

  Future<void> _startCameraPreview() async {
    try {
      await modelsChannel.invokeMethod('startCameraPreview');
    } catch (_) {}
  }

  Future<void> _stopCameraPreview() async {
    try {
      await modelsChannel.invokeMethod('stopCameraPreview');
    } catch (_) {}
  }

  Future<void> _loadPersons() async {
    try {
      final result = await modelsChannel.invokeMethod('listEnrolledPersons');
      if (result is List) {
        setState(() {
          people = result.map((e) => e.toString()).toList();
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() => isLoading = false);
    }
  }

  void addPerson() async {
    final name = await _askText(
      context,
      title: "إضافة شخص",
      hint: "اكتب الاسم",
    );
    if (name == null || name.trim().isEmpty) return;

    try {
      setState(() {
        isEnrolling = true;
        enrollPreviewFrozen = false;
        enrollTaken = 0;
        enrollNeeded = 3;
        enrollStatus = "بفتح الكاميرا...";
      });
      await _startCameraPreview();
      await modelsChannel.invokeMethod('enrollPerson', {'name': name.trim()});
    } catch (e) {
      await _stopCameraPreview();
      if (!mounted) return;
      setState(() {
        isEnrolling = false;
        enrollPreviewFrozen = false;
        enrollStatus = "";
      });
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('خطأ: $e')));
    }
  }

  void deletePerson(int index) {
    final name = people[index];
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text("حذف شخص"),
        content: Text("متأكد إنك عايز تحذف: $name ؟"),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("إلغاء"),
          ),
          FilledButton(
            onPressed: () async {
              Navigator.pop(context);
              await modelsChannel.invokeMethod('deleteEnrolledPerson', name);
              _loadPersons();
            },
            child: const Text("حذف"),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("الأشخاص"),
        actions: [
          IconButton(
            onPressed: isEnrolling ? null : addPerson,
            icon: const Icon(Icons.person_add),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (isEnrolling || enrollPreviewBytes != null) ...[
            _EnrollmentPreviewCard(
              imageBytes: enrollPreviewBytes,
              frozen: enrollPreviewFrozen,
              status: enrollStatus,
              taken: enrollTaken,
              needed: enrollNeeded,
            ),
            const SizedBox(height: 12),
          ],
          if (isLoading)
            const SizedBox(
              height: 220,
              child: Center(child: CircularProgressIndicator()),
            )
          else if (people.isEmpty)
            const SizedBox(
              height: 220,
              child: Center(child: Text("لا يوجد أشخاص مسجلين")),
            )
          else
            ...people.asMap().entries.map((entry) {
              final i = entry.key;
              final person = entry.value;
              return Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Card(
                  child: ListTile(
                    leading: const Icon(Icons.person),
                    title: Text(
                      person,
                      style: const TextStyle(fontWeight: FontWeight.w700),
                    ),
                    trailing: IconButton(
                      icon: const Icon(Icons.delete, color: Colors.red),
                      onPressed: isEnrolling ? null : () => deletePerson(i),
                    ),
                  ),
                ),
              );
            }),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: isEnrolling ? null : addPerson,
        icon: const Icon(Icons.add),
        label: const Text("إضافة"),
      ),
    );
  }
}

class _EnrollmentPreviewCard extends StatelessWidget {
  final Uint8List? imageBytes;
  final bool frozen;
  final String status;
  final int taken;
  final int needed;

  const _EnrollmentPreviewCard({
    required this.imageBytes,
    required this.frozen,
    required this.status,
    required this.taken,
    required this.needed,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final progress = needed <= 0 ? 0.0 : (taken / needed).clamp(0.0, 1.0);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.camera_alt),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    status.isEmpty ? "ثبت الوش قدام الكاميرا" : status,
                    style: const TextStyle(
                      fontWeight: FontWeight.w800,
                      fontSize: 16,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              height: 220,
              width: double.infinity,
              clipBehavior: Clip.antiAlias,
              decoration: BoxDecoration(
                color: cs.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  if (imageBytes == null)
                    const Center(child: Text("بفتح معاينة الكاميرا..."))
                  else
                    Image.memory(
                      imageBytes!,
                      fit: BoxFit.cover,
                      gaplessPlayback: true,
                    ),
                  PositionedDirectional(
                    top: 8,
                    end: 8,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.58),
                        borderRadius: BorderRadius.circular(999),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 4,
                        ),
                        child: Text(
                          frozen ? "CAPTURED" : "LIVE",
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),
            LinearProgressIndicator(value: progress),
            const SizedBox(height: 8),
            Text("صور واضحة: $taken من $needed"),
          ],
        ),
      ),
    );
  }
}

Future<String?> _askText(
  BuildContext context, {
  required String title,
  required String hint,
}) async {
  final c = TextEditingController();
  return showDialog<String>(
    context: context,
    builder: (_) => AlertDialog(
      title: Text(title),
      content: TextField(
        controller: c,
        autofocus: true,
        decoration: InputDecoration(hintText: hint),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text("إلغاء"),
        ),
        FilledButton(
          onPressed: () => Navigator.pop(context, c.text),
          child: const Text("حفظ"),
        ),
      ],
    ),
  );
}

// =======================================================
// 4) SETTINGS SCREEN (placeholders)
// =======================================================
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool vibration = true;
  bool beeps = true;
  double speechRate = 0.6;
  bool useVoskWakeWord = false;
  bool useFrontCamera = false;
  bool mirrorCamera = false;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final engine = await AppSettings.getWakeWordEngine();
    final camera = await AppSettings.getCameraConfig();
    if (!mounted) return;
    setState(() {
      useVoskWakeWord = engine == 'vosk';
      useFrontCamera = camera.useFrontCamera;
      mirrorCamera = camera.mirror;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("الإعدادات")),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            child: Column(
              children: [
                SwitchListTile(
                  value: vibration,
                  onChanged: (v) => setState(() => vibration = v),
                  title: const Text("اهتزاز"),
                ),
                const Divider(height: 0),
                SwitchListTile(
                  value: beeps,
                  onChanged: (v) => setState(() => beeps = v),
                  title: const Text("أصوات قصيرة"),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Column(
              children: [
                SwitchListTile(
                  value: useVoskWakeWord,
                  onChanged: (v) async {
                    setState(() => useVoskWakeWord = v);
                    await AppSettings.setWakeWordEngine(v ? 'vosk' : 'pytorch');
                  },
                  title: const Text("استخدام Vosk للنداء"),
                  subtitle: const Text(
                    "لو مش متفعل، هيستخدم نموذج الكلمة المنبهة",
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Column(
              children: [
                SwitchListTile(
                  value: useFrontCamera,
                  onChanged: (v) async {
                    setState(() => useFrontCamera = v);
                    await AppSettings.setCameraConfig(
                      CameraConfig(useFrontCamera: v, mirror: mirrorCamera),
                    );
                  },
                  title: const Text("استخدام الكاميرا الأمامية"),
                ),
                const Divider(height: 0),
                SwitchListTile(
                  value: mirrorCamera,
                  onChanged: (v) async {
                    setState(() => mirrorCamera = v);
                    await AppSettings.setCameraConfig(
                      CameraConfig(useFrontCamera: useFrontCamera, mirror: v),
                    );
                  },
                  title: const Text("عكس الصورة (Mirror)"),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: ListTile(
              leading: const Icon(Icons.info),
              title: const Text("عن التطبيق"),
              subtitle: const Text("Rushdie - ML Models Integrated"),
            ),
          ),
        ],
      ),
    );
  }
}
