import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() => runApp(const RushdeyApp());

class RushdeyApp extends StatelessWidget {
  const RushdeyApp({super.key});

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF1F4F6B);

    return MaterialApp(
      title: 'Rushdey',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: seed,
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: const Color(0xFFF6F7F9),
        appBarTheme: const AppBarTheme(
          centerTitle: false,
          elevation: 0,
          scrolledUnderElevation: 0,
          backgroundColor: Color(0xFFF6F7F9),
          foregroundColor: Color(0xFF17242E),
          titleTextStyle: TextStyle(
            color: Color(0xFF17242E),
            fontSize: 20,
            fontWeight: FontWeight.w800,
          ),
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          margin: EdgeInsets.zero,
          color: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
            side: const BorderSide(color: Color(0xFFE3E8EE)),
          ),
        ),
        navigationBarTheme: NavigationBarThemeData(
          height: 68,
          elevation: 0,
          backgroundColor: Colors.white,
          indicatorColor: seed.withValues(alpha: 0.12),
          labelTextStyle: WidgetStateProperty.resolveWith(
            (states) => TextStyle(
              fontSize: 12,
              fontWeight: states.contains(WidgetState.selected)
                  ? FontWeight.w800
                  : FontWeight.w600,
            ),
          ),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            minimumSize: const Size(64, 48),
            textStyle: const TextStyle(fontWeight: FontWeight.w800),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            minimumSize: const Size(64, 48),
            textStyle: const TextStyle(fontWeight: FontWeight.w800),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
        ),
      ),
      home: const MainShell(),
    );
  }
}

enum IntentType { ocr, money, face, object }

class CameraConfig {
  final bool useFrontCamera;
  final bool mirror;

  const CameraConfig({required this.useFrontCamera, required this.mirror});
}

class TtsVoiceInfo {
  final String name;
  final String locale;
  final String displayLocale;
  final bool isArabic;
  final bool isEgyptian;
  final bool isLikelyMale;
  final bool isNetworkRequired;
  final int quality;
  final int latency;

  const TtsVoiceInfo({
    required this.name,
    required this.locale,
    required this.displayLocale,
    required this.isArabic,
    required this.isEgyptian,
    required this.isLikelyMale,
    required this.isNetworkRequired,
    required this.quality,
    required this.latency,
  });

  factory TtsVoiceInfo.fromMap(Map<dynamic, dynamic> map) {
    return TtsVoiceInfo(
      name: map['name']?.toString() ?? '',
      locale: map['locale']?.toString() ?? '',
      displayLocale: map['displayLocale']?.toString() ?? '',
      isArabic: map['isArabic'] == true,
      isEgyptian: map['isEgyptian'] == true,
      isLikelyMale: map['isLikelyMale'] == true,
      isNetworkRequired: map['isNetworkRequired'] == true,
      quality: (map['quality'] as num?)?.toInt() ?? 0,
      latency: (map['latency'] as num?)?.toInt() ?? 0,
    );
  }

  String get title {
    final parts = <String>[
      displayLocale.isNotEmpty ? displayLocale : locale,
      if (isLikelyMale) 'ذكر محتمل',
      if (isEgyptian) 'مصري',
    ].where((part) => part.isNotEmpty).toList();
    return parts.isEmpty ? name : parts.join(' - ');
  }

  String get subtitle {
    return [
      name,
      if (isNetworkRequired) 'يحتاج إنترنت' else 'متاح على الجهاز',
    ].join(' - ');
  }
}

class AppSettings {
  static const _keyWakeWordEngine = 'wake_word_engine';
  static const _keyCameraLens = 'camera_lens';
  static const _keyCameraMirror = 'camera_mirror';
  static const _keyCurrencyUseYolo = 'currency_use_yolov8';
  static const _keyTtsVoiceName = 'tts_voice_name';
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

  static Future<bool> getUseYoloCurrencyModel() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_keyCurrencyUseYolo) ?? false;
  }

  static Future<void> setUseYoloCurrencyModel(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_keyCurrencyUseYolo, value);
    await syncCurrencyModelMode(value);
  }

  static Future<void> syncCurrencyModelMode(bool useYolo) async {
    await _modelsChannel.invokeMethod('setCurrencyModelMode', {
      'mode': useYolo ? 'yolo_v8' : 'classic',
      'useYolo': useYolo,
    });
  }

  static Future<String?> getPreferredTtsVoiceName() async {
    final prefs = await SharedPreferences.getInstance();
    final value = prefs.getString(_keyTtsVoiceName);
    return (value == null || value.isEmpty) ? null : value;
  }

  static Future<void> setPreferredTtsVoiceName(String? voiceName) async {
    final prefs = await SharedPreferences.getInstance();
    if (voiceName == null || voiceName.isEmpty) {
      await prefs.remove(_keyTtsVoiceName);
    } else {
      await prefs.setString(_keyTtsVoiceName, voiceName);
    }
    await syncPreferredTtsVoiceName(voiceName);
  }

  static Future<void> syncPreferredTtsVoiceName(String? voiceName) async {
    await _modelsChannel.invokeMethod('setTtsVoice', {'voiceName': voiceName});
  }

  static Future<void> previewTtsVoice() async {
    await _modelsChannel.invokeMethod(
      'speakText',
      'معاك رشدي لمساعدتك وارشادك',
    );
  }

  static Future<List<TtsVoiceInfo>> listTtsVoices() async {
    final result = await _modelsChannel.invokeMapMethod<String, dynamic>(
      'listTtsVoices',
    );
    final rawVoices = (result?['voices'] as List?) ?? const [];
    return rawVoices
        .whereType<Map<dynamic, dynamic>>()
        .map(TtsVoiceInfo.fromMap)
        .where((voice) => voice.name.isNotEmpty)
        .toList();
  }

  static Future<void> openTtsSettings() async {
    await _modelsChannel.invokeMethod('openTtsSettings');
  }

  static Future<void> openTtsInstallData() async {
    await _modelsChannel.invokeMethod('openTtsInstallData');
  }
}

String intentTitle(IntentType intent) {
  switch (intent) {
    case IntentType.ocr:
      return 'قراءة النص';
    case IntentType.money:
      return 'معرفة العملة';
    case IntentType.face:
      return 'معرفة الشخص';
    case IntentType.object:
      return 'كشف العوائق';
  }
}

String intentSubtitle(IntentType intent) {
  switch (intent) {
    case IntentType.ocr:
      return 'اقرأ المكتوب أمامي';
    case IntentType.money:
      return 'دول كام؟';
    case IntentType.face:
      return 'مين قدامي؟';
    case IntentType.object:
      return 'ايه اللي قدامي؟';
  }
}

IconData intentIcon(IntentType intent) {
  switch (intent) {
    case IntentType.ocr:
      return Icons.document_scanner_outlined;
    case IntentType.money:
      return Icons.payments_outlined;
    case IntentType.face:
      return Icons.face_retouching_natural_outlined;
    case IntentType.object:
      return Icons.radar_outlined;
  }
}

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int index = 0;

  final screens = const [CommandsScreen(), PeopleScreen(), SettingsScreen()];

  @override
  void initState() {
    super.initState();
    const MethodChannel(
      'com.example.rushdey/models',
    ).invokeMethod('initModels');
    _syncCameraSettings();
  }

  Future<void> _syncCameraSettings() async {
    final config = await AppSettings.getCameraConfig();
    await AppSettings.syncCameraConfig(config);
    final useYoloCurrency = await AppSettings.getUseYoloCurrencyModel();
    await AppSettings.syncCurrencyModelMode(useYoloCurrency);
    final voiceName = await AppSettings.getPreferredTtsVoiceName();
    await AppSettings.syncPreferredTtsVoiceName(voiceName);
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
            NavigationDestination(
              icon: Icon(Icons.mic_none_outlined),
              selectedIcon: Icon(Icons.mic),
              label: 'الأوامر',
            ),
            NavigationDestination(
              icon: Icon(Icons.people_outline),
              selectedIcon: Icon(Icons.people),
              label: 'الأشخاص',
            ),
            NavigationDestination(
              icon: Icon(Icons.settings_outlined),
              selectedIcon: Icon(Icons.settings),
              label: 'الإعدادات',
            ),
          ],
        ),
      ),
    );
  }
}

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
  String selectedEngine = 'pytorch';
  bool intentListening = false;
  bool previewFrozen = false;
  bool modelBusy = false;
  bool objectDetectionActive = false;
  bool objectDetectionReady = false;
  String status = 'اضغط ابدأ، ثم قل رشدي';
  String lastText = 'لا يوجد';
  String lastIntent = 'لا يوجد';
  String lastModelTitle = 'آخر نتيجة';
  String lastModelResult = 'لا توجد نتيجة بعد';
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
    _wwSub?.cancel();
    _modelSub?.cancel();
    _wwSub = null;
    _modelSub = null;
    unawaited(
      modelsChannel.invokeMethod('stopObjectDetection').catchError((_) {}),
    );
    unawaited(_stopCameraPreview());
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

  void _setupChannels() {
    _wwSub = wakeWordEvents.receiveBroadcastStream().listen((event) {
      if (!mounted || event is! Map) return;
      final type = event['type'] as String?;
      switch (type) {
        case 'status':
          final eventStatus = event['status'] as String?;
          final engine = event['engine'] as String?;
          setState(() {
            listening = eventStatus == 'listening';
            if (engine != null) {
              selectedEngine = engine;
            }
            if (!listening) {
              lastMicLevel = 0.0;
            } else if (engine == 'vosk' && lastMicLevel == 0.0) {
              lastMicLevel = 0.22;
            }
            if (eventStatus == 'fallback_vosk') {
              status = 'نموذج رشدي غير متاح، سيتم استخدام Vosk مؤقتا';
            } else {
              final engineName = engine == 'vosk' ? 'Vosk' : 'نموذج رشدي';
              status = listening ? 'يستمع الآن باستخدام $engineName' : 'متوقف';
            }
          });
          break;
        case 'detected':
          modelsChannel.invokeMethod('speakText', 'نعم، تحت أمرك');
          setState(() {
            status = 'تم سماع رشدي. قل الأمر الآن';
            intentListening = true;
            lastConfidence = ((event['confidence'] as num?)?.toDouble() ?? 1.0)
                .clamp(0.0, 1.0)
                .toDouble();
            if (lastMicLevel < 0.35) {
              lastMicLevel = 0.35;
            }
          });
          wakeWordChannel.invokeMethod('stopListening');
          modelsChannel.invokeMethod('startIntentListening');
          break;
        case 'mic_level':
          setState(
            () => lastMicLevel = _displayMicLevel(event['level'] as num?),
          );
          break;
        case 'confidence':
          setState(
            () => lastConfidence =
                ((event['confidence'] as num?)?.toDouble() ?? 0.0)
                    .clamp(0.0, 1.0)
                    .toDouble(),
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
    });

    _modelSub = modelEvents.receiveBroadcastStream().listen((event) async {
      if (!mounted || event is! Map) return;
      final type = event['type'] as String?;
      switch (type) {
        case 'engines_ready':
          setState(() {
            objectDetectionReady = event['objectDetection'] == true;
          });
          break;
        case 'intent_detected':
          final intentStr = event['intent'] as String?;
          final text = event['text'] as String?;
          setState(() {
            lastText = text?.isNotEmpty == true ? text! : 'لا يوجد';
            lastIntent = _intentArabicName(intentStr);
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
        case 'object_detection_status':
          final active = event['active'] == true;
          final ready = event['ready'] == true;
          final message = event['message']?.toString();
          setState(() {
            objectDetectionActive = active;
            objectDetectionReady = ready;
            modelBusy = false;
            previewFrozen = false;
            lastIntent = 'كشف العوائق';
            lastModelTitle = 'مراقبة الطريق';
            lastModelResult = message?.isNotEmpty == true
                ? message!
                : active
                ? 'يتم مراقبة الطريق أمامك'
                : 'كشف العوائق متوقف';
            status = active
                ? 'كشف العوائق يعمل الآن'
                : ready
                ? 'تم إيقاف كشف العوائق'
                : 'نموذج كشف العوائق غير جاهز';
          });
          break;
        case 'object_detection_result':
          final shouldSpeak = event['shouldSpeak'] == true;
          final message = event['messageAr']?.toString() ?? '';
          final detections = event['detections'] as List?;
          final mainObject = event['mainObject'];
          String summary;
          if (message.isNotEmpty) {
            summary = message;
          } else if (mainObject is Map) {
            final name = mainObject['className']?.toString() ?? 'object';
            final distance = _distanceArabicName(
              mainObject['distanceHint']?.toString(),
            );
            final position = _positionArabicName(
              mainObject['horizontalPosition']?.toString(),
            );
            summary = '$name $distance $position';
          } else {
            final count = detections?.length ?? 0;
            summary = count > 0
                ? 'تم رصد $count أجسام، ولا يوجد عائق قريب'
                : 'لا يوجد عائق قريب أمامك';
          }
          setState(() {
            objectDetectionActive = true;
            objectDetectionReady = event['ready'] == true;
            lastIntent = 'كشف العوائق';
            lastModelTitle = shouldSpeak ? 'تحذير عائق' : 'مراقبة الطريق';
            lastModelResult = summary;
            status = shouldSpeak ? summary : 'مراقبة الطريق أمامك';
          });
          break;
        case 'intent_timeout':
          setState(() {
            status = 'لم أسمع أمرا واضحا. قل رشدي مرة أخرى';
            intentListening = false;
          });
          wakeWordChannel.invokeMethod('startListening', {
            'engine': selectedEngine,
          });
          break;
        case 'intent_error':
          setState(() {
            status = 'حدث خطأ أثناء فهم الأمر. حاول مرة أخرى';
            intentListening = false;
          });
          modelsChannel.invokeMethod(
            'speakText',
            'لم أسمع الأمر بوضوح، قل رشدي مرة أخرى',
          );
          wakeWordChannel.invokeMethod('startListening', {
            'engine': selectedEngine,
          });
          break;
        case 'vosk_status':
          final voskStatus = event['status']?.toString();
          if (voskStatus == 'listening_intent') {
            setState(() {
              status = 'أسمع الأمر الآن';
              intentListening = true;
            });
          } else if (voskStatus == 'downloading_model') {
            setState(() => status = 'يتم تجهيز نموذج التعرف على الكلام');
          } else if (voskStatus == 'preparing_model') {
            setState(() => status = 'يتم تجهيز فهم الأوامر');
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
            status = tts?.isNotEmpty == true ? 'تم: $tts' : 'تم تنفيذ الأمر';
            if (type == 'ocr_result') {
              lastText = text ?? tts ?? 'لا يوجد';
              lastIntent = 'قراءة النص';
              lastModelTitle = 'النص المقروء';
              lastModelResult = text ?? tts ?? 'لا يوجد نص واضح';
            } else if (type == 'currency_result') {
              lastIntent = 'معرفة العملة';
              lastModelTitle = 'نتيجة العملة';
              lastModelResult = arabicName ?? tts ?? 'لم يتم التعرف على العملة';
            } else {
              lastIntent = 'معرفة الشخص';
              lastModelTitle = 'نتيجة التعرف على الشخص';
              lastModelResult = name ?? tts ?? 'لم يتم التعرف على الشخص';
            }
            if (img != null && img.isNotEmpty) {
              lastPreviewBytes = img;
            }
          });
          Future.delayed(const Duration(milliseconds: 1200), () {
            if (mounted) {
              wakeWordChannel.invokeMethod('startListening', {
                'engine': selectedEngine,
              });
            }
          });
          break;
      }
    });
  }

  double _displayMicLevel(num? rawLevel) {
    final raw = (rawLevel?.toDouble() ?? 0.0).clamp(0.0, 1.0).toDouble();
    if (raw == 0.0) return 0.0;

    // PyTorch wake-word RMS levels are usually tiny, while Vosk sends an
    // already-normalized activity heartbeat.
    if (raw < 0.12) {
      return (raw / 0.06).clamp(0.0, 1.0).toDouble();
    }
    return raw;
  }

  String _intentArabicName(String? intent) {
    switch (intent) {
      case 'face_who_is_in_front':
        return 'معرفة الشخص';
      case 'currency_count':
        return 'معرفة العملة';
      case 'ocr_read_text':
        return 'قراءة النص';
      case 'object_obstacle_detection':
        return 'كشف العوائق';
      default:
        return 'غير معروف';
    }
  }

  String _distanceArabicName(String? value) {
    switch (value) {
      case 'very_near':
        return 'قريب جدا';
      case 'near':
        return 'قريب';
      case 'far':
        return 'بعيد';
      default:
        return '';
    }
  }

  String _positionArabicName(String? value) {
    switch (value) {
      case 'left':
        return 'على الشمال';
      case 'center':
        return 'قدامك';
      case 'right':
        return 'على اليمين';
      default:
        return '';
    }
  }

  void _handleIntent(String? intentStr) {
    if (!mounted) return;
    if (intentStr == 'face_who_is_in_front') {
      setState(() {
        modelBusy = true;
        status = 'يتم فتح الكاميرا للتعرف على الشخص';
      });
      modelsChannel.invokeMethod('speakText', 'جاري التعرف على الشخص');
      _runModelCapture('captureAndRecognizeFace');
    } else if (intentStr == 'currency_count') {
      setState(() {
        modelBusy = true;
        status = 'يتم فتح الكاميرا لمعرفة العملة';
      });
      modelsChannel.invokeMethod('speakText', 'جاري معرفة العملة');
      _runModelCapture('captureAndDetectCurrency');
    } else if (intentStr == 'ocr_read_text') {
      setState(() {
        modelBusy = true;
        status = 'يتم فتح الكاميرا لقراءة النص';
      });
      modelsChannel.invokeMethod('speakText', 'جاري قراءة النص');
      _runModelCapture('captureAndReadText');
    } else if (intentStr == 'object_obstacle_detection') {
      _startObjectDetection();
    } else {
      setState(() => status = 'الأمر غير واضح. جرب أمرا آخر');
      modelsChannel.invokeMethod('speakText', 'الأمر غير واضح');
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
        status = 'حدثت مشكلة في الكاميرا. حاول مرة أخرى';
      });
      modelsChannel.invokeMethod(
        'speakText',
        'حدثت مشكلة في الكاميرا، حاول مرة أخرى',
      );
    }
  }

  Future<void> _startObjectDetection() async {
    try {
      if (listening || intentListening) {
        await stopListening();
      }
      if (!mounted) return;
      setState(() {
        modelBusy = false;
        previewFrozen = false;
        objectDetectionActive = true;
        lastIntent = 'كشف العوائق';
        lastModelTitle = 'مراقبة الطريق';
        lastModelResult = 'يتم تشغيل كشف العوائق';
        status = 'يتم تشغيل كشف العوائق';
      });
      final result = await modelsChannel.invokeMapMethod<String, dynamic>(
        'startObjectDetection',
      );
      final ready = result?['ready'] == true;
      final active = result?['active'] == true;
      if (!mounted) return;
      setState(() {
        objectDetectionReady = ready;
        objectDetectionActive = active;
        if (!ready) {
          status = 'نموذج كشف العوائق غير جاهز';
          lastModelResult = 'أضف object_detector.ptl إلى أصول التطبيق';
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        objectDetectionActive = false;
        modelBusy = false;
        status = 'حدثت مشكلة في تشغيل كشف العوائق';
        lastModelTitle = 'كشف العوائق';
        lastModelResult = 'خطأ: $e';
      });
    }
  }

  Future<void> _stopObjectDetection() async {
    try {
      await modelsChannel.invokeMethod('stopObjectDetection');
    } catch (_) {}
    if (!mounted) return;
    setState(() {
      objectDetectionActive = false;
      modelBusy = false;
      status = 'تم إيقاف كشف العوائق';
      lastModelTitle = 'مراقبة الطريق';
      lastModelResult = 'كشف العوائق متوقف';
    });
  }

  Future<void> startListening() async {
    try {
      if (!mounted) return;
      setState(() => status = 'يبدأ الاستماع الآن');
      await wakeWordChannel.invokeMethod('startListening', {
        'engine': selectedEngine,
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => status = 'خطأ: $e');
    }
  }

  Future<void> stopListening() async {
    try {
      await wakeWordChannel.invokeMethod('stopListening');
      if (intentListening) {
        await modelsChannel.invokeMethod('stopIntentListening');
      }
      if (objectDetectionActive) {
        await modelsChannel.invokeMethod('stopObjectDetection');
      }
      if (!mounted) return;
      setState(() {
        listening = false;
        intentListening = false;
        objectDetectionActive = false;
        status = 'متوقف';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => status = 'خطأ: $e');
    }
  }

  void runManualIntent(IntentType intent) async {
    if (listening || intentListening) {
      await stopListening();
    }
    if (!mounted) return;
    if (objectDetectionActive && intent != IntentType.object) {
      await _stopObjectDetection();
    }
    if (!mounted) return;
    if (intent == IntentType.object && objectDetectionActive) {
      await _stopObjectDetection();
      return;
    }
    if (!mounted) return;

    setState(() {
      lastText = 'أمر يدوي';
      lastIntent = intentTitle(intent);
    });

    if (intent == IntentType.face) {
      _handleIntent('face_who_is_in_front');
    } else if (intent == IntentType.money) {
      _handleIntent('currency_count');
    } else if (intent == IntentType.ocr) {
      _handleIntent('ocr_read_text');
    } else {
      _handleIntent('object_obstacle_detection');
    }
  }

  @override
  Widget build(BuildContext context) {
    final intents = [
      IntentType.object,
      IntentType.ocr,
      IntentType.money,
      IntentType.face,
    ];

    return Scaffold(
      appBar: AppBar(
        title: const Text('رشدي'),
        actions: [
          Padding(
            padding: const EdgeInsetsDirectional.only(end: 12),
            child: _StatusPill(
              label: objectDetectionActive
                  ? 'يراقب'
                  : listening || intentListening
                  ? 'يستمع'
                  : 'جاهز',
              active: listening || intentListening || objectDetectionActive,
            ),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 20),
        children: [
          _HeroPanel(
            status: status,
            listening: listening,
            intentListening: intentListening,
            modelBusy: modelBusy,
            objectDetectionActive: objectDetectionActive,
            confidence: lastConfidence,
            micLevel: lastMicLevel,
            onStart: (listening || intentListening || objectDetectionActive)
                ? null
                : startListening,
            onStop: (listening || intentListening || objectDetectionActive)
                ? stopListening
                : null,
          ),
          const SizedBox(height: 14),
          _CameraResultPanel(
            imageBytes: lastPreviewBytes,
            frozen: previewFrozen || modelBusy,
            title: lastModelTitle,
            result: lastModelResult,
            lastText: lastText,
            lastIntent: lastIntent,
          ),
          const SizedBox(height: 14),
          _SectionTitle(
            title: 'تشغيل يدوي',
            subtitle: 'استخدم هذه الأزرار أثناء الاختبار أو عند تعذر الصوت',
          ),
          const SizedBox(height: 10),
          LayoutBuilder(
            builder: (context, constraints) {
              final isCompact = constraints.maxWidth < 560;
              return GridView.builder(
                itemCount: intents.length,
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: isCompact ? 1 : 2,
                  mainAxisSpacing: 10,
                  crossAxisSpacing: 10,
                  mainAxisExtent: isCompact ? 112 : 132,
                ),
                itemBuilder: (_, i) {
                  final intent = intents[i];
                  return _IntentActionCard(
                    intent: intent,
                    enabled: !modelBusy,
                    active:
                        intent == IntentType.object && objectDetectionActive,
                    onTap: () => runManualIntent(intent),
                  );
                },
              );
            },
          ),
        ],
      ),
    );
  }
}

class _HeroPanel extends StatelessWidget {
  final String status;
  final bool listening;
  final bool intentListening;
  final bool modelBusy;
  final bool objectDetectionActive;
  final double confidence;
  final double micLevel;
  final VoidCallback? onStart;
  final VoidCallback? onStop;

  const _HeroPanel({
    required this.status,
    required this.listening,
    required this.intentListening,
    required this.modelBusy,
    required this.objectDetectionActive,
    required this.confidence,
    required this.micLevel,
    required this.onStart,
    required this.onStop,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final active = listening || intentListening || objectDetectionActive;

    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xFF17242E),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                DecoratedBox(
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(10),
                    child: Icon(
                      active ? Icons.hearing : Icons.record_voice_over,
                      color: Colors.white,
                      size: 26,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'المساعد الصوتي',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.w900,
                        ),
                      ),
                      const SizedBox(height: 3),
                      Text(
                        status,
                        style: TextStyle(
                          color: Colors.white.withValues(alpha: 0.78),
                          fontWeight: FontWeight.w600,
                          height: 1.35,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            if (objectDetectionActive) ...[
              const _LiveModePill(
                icon: Icons.radar_outlined,
                label: 'تحذير العوائق يعمل',
              ),
              const SizedBox(height: 12),
            ],
            Row(
              children: [
                Expanded(
                  child: _MetricBar(
                    label: 'ثقة النداء',
                    value: confidence.clamp(0.0, 1.0),
                    color: cs.primaryContainer,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _MetricBar(
                    label: 'مستوى الصوت',
                    value: micLevel.clamp(0.0, 1.0),
                    color: cs.tertiaryContainer,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: onStart,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('ابدأ'),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: onStop,
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.white,
                      side: BorderSide(
                        color: Colors.white.withValues(alpha: 0.55),
                      ),
                    ),
                    icon: const Icon(Icons.stop),
                    label: const Text('إيقاف'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _MetricBar extends StatelessWidget {
  final String label;
  final double value;
  final Color color;

  const _MetricBar({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            color: Colors.white.withValues(alpha: 0.72),
            fontSize: 12,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(999),
          child: LinearProgressIndicator(
            value: value,
            minHeight: 6,
            backgroundColor: Colors.white.withValues(alpha: 0.14),
            valueColor: AlwaysStoppedAnimation<Color>(color),
          ),
        ),
      ],
    );
  }
}

class _LiveModePill extends StatelessWidget {
  final IconData icon;
  final String label;

  const _LiveModePill({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withValues(alpha: 0.18)),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: Colors.white, size: 18),
            const SizedBox(width: 8),
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w800,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CameraResultPanel extends StatelessWidget {
  final Uint8List? imageBytes;
  final bool frozen;
  final String title;
  final String result;
  final String lastText;
  final String lastIntent;

  const _CameraResultPanel({
    required this.imageBytes,
    required this.frozen,
    required this.title,
    required this.result,
    required this.lastText,
    required this.lastIntent,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.camera_alt_outlined, color: cs.primary),
                const SizedBox(width: 8),
                const Expanded(
                  child: Text(
                    'الكاميرا والنتيجة',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w900),
                  ),
                ),
                _StatusPill(label: frozen ? 'لقطة' : 'مباشر', active: frozen),
              ],
            ),
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: AspectRatio(
                aspectRatio: 16 / 9,
                child: ColoredBox(
                  color: cs.surfaceContainerHighest,
                  child: imageBytes == null
                      ? Center(
                          child: Text(
                            'لا توجد معاينة حاليا',
                            style: TextStyle(color: cs.onSurfaceVariant),
                          ),
                        )
                      : Image.memory(
                          imageBytes!,
                          fit: BoxFit.cover,
                          gaplessPlayback: true,
                        ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            DecoratedBox(
              decoration: BoxDecoration(
                color: const Color(0xFFF1F4F7),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            title,
                            style: const TextStyle(fontWeight: FontWeight.w900),
                          ),
                        ),
                        Text(
                          lastIntent,
                          style: TextStyle(
                            color: cs.primary,
                            fontWeight: FontWeight.w800,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    SelectableText(
                      result,
                      textDirection: TextDirection.rtl,
                      style: const TextStyle(
                        fontSize: 18,
                        height: 1.45,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'آخر نص مسموع: $lastText',
                      style: TextStyle(
                        color: cs.onSurfaceVariant,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _IntentActionCard extends StatelessWidget {
  final IntentType intent;
  final bool enabled;
  final bool active;
  final VoidCallback onTap;

  const _IntentActionCard({
    required this.intent,
    required this.enabled,
    this.active = false,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return InkWell(
      onTap: enabled ? onTap : null,
      borderRadius: BorderRadius.circular(8),
      child: Ink(
        decoration: BoxDecoration(
          color: active
              ? cs.primaryContainer.withValues(alpha: 0.55)
              : enabled
              ? Colors.white
              : cs.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: active ? cs.primary : const Color(0xFFE3E8EE),
            width: active ? 1.4 : 1,
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.center,
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(intentIcon(intent), color: cs.primary, size: 24),
              const SizedBox(height: 8),
              Text(
                intentTitle(intent),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontWeight: FontWeight.w900,
                  fontSize: 14,
                ),
              ),
              const SizedBox(height: 3),
              Text(
                intentSubtitle(intent),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  color: cs.onSurfaceVariant,
                  fontWeight: FontWeight.w600,
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

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
  String enrollStatus = '';
  int enrollTaken = 0;
  int enrollNeeded = 3;
  Uint8List? enrollPreviewBytes;
  StreamSubscription? _modelSub;

  @override
  void initState() {
    super.initState();
    _loadPersons();
    _modelSub = modelEvents.receiveBroadcastStream().listen((event) {
      if (event is! Map) return;
      final type = event['type'] as String?;
      switch (type) {
        case 'camera_preview':
          final img = event['imageBytes'] as Uint8List?;
          if (isEnrolling &&
              !enrollPreviewFrozen &&
              img != null &&
              img.isNotEmpty) {
            setState(() => enrollPreviewBytes = img);
          }
          break;
        case 'camera_frozen':
          final img = event['imageBytes'] as Uint8List?;
          if (isEnrolling) {
            setState(() {
              enrollPreviewFrozen = true;
              enrollStatus = 'جاري فحص الصورة';
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
              enrollStatus = enrollStatus.isEmpty
                  ? 'ثبّت الوجه أمام الكاميرا'
                  : enrollStatus;
            });
          }
          break;
        case 'enroll_start':
          setState(() {
            isEnrolling = true;
            enrollPreviewFrozen = false;
            enrollTaken = 0;
            enrollNeeded = (event['photosNeeded'] as num?)?.toInt() ?? 3;
            enrollStatus = 'ثبّت الوجه أمام الكاميرا';
          });
          break;
        case 'enroll_progress':
          final clear = event['clear'] as bool? ?? false;
          setState(() {
            enrollTaken = (event['taken'] as num?)?.toInt() ?? enrollTaken;
            enrollNeeded = (event['needed'] as num?)?.toInt() ?? enrollNeeded;
            enrollStatus = clear
                ? 'تم التقاط صورة واضحة $enrollTaken من $enrollNeeded'
                : 'الوجه غير واضح. قرّبه وثبّته أمام الكاميرا';
          });
          break;
        case 'enroll_done':
          setState(() {
            isEnrolling = false;
            enrollPreviewFrozen = false;
            enrollStatus = event['message']?.toString() ?? '';
          });
          _stopCameraPreview();
          _loadPersons();
          break;
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
    } catch (_) {
      setState(() => isLoading = false);
    }
  }

  void addPerson() async {
    final name = await _askText(
      context,
      title: 'إضافة شخص',
      hint: 'اكتب الاسم',
    );
    if (name == null || name.trim().isEmpty) return;

    try {
      setState(() {
        isEnrolling = true;
        enrollPreviewFrozen = false;
        enrollTaken = 0;
        enrollNeeded = 3;
        enrollStatus = 'يتم فتح الكاميرا';
      });
      await _startCameraPreview();
      await modelsChannel.invokeMethod('enrollPerson', {'name': name.trim()});
    } catch (e) {
      await _stopCameraPreview();
      if (!mounted) return;
      setState(() {
        isEnrolling = false;
        enrollPreviewFrozen = false;
        enrollStatus = '';
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
        title: const Text('حذف شخص'),
        content: Text('هل تريد حذف $name؟'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('إلغاء'),
          ),
          FilledButton(
            onPressed: () async {
              Navigator.pop(context);
              await modelsChannel.invokeMethod('deleteEnrolledPerson', name);
              _loadPersons();
            },
            child: const Text('حذف'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('الأشخاص')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 88),
        children: [
          _SectionTitle(
            title: 'الأشخاص المسجلون',
            subtitle: 'أضف أشخاصا ليتمكن رشدي من التعرف عليهم بالكاميرا',
          ),
          const SizedBox(height: 12),
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
            _EmptyState(
              icon: Icons.person_add_alt_1_outlined,
              title: 'لا يوجد أشخاص بعد',
              subtitle: 'اضغط إضافة ووجه الكاميرا للشخص المراد تسجيله',
            )
          else
            ...people.asMap().entries.map((entry) {
              final i = entry.key;
              final person = entry.value;
              return Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Card(
                  child: ListTile(
                    leading: const CircleAvatar(child: Icon(Icons.person)),
                    title: Text(
                      person,
                      style: const TextStyle(fontWeight: FontWeight.w800),
                    ),
                    subtitle: const Text('قالب وجه محفوظ على الجهاز'),
                    trailing: IconButton(
                      tooltip: 'حذف',
                      icon: const Icon(Icons.delete_outline),
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
        label: const Text('إضافة شخص'),
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
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.camera_alt_outlined, color: cs.primary),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    status.isEmpty ? 'ثبّت الوجه أمام الكاميرا' : status,
                    style: const TextStyle(
                      fontWeight: FontWeight.w900,
                      fontSize: 16,
                    ),
                  ),
                ),
                _StatusPill(label: frozen ? 'لقطة' : 'مباشر', active: frozen),
              ],
            ),
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: AspectRatio(
                aspectRatio: 4 / 3,
                child: ColoredBox(
                  color: cs.surfaceContainerHighest,
                  child: imageBytes == null
                      ? Center(
                          child: Text(
                            'يتم فتح معاينة الكاميرا',
                            style: TextStyle(color: cs.onSurfaceVariant),
                          ),
                        )
                      : Image.memory(
                          imageBytes!,
                          fit: BoxFit.cover,
                          gaplessPlayback: true,
                        ),
                ),
              ),
            ),
            const SizedBox(height: 12),
            LinearProgressIndicator(value: progress),
            const SizedBox(height: 8),
            Text(
              'صور واضحة: $taken من $needed',
              style: const TextStyle(fontWeight: FontWeight.w700),
            ),
          ],
        ),
      ),
    );
  }
}

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool vibration = true;
  bool beeps = true;
  bool useVoskWakeWord = false;
  bool useYoloCurrencyModel = false;
  bool useFrontCamera = false;
  bool mirrorCamera = false;
  bool loadingTtsVoices = false;
  String? selectedTtsVoiceName;
  String selectedTtsVoiceLabel = 'اختيار تلقائي';

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final engine = await AppSettings.getWakeWordEngine();
    final camera = await AppSettings.getCameraConfig();
    final useYoloCurrency = await AppSettings.getUseYoloCurrencyModel();
    final voiceName = await AppSettings.getPreferredTtsVoiceName();
    if (!mounted) return;
    setState(() {
      useVoskWakeWord = engine == 'vosk';
      useYoloCurrencyModel = useYoloCurrency;
      useFrontCamera = camera.useFrontCamera;
      mirrorCamera = camera.mirror;
      selectedTtsVoiceName = voiceName;
      selectedTtsVoiceLabel = voiceName ?? 'اختيار تلقائي';
    });
  }

  Future<void> _showTtsVoicePicker() async {
    setState(() => loadingTtsVoices = true);
    List<TtsVoiceInfo> voices;
    try {
      voices = await AppSettings.listTtsVoices();
    } catch (e) {
      if (!mounted) return;
      setState(() => loadingTtsVoices = false);
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('تعذر تحميل الأصوات: $e')));
      return;
    }

    if (!mounted) return;
    setState(() => loadingTtsVoices = false);

    final visibleVoices = voices.where((voice) => voice.isArabic).toList();
    final shownVoices = visibleVoices.isEmpty ? voices : visibleVoices;
    final currentValue = selectedTtsVoiceName ?? '';

    final selected = await showDialog<String>(
      context: context,
      builder: (dialogContext) {
        return AlertDialog(
          title: const Text('اختيار صوت النطق'),
          content: SizedBox(
            width: double.maxFinite,
            height: 420,
            child: shownVoices.isEmpty
                ? const Center(
                    child: Text(
                      'لا توجد أصوات متاحة حاليا. افتح إعدادات النطق لتحميل أصوات عربية.',
                    ),
                  )
                : ListView(
                    children: [
                      ListTile(
                        leading: Icon(
                          currentValue.isEmpty
                              ? Icons.radio_button_checked
                              : Icons.radio_button_unchecked,
                        ),
                        onTap: () => Navigator.pop(dialogContext, ''),
                        title: const Text('اختيار تلقائي'),
                        subtitle: const Text('يفضل صوت عربي ذكر إن كان متاحا'),
                      ),
                      const Divider(height: 0),
                      ...shownVoices.map((voice) {
                        final selected = voice.name == currentValue;
                        return ListTile(
                          leading: Icon(
                            selected
                                ? Icons.radio_button_checked
                                : Icons.radio_button_unchecked,
                          ),
                          onTap: () => Navigator.pop(dialogContext, voice.name),
                          title: Text(voice.title),
                          subtitle: Text(voice.subtitle),
                        );
                      }),
                    ],
                  ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                unawaited(AppSettings.openTtsInstallData());
              },
              child: const Text('تحميل أصوات'),
            ),
            TextButton(
              onPressed: () {
                unawaited(AppSettings.openTtsSettings());
              },
              child: const Text('إعدادات النظام'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(dialogContext),
              child: const Text('إغلاق'),
            ),
          ],
        );
      },
    );

    if (selected == null) return;

    final voiceName = selected.isEmpty ? null : selected;
    await AppSettings.setPreferredTtsVoiceName(voiceName);
    unawaited(AppSettings.previewTtsVoice());

    TtsVoiceInfo? pickedVoice;
    for (final voice in shownVoices) {
      if (voice.name == voiceName) {
        pickedVoice = voice;
        break;
      }
    }

    if (!mounted) return;
    setState(() {
      selectedTtsVoiceName = voiceName;
      selectedTtsVoiceLabel = voiceName == null
          ? 'اختيار تلقائي'
          : (pickedVoice?.title ?? voiceName);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('الإعدادات')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 20),
        children: [
          const _SectionTitle(
            title: 'تفضيلات التشغيل',
            subtitle: 'إعدادات بسيطة تؤثر على تجربة الاختبار والاستخدام',
          ),
          const SizedBox(height: 12),
          Card(
            child: Column(
              children: [
                SwitchListTile(
                  value: vibration,
                  onChanged: (v) => setState(() => vibration = v),
                  secondary: const Icon(Icons.vibration),
                  title: const Text('اهتزاز'),
                  subtitle: const Text('تنبيه لمسي عند بدء أو انتهاء الأوامر'),
                ),
                const Divider(height: 0),
                SwitchListTile(
                  value: beeps,
                  onChanged: (v) => setState(() => beeps = v),
                  secondary: const Icon(Icons.volume_up_outlined),
                  title: const Text('أصوات قصيرة'),
                  subtitle: const Text('إشارات صوتية بجانب النطق العربي'),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: ListTile(
              leading: const Icon(Icons.record_voice_over_outlined),
              title: const Text('صوت النطق'),
              subtitle: Text(selectedTtsVoiceLabel),
              trailing: loadingTtsVoices
                  ? const SizedBox(
                      width: 22,
                      height: 22,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.chevron_left),
              onTap: loadingTtsVoices ? null : _showTtsVoicePicker,
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: SwitchListTile(
              value: useVoskWakeWord,
              onChanged: (v) async {
                setState(() => useVoskWakeWord = v);
                await AppSettings.setWakeWordEngine(v ? 'vosk' : 'pytorch');
              },
              secondary: const Icon(Icons.graphic_eq),
              title: const Text('استخدام Vosk للنداء'),
              subtitle: const Text('اتركه مغلقا لاستخدام نموذج رشدي المخصص'),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: SwitchListTile(
              value: useYoloCurrencyModel,
              onChanged: (v) async {
                setState(() => useYoloCurrencyModel = v);
                await AppSettings.setUseYoloCurrencyModel(v);
              },
              secondary: const Icon(Icons.payments_outlined),
              title: const Text('استخدام YOLOv8 للعملة'),
              subtitle: const Text('اتركه مغلقا لاستخدام نموذج العملة القديم'),
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
                  secondary: const Icon(Icons.flip_camera_android_outlined),
                  title: const Text('الكاميرا الأمامية'),
                  subtitle: const Text('مفيد لاختبار تسجيل الأشخاص'),
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
                  secondary: const Icon(Icons.flip_outlined),
                  title: const Text('عكس الصورة'),
                  subtitle: const Text('يفضل مع الكاميرا الأمامية فقط'),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          const Card(
            child: ListTile(
              leading: Icon(Icons.info_outline),
              title: Text('عن التطبيق'),
              subtitle: Text(
                'Rushdey - مساعد عربي يعمل بنماذج مدمجة على الجهاز',
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  final String title;
  final String subtitle;

  const _SectionTitle({required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w900),
        ),
        const SizedBox(height: 4),
        Text(
          subtitle,
          style: TextStyle(
            color: cs.onSurfaceVariant,
            fontWeight: FontWeight.w600,
            height: 1.35,
          ),
        ),
      ],
    );
  }
}

class _StatusPill extends StatelessWidget {
  final String label;
  final bool active;

  const _StatusPill({required this.label, required this.active});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: active ? cs.primaryContainer : const Color(0xFFEFF2F5),
        borderRadius: BorderRadius.circular(999),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        child: Text(
          label,
          style: TextStyle(
            color: active ? cs.onPrimaryContainer : cs.onSurfaceVariant,
            fontSize: 12,
            fontWeight: FontWeight.w900,
          ),
        ),
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;

  const _EmptyState({
    required this.icon,
    required this.title,
    required this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(22),
        child: Column(
          children: [
            Icon(icon, size: 42, color: cs.primary),
            const SizedBox(height: 12),
            Text(
              title,
              style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w900),
            ),
            const SizedBox(height: 6),
            Text(
              subtitle,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: cs.onSurfaceVariant,
                height: 1.4,
                fontWeight: FontWeight.w600,
              ),
            ),
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
  final controller = TextEditingController();
  return showDialog<String>(
    context: context,
    builder: (_) => AlertDialog(
      title: Text(title),
      content: TextField(
        controller: controller,
        autofocus: true,
        textInputAction: TextInputAction.done,
        decoration: InputDecoration(hintText: hint),
        onSubmitted: (_) => Navigator.pop(context, controller.text),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('إلغاء'),
        ),
        FilledButton(
          onPressed: () => Navigator.pop(context, controller.text),
          child: const Text('حفظ'),
        ),
      ],
    ),
  );
}
