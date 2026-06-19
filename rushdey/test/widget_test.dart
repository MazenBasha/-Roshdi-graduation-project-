import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:rushdey/main.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const modelsChannel = MethodChannel('com.example.rushdey/models');

  setUp(() {
    SharedPreferences.setMockInitialValues({});
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(modelsChannel, (call) async {
          if (call.method == 'listEnrolledPersons') return <String>[];
          if (call.method == 'getCameraConfig') {
            return {'lensFacing': 'back', 'mirror': false};
          }
          return true;
        });
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(modelsChannel, null);
  });

  testWidgets('Rushdey opens with commands, people, and settings', (
    tester,
  ) async {
    await tester.pumpWidget(const RushdeyApp());
    await tester.pumpAndSettle();

    expect(find.text('رشدي'), findsOneWidget);
    expect(find.text('الأوامر'), findsOneWidget);
    expect(find.text('الأشخاص'), findsOneWidget);
    expect(find.text('الإعدادات'), findsOneWidget);
    expect(find.text('أمان'), findsNothing);
    expect(find.text('ابدأ'), findsOneWidget);

    await tester.tap(find.text('الأشخاص'));
    await tester.pumpAndSettle();

    expect(find.text('الأشخاص المسجلون'), findsOneWidget);
    expect(find.text('لا يوجد أشخاص بعد'), findsOneWidget);
  });
}
