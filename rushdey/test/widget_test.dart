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

  testWidgets('Rushdey opens and navigates to commands', (tester) async {
    await tester.pumpWidget(const RushdieApp());
    await tester.pumpAndSettle();

    expect(find.text('وضع الأمان'), findsOneWidget);
    expect(find.text('أوامر'), findsOneWidget);

    await tester.tap(find.text('أوامر'));
    await tester.pumpAndSettle();

    expect(find.text('الأوامر'), findsOneWidget);
    expect(find.text('ابدأ الاستماع'), findsOneWidget);
  });
}
