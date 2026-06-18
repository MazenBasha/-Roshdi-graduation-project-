plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.rushdey"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        applicationId = "com.example.rushdey"
        minSdk = 24
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    // Keep model/training-data assets uncompressed so native runtimes can map/copy them reliably.
    aaptOptions {
        noCompress += listOf("tflite", "ptl", "pt", "traineddata")
    }
}

flutter {
    source = "../.."
}

dependencies {
    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.7.3")

    // JSON
    implementation("org.json:json:20230227")

    // TensorFlow Lite for face recognition
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    // SELECT_TF_OPS (needed for MobileFaceNet with custom ops)
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0")

    // Vosk Android for Arabic intent detection
    implementation("com.alphacephei:vosk-android:0.3.47@aar")
    implementation("net.java.dev.jna:jna:5.13.0@aar")

    // AndroidX core (for FileProvider)
    implementation("androidx.core:core-ktx:1.12.0")

    // Guava for ListenableFuture (CameraX)
    implementation("com.google.guava:guava:31.1-android")

    // CameraX
    val camerax_version = "1.3.1"
    implementation("androidx.camera:camera-core:$camerax_version")
    implementation("androidx.camera:camera-camera2:$camerax_version")
    implementation("androidx.camera:camera-lifecycle:$camerax_version")

    // Offline Arabic OCR through Tesseract and bundled ara.traineddata
    implementation("cz.adaptech.tesseract4android:tesseract4android:4.9.0")
    implementation("com.google.mlkit:text-recognition:16.0.1")

    // ML Kit Face Detection
    implementation("com.google.mlkit:face-detection:16.1.6")

    // PyTorch Android (same version as pytorch_lite Flutter plugin)
    implementation("org.pytorch:pytorch_android:2.1.0")
}
