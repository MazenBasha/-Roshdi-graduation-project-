from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "intent_detection"))

from arabic_intent_router import CONFIG_PATH, detect_intent, load_intent_config


TEST_CASES = [
    ("مين قدامي", "face_who_is_in_front"),
    ("ده مين", "face_who_is_in_front"),
    ("دول كام", "currency_count"),
    ("معايا كام فلوس", "currency_count"),
    ("اقرأ المكتوب", "ocr_read_text"),
    ("النص مكتوب ايه", "ocr_read_text"),
    ("ايه اللي قدامي", "object_obstacle_detection"),
    ("فيه عائق قدامي", "object_obstacle_detection"),
    ("الجو جميل", "unknown"),
    ("", "none"),
]


def main() -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit("Install MLflow first: pip install mlflow") from exc

    config = load_intent_config(CONFIG_PATH)
    correct = 0
    results = []

    for text, expected in TEST_CASES:
        predicted, confidence = detect_intent(text, config=config)
        passed = predicted == expected
        correct += int(passed)
        results.append(
            {
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
                "pass": passed,
            }
        )

    accuracy = correct / len(TEST_CASES)
    report_path = ROOT / "mlflow_tracking" / "vosk_intent_eval_report.json"
    report_path.write_text(
        __import__("json").dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mlflow.set_experiment("rushdey-vosk-intent-router")
    with mlflow.start_run(run_name="arabic-intent-router-static-eval"):
        mlflow.log_param("asr_engine", "Vosk Arabic offline model")
        mlflow.log_param("router_type", "keyword + fuzzy phrase matching")
        mlflow.log_param("wake_word", config["wake_word"])
        mlflow.log_metric("static_intent_accuracy", accuracy)
        mlflow.log_artifact(str(CONFIG_PATH), artifact_path="config")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        mlflow.set_tags(
            {
                "component": "vosk_intent_router",
                "runtime": "android_on_device",
                "privacy": "offline_speech_recognition",
            }
        )

    print(f"Static intent accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
