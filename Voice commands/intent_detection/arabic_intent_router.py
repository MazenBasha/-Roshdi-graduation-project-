from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "intent_phrases_ar.json"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def normalize_arabic(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ة": "ه",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def load_intent_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def phrase_score(text: str, phrase: str) -> float:
    text = normalize_arabic(text)
    phrase = normalize_arabic(phrase)
    if not text or not phrase:
        return 0.0
    if phrase in text:
        return 1.0
    return SequenceMatcher(None, text, phrase).ratio()


def detect_intent(text: str, config: dict | None = None, threshold: float = 0.55) -> tuple[str, float]:
    config = config or load_intent_config()
    normalized = normalize_arabic(text)
    if not normalized:
        return "none", 0.0

    best_intent = "unknown"
    best_score = threshold

    for intent, phrases in config["intents"].items():
        score = max((phrase_score(normalized, phrase) for phrase in phrases), default=0.0)
        if score > best_score:
            best_intent = intent
            best_score = score

    return best_intent, float(best_score if best_intent != "unknown" else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rushdey Arabic Vosk intent router")
    parser.add_argument("text", nargs="*", help="Arabic text recognized by Vosk")
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    text = " ".join(args.text).strip()
    intent, confidence = detect_intent(text, threshold=args.threshold)
    print(json.dumps({"text": text, "intent": intent, "confidence": confidence}, ensure_ascii=False))


if __name__ == "__main__":
    main()
