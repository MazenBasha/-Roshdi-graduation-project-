import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = Path("D:/Zewail City/GP/Rushdiee_bot/wake_word/clips_exact")
CLEANED_DIR = BASE_DIR / "cleaned_dataset"
POSITIVE_DIR = CLEANED_DIR / "positive"
NEGATIVE_DIR = CLEANED_DIR / "negative"
IN_PROGRESS_DIR = BASE_DIR / "in_progress"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
CLAIM_TIMEOUT_SECONDS = 30 * 60  # Reclaim stale claims after 30 minutes.

ASSIGNMENT_LOCK = threading.Lock()


def ensure_structure() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
    IN_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


def utc_now() -> float:
    return time.time()


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def find_next_raw_file() -> Path | None:
    candidates = [p for p in RAW_DIR.rglob("*") if is_audio_file(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: str(p).lower())
    return candidates[0]


def cleanup_empty_folders(start_dir: Path, stop_dir: Path) -> None:
    current = start_dir
    stop_dir = stop_dir.resolve()

    while True:
        try:
            current_resolved = current.resolve()
        except FileNotFoundError:
            break

        if current_resolved == stop_dir:
            break

        if not current.exists() or not current.is_dir():
            break

        try:
            current.rmdir()
        except OSError:
            break

        current = current.parent


def token_to_audio_path(token: str) -> Path | None:
    for p in IN_PROGRESS_DIR.glob(f"{token}.*"):
        if p.suffix.lower() in AUDIO_EXTENSIONS:
            return p
    return None


def token_meta_path(token: str) -> Path:
    return IN_PROGRESS_DIR / f"{token}.json"


def load_meta(token: str) -> dict | None:
    meta_path = token_meta_path(token)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_meta(token: str, meta: dict) -> None:
    token_meta_path(token).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def remove_assignment_files(token: str) -> None:
    audio_path = token_to_audio_path(token)
    if audio_path and audio_path.exists():
        try:
            audio_path.unlink()
        except OSError:
            pass

    meta_path = token_meta_path(token)
    if meta_path.exists():
        try:
            meta_path.unlink()
        except OSError:
            pass


def get_claim_for_user(user_id: str) -> tuple[str, dict, Path] | None:
    if not user_id:
        return None

    now_ts = utc_now()

    for meta_file in IN_PROGRESS_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if meta.get("claimed_by") != user_id:
            continue

        claimed_at = float(meta.get("claimed_at", 0.0) or 0.0)
        token = meta_file.stem
        audio_path = token_to_audio_path(token)

        if audio_path is None or not audio_path.exists():
            try:
                meta_file.unlink()
            except OSError:
                pass
            continue

        # If stale, reclaim it instead of reusing it.
        if now_ts - claimed_at > CLAIM_TIMEOUT_SECONDS:
            reclaim_assignment(token, meta, audio_path)
            continue

        return token, meta, audio_path

    return None


def reclaim_assignment(token: str, meta: dict, audio_path: Path) -> None:
    original_rel = meta.get("original_rel")
    if original_rel:
        restore_target = RAW_DIR / original_rel
        restore_target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(audio_path), str(restore_target))
        except OSError:
            pass
    else:
        fallback_target = RAW_DIR / "recovered" / audio_path.name
        fallback_target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(audio_path), str(fallback_target))
        except OSError:
            pass

    meta_path = token_meta_path(token)
    if meta_path.exists():
        try:
            meta_path.unlink()
        except OSError:
            pass


def reclaim_stale_assignments() -> None:
    now_ts = utc_now()

    for meta_file in IN_PROGRESS_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        claimed_at = float(meta.get("claimed_at", 0.0) or 0.0)
        if now_ts - claimed_at <= CLAIM_TIMEOUT_SECONDS:
            continue

        token = meta_file.stem
        audio_path = token_to_audio_path(token)
        if audio_path and audio_path.exists():
            reclaim_assignment(token, meta, audio_path)
        else:
            try:
                meta_file.unlink()
            except OSError:
                pass


def claim_next_file(user_id: str) -> tuple[str, dict, Path] | None:
    next_file = find_next_raw_file()
    if next_file is None:
        return None

    token = uuid.uuid4().hex
    in_progress_audio = IN_PROGRESS_DIR / f"{token}{next_file.suffix.lower()}"

    try:
        shutil.move(str(next_file), str(in_progress_audio))
    except OSError:
        return None

    try:
        rel = next_file.relative_to(RAW_DIR)
        original_rel = str(rel).replace("\\", "/")
    except ValueError:
        original_rel = next_file.name

    meta = {
        "token": token,
        "claimed_by": user_id,
        "claimed_at": utc_now(),
        "original_rel": original_rel,
        "original_name": next_file.name,
    }
    save_meta(token, meta)

    cleanup_empty_folders(next_file.parent, RAW_DIR)

    return token, meta, in_progress_audio


def unique_output_name(meta: dict, audio_path: Path) -> str:
    original_rel = meta.get("original_rel", audio_path.name)
    original_name = meta.get("original_name", audio_path.stem)
    digest = hashlib.sha1(original_rel.encode("utf-8")).hexdigest()[:10]
    stem = Path(original_name).stem
    suffix = audio_path.suffix.lower()
    return f"{stem}__{digest}{suffix}"


def get_counts() -> dict:
    raw_count = sum(1 for p in RAW_DIR.rglob("*") if is_audio_file(p))
    in_progress_count = sum(1 for p in IN_PROGRESS_DIR.glob("*") if is_audio_file(p))
    positive_count = sum(1 for p in POSITIVE_DIR.glob("*") if is_audio_file(p))
    negative_count = sum(1 for p in NEGATIVE_DIR.glob("*") if is_audio_file(p))

    return {
        "raw": raw_count,
        "in_progress": in_progress_count,
        "positive": positive_count,
        "negative": negative_count,
        "remaining": raw_count + in_progress_count,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/next")
def next_sample():
    user_id = (request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    with ASSIGNMENT_LOCK:
        reclaim_stale_assignments()

        existing = get_claim_for_user(user_id)
        if existing is None:
            existing = claim_next_file(user_id)

        counts = get_counts()

        if existing is None:
            return jsonify(
                {
                    "has_sample": False,
                    "message": "No more audio files to label.",
                    "counts": counts,
                }
            )

        token, meta, audio_path = existing

        return jsonify(
            {
                "has_sample": True,
                "token": token,
                "audio_url": f"/audio/{token}",
                "display_name": meta.get("original_rel", audio_path.name),
                "counts": counts,
            }
        )


@app.get("/audio/<token>")
def stream_audio(token: str):
    if not token or any(c not in "0123456789abcdef" for c in token.lower()):
        return jsonify({"error": "Invalid token"}), 400

    audio_path = token_to_audio_path(token)
    if audio_path is None or not audio_path.exists():
        return jsonify({"error": "Audio not found"}), 404

    # Conditional responses help efficient repeated playback requests.
    return send_file(audio_path, conditional=True)


@app.post("/label")
def label_sample():
    payload = request.get_json(silent=True) or {}
    token = str(payload.get("token", "")).strip()
    label = str(payload.get("label", "")).strip().lower()
    user_id = str(payload.get("user_id", "")).strip()

    if not token or not user_id:
        return jsonify({"error": "token and user_id are required"}), 400
    if label not in {"positive", "negative"}:
        return jsonify({"error": "label must be positive or negative"}), 400

    with ASSIGNMENT_LOCK:
        meta = load_meta(token)
        if meta is None:
            return jsonify({"error": "Assignment not found or already processed"}), 404

        if meta.get("claimed_by") != user_id:
            return jsonify({"error": "This sample is assigned to another user"}), 409

        audio_path = token_to_audio_path(token)
        if audio_path is None or not audio_path.exists():
            remove_assignment_files(token)
            return jsonify({"error": "Audio file missing for assignment"}), 404

        destination_dir = POSITIVE_DIR if label == "positive" else NEGATIVE_DIR
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination_name = unique_output_name(meta, audio_path)
        destination_path = destination_dir / destination_name

        # Ensure no accidental overwrite if hash collision occurs.
        if destination_path.exists():
            destination_path = destination_dir / f"{uuid.uuid4().hex}_{destination_name}"

        try:
            shutil.move(str(audio_path), str(destination_path))
        except OSError as exc:
            return jsonify({"error": f"Failed to move file: {exc}"}), 500

        meta_path = token_meta_path(token)
        if meta_path.exists():
            try:
                meta_path.unlink()
            except OSError:
                pass

        return jsonify({
            "ok": True,
            "saved_to": str(destination_path.relative_to(BASE_DIR)).replace("\\", "/"),
            "counts": get_counts(),
        })


if __name__ == "__main__":
    ensure_structure()
    # Accessible from local network. Example: http://YOUR_LOCAL_IP:5000
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
