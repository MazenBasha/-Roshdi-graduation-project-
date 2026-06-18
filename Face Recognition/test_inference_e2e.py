"""
End-to-end inference + dynamic enrollment smoke test.

Verifies the spec'd behavior:
    1. First image of person A    -> "unknown" (DB empty)
    2. Enroll person A under name -> DB now has 1 entry
    3. Different image of person A -> recognized as A
    4. Image of person B          -> "unknown" (only A is enrolled)
    5. Enroll person B            -> DB has 2 entries
    6. Another image of A         -> A
    7. Another image of B         -> B

Uses LFW images of George_W_Bush and Tony_Blair (lots of samples each).
"""

import shutil
from pathlib import Path

import cv2

from inference import FaceRecognizer
from utils import FaceDatabase
import config

# Reset DB so the test is reproducible.
if config.FACE_DB_PATH.exists():
    config.FACE_DB_PATH.unlink()
print(f"[test] cleared face DB at {config.FACE_DB_PATH}")

LFW_ROOT = Path("data/sklearn_lfw/lfw_home/lfw_funneled")
A_IMAGES = sorted((LFW_ROOT / "George_W_Bush").glob("*.jpg"))[:4]
B_IMAGES = sorted((LFW_ROOT / "Tony_Blair").glob("*.jpg"))[:3]
assert len(A_IMAGES) >= 4 and len(B_IMAGES) >= 3, "Not enough LFW samples"

print(f"[test] A (George_W_Bush): {len(A_IMAGES)} images")
print(f"[test] B (Tony_Blair):    {len(B_IMAGES)} images")

rec = FaceRecognizer()
print(f"[test] recognizer ready, DB size = {len(rec.db)}\n")


def run(label: str, img_path: Path, recognizer=None):
    img = cv2.imread(str(img_path))
    results = (recognizer or rec).recognize_image(img)
    if not results:
        print(f"  [{label}] {img_path.name}: NO FACE DETECTED")
        return None
    r = results[0]
    name_str = f"'{r.name}'" if r.name else "UNKNOWN"
    print(f"  [{label}] {img_path.name}: {name_str}  (sim={r.similarity:.3f}, det={r.detection_score:.2f})")
    return r


print("Step 1: query A_0 with empty DB -> expect UNKNOWN")
r = run("A0", A_IMAGES[0])
assert r is not None and r.name is None, "expected unknown on empty DB"

print("\nStep 2: enroll A_0 as 'George_W_Bush'")
rec.enroll("George_W_Bush", r.embedding)
print(f"  DB size = {len(rec.db)}, names = {rec.db.names()}")

print("\nStep 3: query A_1 (different image of A) -> expect 'George_W_Bush'")
r = run("A1", A_IMAGES[1])
assert r is not None and r.name == "George_W_Bush", f"expected George_W_Bush, got {r.name if r else None}"

print("\nStep 4: query B_0 (different person, not enrolled) -> expect UNKNOWN")
r = run("B0", B_IMAGES[0])
assert r is not None and r.name is None, f"expected unknown, got {r.name if r else None}"

print("\nStep 5: enroll B_0 as 'Tony_Blair'")
rec.enroll("Tony_Blair", r.embedding)
print(f"  DB size = {len(rec.db)}, names = {rec.db.names()}")

print("\nStep 6: query A_2 -> expect 'George_W_Bush'")
r = run("A2", A_IMAGES[2])
assert r is not None and r.name == "George_W_Bush", f"expected George_W_Bush, got {r.name if r else None}"

print("\nStep 7: query B_1 -> expect 'Tony_Blair'")
r = run("B1", B_IMAGES[1])
assert r is not None and r.name == "Tony_Blair", f"expected Tony_Blair, got {r.name if r else None}"

print("\nStep 8: verify DB persists across recognizer instances")
del rec
rec2 = FaceRecognizer()
print(f"  reloaded DB has {len(rec2.db)} entries: {rec2.db.names()}")
assert sorted(rec2.db.names()) == ["George_W_Bush", "Tony_Blair"]

print("\nStep 9: more rolling-window enrollments accumulate")
for i, p in enumerate(A_IMAGES[1:4], start=1):
    img = cv2.imread(str(p))
    res = rec2.recognize_image(img)
    if res:
        rec2.enroll("George_W_Bush", res[0].embedding)
n_emb = len(rec2.db.records["George_W_Bush"].embeddings)
print(f"  George_W_Bush now has {n_emb} embeddings stored")

print("\nStep 10: re-query A and B after multiple enrollments")
r = run("A3", A_IMAGES[3], recognizer=rec2)
assert r is not None and r.name == "George_W_Bush"
r = run("B2", B_IMAGES[2], recognizer=rec2)
assert r is not None and r.name == "Tony_Blair"

print("\nALL ASSERTIONS PASSED")
print(f"Final DB: {dict((n, len(rec2.db.records[n].embeddings)) for n in rec2.db.names())}")
