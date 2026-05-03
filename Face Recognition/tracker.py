"""
Multi-face tracker for the realtime recognition loop.

Why a tracker at all
--------------------
Per-frame recognition without tracking has three failure modes that wreck
the user experience:

1. **Label flicker** — the same face oscillates between "Ali" and "Unknown"
   from frame to frame because cosine similarity drifts around the threshold.
2. **Identity hopping** — when two faces are close, frame N may label them
   {Ali, Bob} and frame N+1 {Bob, Ali} just from detection ordering.
3. **Wasted compute** — re-embedding a face that hasn't moved much since
   last frame contributes nothing.

A track aggregates evidence over time. We:
  - Match new detections to existing tracks by IoU on bounding boxes.
  - Smooth each track's embedding with EMA so the matched name is stable.
  - Vote on the name from the last K matches; only switch label when the
    new name wins the majority. This kills flicker.
  - Drop tracks that haven't been seen for `max_missed` frames.

The tracker is greedy O(N*M) on detections-vs-tracks per frame. For wearable
glasses (typically <8 faces in view), this is microseconds and not a
bottleneck — the embedding forward pass dominates.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import count

import numpy as np


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU on [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """One persistent face track across frames."""
    track_id: int
    bbox: np.ndarray                                      # [x1,y1,x2,y2], updated every frame
    embedding: np.ndarray                                 # EMA-smoothed L2-normalized
    last_name: str | None = None                          # majority-voted display name
    last_similarity: float = 0.0                          # smoothed similarity to DB entry
    missed_frames: int = 0
    age: int = 0
    name_history: deque = field(default_factory=lambda: deque(maxlen=10))
    sim_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def update_bbox(self, bbox: np.ndarray) -> None:
        self.bbox = bbox
        self.missed_frames = 0
        self.age += 1

    def update_embedding(self, new_emb: np.ndarray, alpha: float = 0.3) -> None:
        """EMA-smooth the track's embedding. alpha=0.3 gives ~3-frame half-life."""
        emb = (1 - alpha) * self.embedding + alpha * new_emb
        norm = np.linalg.norm(emb)
        self.embedding = emb / max(norm, 1e-10)

    def vote_name(self, raw_name: str | None, raw_sim: float, min_votes: int = 3) -> None:
        """Append a per-frame raw match and update the displayed name by majority.

        We refuse to switch the displayed name until at least `min_votes` of the
        last 10 frames agree. This is what kills label flicker — a single
        misidentified frame can't change what the user sees.
        """
        self.name_history.append(raw_name)
        self.sim_history.append(raw_sim)

        # Smooth similarity with a small EMA for nicer overlay text.
        self.last_similarity = 0.6 * self.last_similarity + 0.4 * raw_sim

        # Majority vote, ignoring None (unknown) unless that IS the majority.
        counts = Counter(self.name_history)
        # If unknown is the strict majority of the window, surface unknown.
        n = len(self.name_history)
        unknown_count = counts.get(None, 0)
        if unknown_count > n // 2:
            self.last_name = None
            return

        # Otherwise pick the most common KNOWN name with at least `min_votes`.
        named = [(name, c) for name, c in counts.items() if name is not None]
        if not named:
            self.last_name = None
            return
        named.sort(key=lambda x: -x[1])
        best_name, best_count = named[0]
        if best_count >= min_votes:
            self.last_name = best_name
        # else: keep whatever we had (sticky)


class FaceTracker:
    """Multi-target IoU + embedding tracker.

    Update with the per-frame list of (bbox, embedding) and get back the
    list of Track objects that survived. A Track persists across frames
    even when MTCNN momentarily drops it (up to `max_missed` frames),
    which avoids the "label disappears for one frame and pops back as
    Unknown" UX bug.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_missed: int = 8,
        ema_alpha: float = 0.30,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.ema_alpha = ema_alpha
        self.tracks: list[Track] = []
        self._next_id = count(1)

    def step(
        self,
        detections: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[tuple[Track, int]]:
        """Process one frame.

        `detections` is a list of (bbox_xyxy, L2-normalized embedding) tuples
        — one per detected face this frame.

        Returns a list of (track, detection_index) pairs for detections that
        matched (or created) a track. Detection_index is the position of the
        matched detection in the input list, so callers can correlate back to
        their per-detection annotations (e.g. landmarks, aligned crop).
        """
        # Step 1: bump every existing track's missed counter; we'll reset it
        # for the ones that match this frame.
        for t in self.tracks:
            t.missed_frames += 1

        n_dets = len(detections)
        n_tracks = len(self.tracks)
        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()
        results: list[tuple[Track, int]] = []

        # Step 2: greedy IoU matching, highest-IoU pair first.
        if n_dets > 0 and n_tracks > 0:
            iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
            for ti, t in enumerate(self.tracks):
                for di, (box, _emb) in enumerate(detections):
                    iou_matrix[ti, di] = _iou(t.bbox, box)
            # Repeatedly take the best remaining (track, det) pair.
            while True:
                ti, di = np.unravel_index(int(np.argmax(iou_matrix)), iou_matrix.shape)
                best = float(iou_matrix[ti, di])
                if best < self.iou_threshold:
                    break
                t = self.tracks[ti]
                box, emb = detections[di]
                t.update_bbox(box)
                t.update_embedding(emb, alpha=self.ema_alpha)
                results.append((t, di))
                matched_dets.add(di)
                matched_tracks.add(ti)
                # Suppress this row & column so they don't match again.
                iou_matrix[ti, :] = -1
                iou_matrix[:, di] = -1

        # Step 3: unmatched detections become new tracks.
        for di, (box, emb) in enumerate(detections):
            if di in matched_dets:
                continue
            t = Track(
                track_id=next(self._next_id),
                bbox=box.copy(),
                embedding=emb.astype(np.float32).copy(),
            )
            t.age = 1
            self.tracks.append(t)
            results.append((t, di))

        # Step 4: drop stale tracks.
        self.tracks = [t for t in self.tracks if t.missed_frames <= self.max_missed]

        return results
