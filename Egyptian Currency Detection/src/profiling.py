"""
Lightweight performance profiling for inference.

Used to measure latency / memory / throughput on the CPU-only setup
without pulling in heavyweight benchmark frameworks.
"""

import time
import tracemalloc
from typing import Callable, Dict, List

try:
    import psutil
    _PROC = psutil.Process()
except Exception:
    psutil = None
    _PROC = None


def _rss_mb() -> float:
    if _PROC is None:
        return 0.0
    try:
        return _PROC.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _cpu_percent() -> float:
    if _PROC is None:
        return 0.0
    try:
        # interval=None reports the percent since the last call, non-blocking.
        return _PROC.cpu_percent(interval=None)
    except Exception:
        return 0.0


class PerformanceProfiler:
    @staticmethod
    def profile_inference(func: Callable, *args, **kwargs) -> Dict:
        """Run `func(*args, **kwargs)` and return timing + memory data."""
        mem_before = _rss_mb()
        if _PROC is not None:
            _PROC.cpu_percent(interval=None)  # prime the counter

        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            error = None
        except Exception as e:
            result = None
            error = repr(e)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_after = _rss_mb()
        return {
            "elapsed_ms": round(elapsed_ms, 3),
            "memory_used_mb": round(max(mem_after - mem_before, 0.0), 3),
            "memory_peak_mb": round(peak_bytes / (1024 * 1024), 3),
            "cpu_percent": round(_cpu_percent(), 2),
            "result": result,
            "error": error,
        }

    @staticmethod
    def profile_batch(model, image_paths: List[str], batch_size: int = 4) -> Dict:
        """Profile inference on multiple images, batched."""
        import cv2  # local import keeps the module importable without cv2 for unit tests

        n = len(image_paths)
        if n == 0:
            return {
                "num_images": 0,
                "batch_size": batch_size,
                "total_time_ms": 0.0,
                "avg_time_per_image_ms": 0.0,
                "avg_memory_per_image_mb": 0.0,
                "peak_memory_mb": 0.0,
                "images_per_second": 0.0,
                "times_per_batch": [],
            }

        times_per_batch: List[float] = []
        mem_deltas: List[float] = []
        peak_mem = _rss_mb()

        total_t0 = time.perf_counter()
        for start in range(0, n, batch_size):
            batch_paths = image_paths[start : start + batch_size]
            frames = [cv2.imread(p) for p in batch_paths]
            frames = [f for f in frames if f is not None]
            if not frames:
                times_per_batch.append(0.0)
                continue

            mem_before = _rss_mb()
            t0 = time.perf_counter()
            model.predict(source=frames, verbose=False)
            batch_ms = (time.perf_counter() - t0) * 1000
            mem_after = _rss_mb()

            times_per_batch.append(round(batch_ms, 3))
            mem_deltas.append(max(mem_after - mem_before, 0.0))
            peak_mem = max(peak_mem, mem_after)

        total_ms = (time.perf_counter() - total_t0) * 1000
        avg_ms = total_ms / n if n else 0.0
        avg_mem = (sum(mem_deltas) / len(mem_deltas)) if mem_deltas else 0.0
        ips = (n / (total_ms / 1000.0)) if total_ms > 0 else 0.0

        return {
            "num_images": n,
            "batch_size": batch_size,
            "total_time_ms": round(total_ms, 3),
            "avg_time_per_image_ms": round(avg_ms, 3),
            "avg_memory_per_image_mb": round(avg_mem, 3),
            "peak_memory_mb": round(peak_mem, 3),
            "images_per_second": round(ips, 3),
            "times_per_batch": times_per_batch,
        }