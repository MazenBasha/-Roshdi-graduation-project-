"""
Lightweight model registry backed by a single JSON file.

This is intentionally tiny: a flat list of model entries on disk plus a few
static methods to register / switch / list.  No DB, no remote bucket - good
enough for a single-developer grad project.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
REGISTRY_FILE = os.path.join(_PROJECT_ROOT, "outputs", "model_registry.json")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _file_size_mb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / (1024 * 1024), 3)
    except OSError:
        return 0.0


class ModelRegistry:
    REGISTRY_FILE = REGISTRY_FILE

    @staticmethod
    def load_registry() -> List[Dict]:
        path = ModelRegistry.REGISTRY_FILE
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return []

    @staticmethod
    def save_registry(registry: List[Dict]) -> None:
        _ensure_dir(ModelRegistry.REGISTRY_FILE)
        with open(ModelRegistry.REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

    @staticmethod
    def register_model(weights_path: str, metrics: Optional[Dict] = None,
                       notes: str = "") -> Dict:
        """Append a model entry.  First registration becomes active by default."""
        metrics = metrics or {}
        registry = ModelRegistry.load_registry()

        # If the same path is already registered, update its metrics + notes
        # in place rather than appending a duplicate row.
        existing = next((e for e in registry if e.get("path") == weights_path), None)
        entry = existing or {"active": len(registry) == 0}

        entry.update({
            "timestamp": datetime.now().isoformat(),
            "path": weights_path,
            "size_mb": _file_size_mb(weights_path),
            "mAP50": metrics.get("mAP50"),
            "mAP50_95": metrics.get("mAP50_95"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "notes": notes,
        })
        entry.setdefault("active", False)

        if existing is None:
            if not registry:
                entry["active"] = True
            registry.append(entry)

        ModelRegistry.save_registry(registry)
        return entry

    @staticmethod
    def set_active_model(index: int) -> Dict:
        registry = ModelRegistry.load_registry()
        if not registry:
            raise ValueError("Registry is empty - register a model first")
        if not 0 <= index < len(registry):
            raise IndexError(
                f"Index {index} out of range (registry has {len(registry)} models)"
            )
        for i, entry in enumerate(registry):
            entry["active"] = (i == index)
        ModelRegistry.save_registry(registry)
        return registry[index]

    @staticmethod
    def get_active_model() -> Optional[str]:
        """Return the active weights path, or None if registry is empty."""
        registry = ModelRegistry.load_registry()
        for entry in registry:
            if entry.get("active") and entry.get("path"):
                return entry["path"]
        if registry:
            return registry[0].get("path")
        return None

    @staticmethod
    def list_models() -> List[Dict]:
        """Print all registered models with status; return the registry list."""
        registry = ModelRegistry.load_registry()
        if not registry:
            print("(no models registered)")
            return registry

        print(f"{'#':<3} {'ACTIVE':<7} {'mAP50':<7} {'mAP50-95':<9} "
              f"{'SIZE_MB':<9} PATH")
        print("-" * 90)
        for i, entry in enumerate(registry):
            active = "yes" if entry.get("active") else ""
            mAP50 = entry.get("mAP50")
            mAP50_95 = entry.get("mAP50_95")
            print(
                f"{i:<3} {active:<7} "
                f"{(f'{mAP50:.3f}' if mAP50 is not None else '-'): <7} "
                f"{(f'{mAP50_95:.3f}' if mAP50_95 is not None else '-'): <9} "
                f"{entry.get('size_mb', 0):<9} {entry.get('path', '')}"
            )
            if entry.get("notes"):
                print(f"      notes: {entry['notes']}")
        return registry