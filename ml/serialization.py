"""Deterministic model + metadata serialization.

Models are saved with `joblib` (never `pickle` directly, and never a
format that executes arbitrary code on load beyond joblib's own
well-known trust model — the same caveat as any pickle-based format:
only load files you trust). Metadata is plain, atomically-written,
alphabetically-keyed JSON. Both writes are atomic (temp file + `os.replace`).
"""

from __future__ import annotations

import json
import os
import tempfile

import joblib

from ml.models import ModelRegistryEntry


def _atomic_write_bytes(path: str, write_fn) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-ml-")
    try:
        os.close(fd)
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_model(model: object, path: str) -> None:
    """Atomically save a fitted model (or calibrated wrapper) with joblib."""
    _atomic_write_bytes(path, lambda tmp_path: joblib.dump(model, tmp_path))


def load_model(path: str) -> object:
    """Load a model previously saved with `save_model`. Only load files you trust."""
    return joblib.load(path)


def save_metadata(entry: ModelRegistryEntry, path: str) -> None:
    """Atomically save a ModelRegistryEntry as pretty-printed, alphabetically-keyed JSON."""
    content = json.dumps(entry.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"

    def _write(tmp_path: str) -> None:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(content)

    _atomic_write_bytes(path, _write)


def load_metadata(path: str) -> ModelRegistryEntry:
    """Load a ModelRegistryEntry previously saved with `save_metadata`."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return ModelRegistryEntry.model_validate(data)
