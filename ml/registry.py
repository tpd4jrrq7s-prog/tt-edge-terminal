"""An explicit, in-memory model registry — no database, no global singleton.

Each `InMemoryModelRegistry` instance owns its own entries; the caller
constructs and holds it. Entries reference model/metadata file paths
produced by `ml.serialization`, but the registry itself does not read
or write those files — it only tracks the typed metadata.
"""

from __future__ import annotations

from ml.models import ModelRegistryEntry


class InMemoryModelRegistry:
    """Tracks `ModelRegistryEntry` records by `model_id`."""

    def __init__(self) -> None:
        self._entries: dict[str, ModelRegistryEntry] = {}

    def register(self, entry: ModelRegistryEntry) -> None:
        self._entries[entry.model_id] = entry

    def get(self, model_id: str) -> ModelRegistryEntry | None:
        return self._entries.get(model_id)

    def list_entries(self) -> list[ModelRegistryEntry]:
        return sorted(self._entries.values(), key=lambda e: e.training_timestamp)

    def count(self) -> int:
        return len(self._entries)
