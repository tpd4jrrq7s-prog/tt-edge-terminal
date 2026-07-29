"""Typed import checkpoints and an in-memory `CheckpointStore`.

Checkpoints only advance after a batch's records have been
successfully staged and persisted — see `historical_ingestion.service`
for exactly where that happens. A failed/aborted batch never advances
the checkpoint, so resumption always restarts from the last known-good
cursor.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from historical_ingestion.errors import CheckpointVersionError


def _require_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware (got a naive datetime)")
    return value


class ImportCheckpoint(BaseModel):
    """Durable (within this process) progress marker for one (source, provider) pair."""

    source_name: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    cursor: str | None = None
    last_successful_batch_id: str | None = None
    last_successful_source_timestamp: datetime | None = None
    processed_record_count: int = Field(default=0, ge=0)
    repository_fingerprint: str = ""
    checkpoint_version: int = Field(..., ge=1)
    updated_at: datetime

    @model_validator(mode="after")
    def _validate_timestamps(self) -> "ImportCheckpoint":
        if self.last_successful_source_timestamp is not None:
            _require_aware(self.last_successful_source_timestamp, field_name="last_successful_source_timestamp")
        _require_aware(self.updated_at, field_name="updated_at")
        return self


class InMemoryCheckpointStore:
    """Deterministic, in-memory `CheckpointStore` implementation. No database."""

    def __init__(self, checkpoint_version: int = 1) -> None:
        self._checkpoint_version = checkpoint_version
        self._by_key: dict[tuple[str, str], ImportCheckpoint] = {}

    def get(self, source_name: str, provider: str) -> ImportCheckpoint | None:
        checkpoint = self._by_key.get((source_name, provider))
        if checkpoint is None:
            return None
        if checkpoint.checkpoint_version != self._checkpoint_version:
            raise CheckpointVersionError(
                f"Stored checkpoint version {checkpoint.checkpoint_version} for "
                f"({source_name!r}, {provider!r}) is incompatible with expected "
                f"version {self._checkpoint_version}"
            )
        return checkpoint.model_copy(deep=True)

    def save(self, checkpoint: ImportCheckpoint) -> None:
        if checkpoint.checkpoint_version != self._checkpoint_version:
            raise CheckpointVersionError(
                f"Cannot save checkpoint with version {checkpoint.checkpoint_version}; "
                f"this store expects version {self._checkpoint_version}"
            )
        self._by_key[(checkpoint.source_name, checkpoint.provider)] = checkpoint.model_copy(deep=True)

    def reset(self, source_name: str, provider: str) -> None:
        self._by_key.pop((source_name, provider), None)
