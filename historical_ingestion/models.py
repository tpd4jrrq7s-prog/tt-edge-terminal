"""Raw source records, source batches, and canonical import records.

Canonical import records (`Imported*`) are deliberately more permissive
than the `persistence.models` records they will eventually become —
staged validation (`historical_ingestion.validation`) is what decides
whether a record is good enough to persist, not the Pydantic schema
itself. This mirrors the same "permissive raw layer, strict validation
stage" split used in Phase 2A's `ingestion.models`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def _require_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware (got a naive datetime)")
    return value


RecordType = Literal["player", "match", "odds", "competition", "ranking"]


class RawRecord(BaseModel):
    """One raw, not-yet-adapted record from a provider source.

    `payload` is intentionally an opaque, provider-specific structure —
    only the adapter configured for `provider` knows how to interpret
    it. This is the one place in the whole pipeline an unstructured
    payload is allowed to appear.
    """

    provider: str = Field(..., min_length=1)
    provider_record_id: str = Field(..., min_length=1)
    record_type: RecordType
    payload: dict[str, Any]
    source_batch_id: str = Field(..., min_length=1)
    source_timestamp: datetime
    fetched_at: datetime

    @model_validator(mode="after")
    def _validate_timestamps(self) -> "RawRecord":
        _require_aware(self.source_timestamp, field_name="source_timestamp")
        _require_aware(self.fetched_at, field_name="fetched_at")
        return self


class SourceBatch(BaseModel):
    """One batch of raw records read from a `HistoricalDataSource`."""

    source_name: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    source_version: str = Field(..., min_length=1)
    batch_id: str = Field(..., min_length=1)
    records: list[RawRecord] = Field(default_factory=list)
    next_cursor: str | None = None
    source_timestamp: datetime
    fetched_at: datetime
    source_metadata: dict[str, str] = Field(default_factory=dict)
    checksum: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_timestamps(self) -> "SourceBatch":
        _require_aware(self.source_timestamp, field_name="source_timestamp")
        _require_aware(self.fetched_at, field_name="fetched_at")
        return self


class SourceHealth(BaseModel):
    """Lightweight health/status metadata a source can report about itself."""

    source_name: str
    provider: str
    healthy: bool
    detail: str
    checked_at: datetime

    @model_validator(mode="after")
    def _validate_checked_at(self) -> "SourceHealth":
        _require_aware(self.checked_at, field_name="checked_at")
        return self


class ImportProvenance(BaseModel):
    """Provenance fields shared by every canonical import record."""

    provider: str = Field(..., min_length=1)
    provider_record_id: str = Field(..., min_length=1)
    source_batch_id: str = Field(..., min_length=1)
    source_timestamp: datetime
    ingested_at: datetime
    raw_fingerprint: str = Field(..., min_length=1)
    mapping_version: str = Field(..., min_length=1)
    warnings: list[str] = Field(default_factory=list)
    raw_payload_ref: str | None = None

    @model_validator(mode="after")
    def _validate_timestamps(self) -> "ImportProvenance":
        _require_aware(self.source_timestamp, field_name="source_timestamp")
        _require_aware(self.ingested_at, field_name="ingested_at")
        return self


class ImportedPlayer(BaseModel):
    """A canonical, not-yet-validated player observation from a provider."""

    provenance: ImportProvenance
    name: str
    country: str | None = None
    external_player_id: str | None = None
    ranking_hint: int | None = None


class ImportedSet(BaseModel):
    """A single set score within an `ImportedMatch`, prior to validation."""

    set_number: int | None = None
    player_a_points: int | None = None
    player_b_points: int | None = None


class ImportedMatch(BaseModel):
    """A canonical, not-yet-validated match observation from a provider."""

    provenance: ImportProvenance
    competition_id: str | None = None
    competition_name: str | None = None
    player_a_external_id: str | None = None
    player_b_external_id: str | None = None
    scheduled_at: datetime | None = None
    actual_start_at: datetime | None = None
    completed_at: datetime | None = None
    status_raw: str | None = None
    status: str | None = Field(default=None, description="Canonical status after adapter mapping; None if unmapped")
    best_of: int | None = None
    winner_external_id: str | None = None
    sets: list[ImportedSet] = Field(default_factory=list)


class ImportedOdds(BaseModel):
    """A canonical, not-yet-validated odds observation from a provider."""

    provenance: ImportProvenance
    provider_match_id: str | None = None
    bookmaker: str | None = None
    selection_external_id: str | None = None
    decimal_odds: float | None = None
    captured_at: datetime | None = None
    market_id: str | None = None


class ImportedCompetition(BaseModel):
    """A canonical, not-yet-validated competition observation from a provider."""

    provenance: ImportProvenance
    name: str
    country: str | None = None
    level: str | None = None
    format: str | None = None
    season: str | None = None
    indoor: bool | None = None
    active_from: datetime | None = None
    active_to: datetime | None = None


class ImportedRanking(BaseModel):
    """A canonical, not-yet-validated ranking observation from a provider."""

    provenance: ImportProvenance
    player_external_id: str | None = None
    ranking: int | None = None
    ranking_points: float | None = None
    effective_at: datetime | None = None


class ImportedBatch(BaseModel):
    """The full set of canonical records adapted from one `SourceBatch`."""

    source_batch_id: str
    provider: str
    players: list[ImportedPlayer] = Field(default_factory=list)
    matches: list[ImportedMatch] = Field(default_factory=list)
    odds: list[ImportedOdds] = Field(default_factory=list)
    competitions: list[ImportedCompetition] = Field(default_factory=list)
    rankings: list[ImportedRanking] = Field(default_factory=list)
