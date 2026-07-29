"""Temporal historical record models.

Every timestamp on these models must be timezone-aware — naive
datetimes are rejected outright rather than silently assumed to be UTC.
Source (provider) timestamps are always kept separate from ingestion
timestamps so the two can never be confused.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


def _require_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field_name} must be timezone-aware (got a naive datetime)")
    return value


class MatchRecordStatus(str, Enum):
    """Lifecycle status of a historical match record."""

    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    RETIRED = "retired"
    CANCELLED = "cancelled"


COMPLETED_STATUSES = frozenset({MatchRecordStatus.FINISHED, MatchRecordStatus.RETIRED})


class DataQualityMetadata(BaseModel):
    """Lightweight, per-record data-quality signal carried alongside historical data."""

    completeness_score: float = Field(default=100.0, ge=0.0, le=100.0)
    warnings: list[str] = Field(default_factory=list)


class HistoricalPlayerRecord(BaseModel):
    """A player as known to the historical repository, tied to one provider identity."""

    id: str = Field(..., min_length=1, description="Stable internal player ID")
    name: str = Field(..., min_length=1)
    country: str | None = None
    provider: str = Field(..., min_length=1)
    provider_player_id: str = Field(..., min_length=1)
    ingested_at: datetime

    @model_validator(mode="after")
    def _validate_timestamps(self) -> "HistoricalPlayerRecord":
        _require_aware(self.ingested_at, field_name="ingested_at")
        return self


class HistoricalSetRecord(BaseModel):
    """A single completed set's score within a historical match, from a fixed A/B perspective."""

    set_number: int = Field(..., ge=1)
    player_a_points: int = Field(..., ge=0)
    player_b_points: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _not_a_tie(self) -> "HistoricalSetRecord":
        if self.player_a_points == self.player_b_points:
            raise ValueError("A completed set cannot end in a tie")
        return self

    @property
    def winner(self) -> str:
        return "player_a" if self.player_a_points > self.player_b_points else "player_b"

    @property
    def margin(self) -> int:
        return self.player_a_points - self.player_b_points


class HistoricalMatchRecord(BaseModel):
    """A single historical match, with explicit provider/ingestion provenance and timing."""

    id: str = Field(..., min_length=1, description="Stable internal match ID")
    provider: str = Field(..., min_length=1, description="Source/provider name")
    provider_match_id: str = Field(..., min_length=1, description="Match ID within that provider")
    competition_id: str | None = None
    competition_name: str | None = None
    player_a_id: str = Field(..., min_length=1)
    player_b_id: str = Field(..., min_length=1)
    scheduled_at: datetime
    actual_start_at: datetime | None = None
    completed_at: datetime | None = None
    status: MatchRecordStatus = MatchRecordStatus.SCHEDULED
    best_of: int | None = Field(default=None, ge=1)
    winner_id: str | None = None
    sets: list[HistoricalSetRecord] = Field(default_factory=list)
    provider_timestamp: datetime
    ingested_at: datetime
    data_quality: DataQualityMetadata = Field(default_factory=DataQualityMetadata)

    @model_validator(mode="after")
    def _validate_timestamps_are_aware(self) -> "HistoricalMatchRecord":
        _require_aware(self.scheduled_at, field_name="scheduled_at")
        if self.actual_start_at is not None:
            _require_aware(self.actual_start_at, field_name="actual_start_at")
        if self.completed_at is not None:
            _require_aware(self.completed_at, field_name="completed_at")
        _require_aware(self.provider_timestamp, field_name="provider_timestamp")
        _require_aware(self.ingested_at, field_name="ingested_at")
        return self

    @model_validator(mode="after")
    def _validate_timestamp_order(self) -> "HistoricalMatchRecord":
        if self.actual_start_at is not None and self.actual_start_at < self.scheduled_at:
            # A match can start early relative to a nominal schedule slot only within reason;
            # we do not fabricate a "reasonable" bound, so we only reject a start recorded
            # before the match was even known to be scheduled is not itself an error — but a
            # completion before a start is always contradictory, checked below.
            pass
        if self.completed_at is not None:
            if self.actual_start_at is not None and self.completed_at < self.actual_start_at:
                raise ValueError("completed_at cannot precede actual_start_at")
            if self.actual_start_at is None and self.completed_at < self.scheduled_at:
                raise ValueError("completed_at cannot precede scheduled_at")
        return self

    @model_validator(mode="after")
    def _validate_players_and_winner(self) -> "HistoricalMatchRecord":
        if self.player_a_id == self.player_b_id:
            raise ValueError("player_a_id and player_b_id must differ")
        if self.winner_id is not None and self.winner_id not in (self.player_a_id, self.player_b_id):
            raise ValueError("winner_id must be one of player_a_id or player_b_id")
        return self

    @model_validator(mode="after")
    def _validate_status_consistency(self) -> "HistoricalMatchRecord":
        if self.status in COMPLETED_STATUSES and self.winner_id is None:
            raise ValueError(f"status {self.status.value!r} requires a winner_id")
        if self.status not in COMPLETED_STATUSES and self.winner_id is not None:
            raise ValueError(f"status {self.status.value!r} must not have a winner_id")
        if self.status is MatchRecordStatus.SCHEDULED:
            if self.sets or self.actual_start_at is not None or self.completed_at is not None:
                raise ValueError("a scheduled match cannot already have sets, a start, or a completion time")
        return self

    @model_validator(mode="after")
    def _validate_sets(self) -> "HistoricalMatchRecord":
        seen_numbers: set[int] = set()
        for s in self.sets:
            if s.set_number in seen_numbers:
                raise ValueError(f"duplicate set_number {s.set_number} in match {self.id!r}")
            seen_numbers.add(s.set_number)
        return self

    @property
    def effective_timestamp(self) -> datetime:
        """The most authoritative "this match happened at" timestamp for cutoff comparisons.

        Prefers completion, then actual start, then the original schedule slot.
        """
        return self.completed_at or self.actual_start_at or self.scheduled_at

    @property
    def sets_a_won(self) -> int:
        return sum(1 for s in self.sets if s.winner == "player_a")

    @property
    def sets_b_won(self) -> int:
        return sum(1 for s in self.sets if s.winner == "player_b")


class HistoricalOddsRecord(BaseModel):
    """A single bookmaker odds observation, captured at a specific point in time."""

    id: str = Field(..., min_length=1, description="Stable internal odds observation ID")
    match_id: str = Field(..., min_length=1)
    bookmaker: str = Field(..., min_length=1)
    selection_id: str = Field(..., min_length=1, description="Player/selection ID these odds apply to")
    decimal_odds: float = Field(..., gt=1.0)
    captured_at: datetime
    provider: str = Field(..., min_length=1)
    market_id: str | None = None

    @model_validator(mode="after")
    def _validate_captured_at(self) -> "HistoricalOddsRecord":
        _require_aware(self.captured_at, field_name="captured_at")
        return self
