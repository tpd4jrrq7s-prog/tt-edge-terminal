"""Deterministic duplicate detection and conflict classification.

Internal IDs are derived deterministically from `(provider,
provider_record_id)`, so re-importing the exact same provider record
always maps to the same internal ID and is naturally detected as
idempotent — no separate index is needed for that case. Cross-provider
likely-duplicates are detected by comparing normalized players +
scheduled time within a configurable window, reusing
`MatchRepository.list_head_to_head_before` rather than any new
"list all" capability.

Conflicting match results are never auto-merged — only a strictly safe
progression (a still-`scheduled` record being completed by a later
import of the *same* provider record) is treated as `merged_safe`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel

from config.historical_ingestion import HistoricalIngestionSettings
from persistence.models import HistoricalSetRecord, MatchRecordStatus
from persistence.protocols import MatchRepository, OddsRepository


class DuplicateOutcome(str, Enum):
    INSERTED = "inserted"
    SKIPPED_IDEMPOTENT = "skipped_idempotent"
    MERGED_SAFE = "merged_safe"
    QUARANTINED = "quarantined"
    REJECTED_CONFLICT = "rejected_conflict"


class MatchDuplicateDecision(BaseModel):
    outcome: DuplicateOutcome
    internal_match_id: str
    reasons: list[str]
    conflicting_existing_match_id: str | None = None


class OddsDuplicateDecision(BaseModel):
    outcome: DuplicateOutcome
    internal_odds_id: str
    reasons: list[str]
    conflicting_existing_odds_id: str | None = None


def compute_match_internal_id(provider: str, provider_match_id: str) -> str:
    """Deterministic internal match ID: re-importing the same provider record always maps here."""
    return f"{provider}:{provider_match_id}"


def compute_odds_internal_id(
    provider: str, internal_match_id: str, bookmaker: str, selection_id: str, captured_at: datetime
) -> str:
    """Deterministic internal odds ID: same (match, bookmaker, selection, timestamp) always maps here."""
    return f"{provider}:{internal_match_id}:{bookmaker}:{selection_id}:{captured_at.isoformat()}"


def _sets_signature(sets: list[HistoricalSetRecord]) -> tuple:
    return tuple((s.set_number, s.player_a_points, s.player_b_points) for s in sorted(sets, key=lambda s: s.set_number))


def evaluate_match_duplicate(
    match_repository: MatchRepository,
    provider: str,
    provider_match_id: str,
    player_a_id: str,
    player_b_id: str,
    scheduled_at: datetime,
    status: MatchRecordStatus,
    winner_id: str | None,
    sets: list[HistoricalSetRecord],
    settings: HistoricalIngestionSettings,
) -> MatchDuplicateDecision:
    """Classify a candidate match against existing repository state."""
    internal_id = compute_match_internal_id(provider, provider_match_id)
    existing = match_repository.get(internal_id)

    if existing is not None:
        if (
            existing.winner_id == winner_id
            and existing.scheduled_at == scheduled_at
            and existing.status == status
            and _sets_signature(existing.sets) == _sets_signature(sets)
        ):
            return MatchDuplicateDecision(
                outcome=DuplicateOutcome.SKIPPED_IDEMPOTENT,
                internal_match_id=internal_id,
                reasons=["identical re-import of the same provider match id"],
            )

        # A still-scheduled existing record (no result yet) being completed by a later
        # import of the *same* provider record is a safe progression, not a conflict.
        if existing.status is MatchRecordStatus.SCHEDULED and status is not MatchRecordStatus.SCHEDULED:
            return MatchDuplicateDecision(
                outcome=DuplicateOutcome.MERGED_SAFE,
                internal_match_id=internal_id,
                reasons=[f"existing record was still scheduled; safely updating to status {status.value!r}"],
            )

        conflicting_field = "winner" if existing.winner_id != winner_id else (
            "scheduled_at" if existing.scheduled_at != scheduled_at else "set scores"
        )
        return MatchDuplicateDecision(
            outcome=DuplicateOutcome.REJECTED_CONFLICT,
            internal_match_id=internal_id,
            reasons=[f"same provider match id but conflicting {conflicting_field}"],
            conflicting_existing_match_id=internal_id,
        )

    window = timedelta(seconds=settings.duplicate_timestamp_window_seconds)
    # Deliberately far beyond the window: this scans existing repository state for a
    # likely duplicate by comparing *scheduled_at* directly, not `effective_timestamp`
    # (which prefers completion time and would wrongly exclude an already-completed
    # existing match whose completion falls outside the comparison window).
    far_future_cutoff = scheduled_at + timedelta(days=3650)
    candidates = match_repository.list_head_to_head_before(player_a_id, player_b_id, far_future_cutoff)
    for candidate in candidates:
        if candidate.provider != provider and abs((candidate.scheduled_at - scheduled_at).total_seconds()) <= window.total_seconds():
            return MatchDuplicateDecision(
                outcome=DuplicateOutcome.QUARANTINED,
                internal_match_id=internal_id,
                reasons=[
                    f"likely cross-provider duplicate of existing match {candidate.id!r} "
                    f"from provider {candidate.provider!r}"
                ],
                conflicting_existing_match_id=candidate.id,
            )

    return MatchDuplicateDecision(
        outcome=DuplicateOutcome.INSERTED, internal_match_id=internal_id, reasons=["no existing record found"]
    )


def evaluate_odds_duplicate(
    odds_repository: OddsRepository,
    provider: str,
    internal_match_id: str,
    bookmaker: str,
    selection_id: str,
    decimal_odds: float,
    captured_at: datetime,
) -> OddsDuplicateDecision:
    """Classify a candidate odds observation against existing repository state."""
    internal_id = compute_odds_internal_id(provider, internal_match_id, bookmaker, selection_id, captured_at)
    existing = odds_repository.get(internal_id)

    if existing is None:
        return OddsDuplicateDecision(
            outcome=DuplicateOutcome.INSERTED, internal_odds_id=internal_id, reasons=["new odds observation"]
        )

    if existing.decimal_odds == decimal_odds:
        return OddsDuplicateDecision(
            outcome=DuplicateOutcome.SKIPPED_IDEMPOTENT,
            internal_odds_id=internal_id,
            reasons=["identical re-import of the same odds observation"],
        )

    return OddsDuplicateDecision(
        outcome=DuplicateOutcome.REJECTED_CONFLICT,
        internal_odds_id=internal_id,
        reasons=["same observation key (match/bookmaker/selection/timestamp) but conflicting decimal_odds"],
        conflicting_existing_odds_id=internal_id,
    )
