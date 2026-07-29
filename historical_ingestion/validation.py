"""Staged validation: structural, semantic, and temporal.

Canonical `Imported*` models are deliberately permissive (see
`historical_ingestion.models`) — every business rule below produces an
explicit, typed `ValidationIssue` rather than relying on Pydantic
exceptions, so a whole batch's issues can be collected and reported
before any accept/reject decision is made.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from config.historical_ingestion import HistoricalIngestionSettings
from historical_ingestion.models import ImportedMatch, ImportedOdds

_ALLOWED_STATUSES = {"scheduled", "live", "finished", "retired", "cancelled"}
_COMPLETED_STATUSES = {"finished", "retired"}

Stage = Literal["structural", "semantic", "temporal"]


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class ValidationIssue(BaseModel):
    severity: IssueSeverity
    code: str
    message: str
    record_id: str
    field_path: str
    stage: Stage
    provider: str


def _issue(
    severity: IssueSeverity, code: str, message: str, record_id: str, field_path: str, stage: Stage, provider: str
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity, code=code, message=message, record_id=record_id,
        field_path=field_path, stage=stage, provider=provider,
    )


# --------------------------------------------------------------------------
# Stage A — structural
# --------------------------------------------------------------------------


def validate_match_structural(match: ImportedMatch) -> list[ValidationIssue]:
    rid = match.provenance.provider_record_id
    provider = match.provenance.provider
    issues: list[ValidationIssue] = []

    if not match.player_a_external_id:
        issues.append(_issue(IssueSeverity.FATAL, "missing_player_a", "player_a_external_id is missing", rid, "player_a_external_id", "structural", provider))
    if not match.player_b_external_id:
        issues.append(_issue(IssueSeverity.FATAL, "missing_player_b", "player_b_external_id is missing", rid, "player_b_external_id", "structural", provider))
    if match.scheduled_at is None:
        issues.append(_issue(IssueSeverity.ERROR, "missing_scheduled_at", "scheduled_at could not be parsed or is missing", rid, "scheduled_at", "structural", provider))
    if match.status_raw is None:
        issues.append(_issue(IssueSeverity.ERROR, "missing_status", "status is missing from the raw record", rid, "status_raw", "structural", provider))
    if match.status is None and match.status_raw is not None:
        issues.append(_issue(IssueSeverity.WARNING, "unmapped_status", f"status {match.status_raw!r} has no canonical mapping", rid, "status", "structural", provider))
    if match.status is not None and match.status not in _ALLOWED_STATUSES:
        issues.append(_issue(IssueSeverity.FATAL, "invalid_status", f"mapped status {match.status!r} is not a recognized canonical status", rid, "status", "structural", provider))
    if match.best_of is not None and match.best_of < 1:
        issues.append(_issue(IssueSeverity.ERROR, "invalid_best_of", f"best_of must be >= 1 (got {match.best_of})", rid, "best_of", "structural", provider))

    for s in match.sets:
        if s.set_number is None:
            issues.append(_issue(IssueSeverity.ERROR, "missing_set_number", "a set is missing its set_number", rid, "sets", "structural", provider))
        if s.player_a_points is None or s.player_b_points is None:
            issues.append(_issue(IssueSeverity.ERROR, "missing_set_score", "a set is missing a point value", rid, "sets", "structural", provider))
        elif s.player_a_points < 0 or s.player_b_points < 0:
            issues.append(_issue(IssueSeverity.FATAL, "negative_set_score", "set scores cannot be negative", rid, "sets", "structural", provider))
    return issues


def validate_odds_structural(odds: ImportedOdds) -> list[ValidationIssue]:
    rid = odds.provenance.provider_record_id
    provider = odds.provenance.provider
    issues: list[ValidationIssue] = []
    if not odds.provider_match_id:
        issues.append(_issue(IssueSeverity.FATAL, "missing_match_id", "provider_match_id is missing", rid, "provider_match_id", "structural", provider))
    if not odds.bookmaker:
        issues.append(_issue(IssueSeverity.FATAL, "missing_bookmaker", "bookmaker is missing", rid, "bookmaker", "structural", provider))
    if not odds.selection_external_id:
        issues.append(_issue(IssueSeverity.FATAL, "missing_selection", "selection_external_id is missing", rid, "selection_external_id", "structural", provider))
    if odds.decimal_odds is None:
        issues.append(_issue(IssueSeverity.FATAL, "missing_odds", "decimal_odds is missing", rid, "decimal_odds", "structural", provider))
    if odds.captured_at is None:
        issues.append(_issue(IssueSeverity.ERROR, "missing_captured_at", "captured_at could not be parsed or is missing", rid, "captured_at", "structural", provider))
    return issues


# --------------------------------------------------------------------------
# Stage B — semantic
# --------------------------------------------------------------------------


def validate_match_semantic(match: ImportedMatch) -> list[ValidationIssue]:
    rid = match.provenance.provider_record_id
    provider = match.provenance.provider
    issues: list[ValidationIssue] = []

    if match.player_a_external_id and match.player_a_external_id == match.player_b_external_id:
        issues.append(_issue(IssueSeverity.FATAL, "same_player", "player_a and player_b are identical", rid, "player_a_external_id", "semantic", provider))

    if match.winner_external_id is not None and match.winner_external_id not in (
        match.player_a_external_id, match.player_b_external_id
    ):
        issues.append(_issue(IssueSeverity.FATAL, "winner_not_participant", "winner_external_id is not one of the two players", rid, "winner_external_id", "semantic", provider))

    if match.status in _COMPLETED_STATUSES and match.winner_external_id is None:
        issues.append(_issue(IssueSeverity.ERROR, "completed_without_winner", f"status {match.status!r} requires a winner", rid, "winner_external_id", "semantic", provider))
    if match.status == "scheduled" and match.winner_external_id is not None:
        issues.append(_issue(IssueSeverity.ERROR, "scheduled_with_winner", "a scheduled match must not have a winner", rid, "winner_external_id", "semantic", provider))

    seen_numbers: set[int] = set()
    for s in match.sets:
        if s.set_number is not None:
            if s.set_number in seen_numbers:
                issues.append(_issue(IssueSeverity.ERROR, "duplicate_set_number", f"duplicate set_number {s.set_number}", rid, "sets", "semantic", provider))
            seen_numbers.add(s.set_number)

    valid_sets = [s for s in match.sets if s.player_a_points is not None and s.player_b_points is not None]
    if match.winner_external_id is not None and valid_sets:
        a_sets = sum(1 for s in valid_sets if s.player_a_points > s.player_b_points)
        b_sets = len(valid_sets) - a_sets
        implied_winner = (
            match.player_a_external_id if a_sets > b_sets else (match.player_b_external_id if b_sets > a_sets else None)
        )
        if implied_winner is not None and implied_winner != match.winner_external_id:
            issues.append(_issue(IssueSeverity.ERROR, "score_result_mismatch", "recorded winner contradicts the set scores", rid, "winner_external_id", "semantic", provider))

    if match.best_of is not None and len(valid_sets) > match.best_of:
        issues.append(_issue(IssueSeverity.ERROR, "sets_exceed_best_of", f"{len(valid_sets)} sets recorded exceeds best_of={match.best_of}", rid, "sets", "semantic", provider))

    return issues


def validate_odds_semantic(odds: ImportedOdds) -> list[ValidationIssue]:
    rid = odds.provenance.provider_record_id
    provider = odds.provenance.provider
    issues: list[ValidationIssue] = []
    if odds.decimal_odds is not None and odds.decimal_odds <= 1.0:
        issues.append(_issue(IssueSeverity.FATAL, "invalid_odds", f"decimal_odds must be > 1.0 (got {odds.decimal_odds})", rid, "decimal_odds", "semantic", provider))
    return issues


# --------------------------------------------------------------------------
# Stage C — temporal
# --------------------------------------------------------------------------


def validate_match_temporal(
    match: ImportedMatch, settings: HistoricalIngestionSettings, now: datetime
) -> list[ValidationIssue]:
    rid = match.provenance.provider_record_id
    provider = match.provenance.provider
    tolerance = timedelta(seconds=settings.timestamp_tolerance_seconds)
    issues: list[ValidationIssue] = []

    if match.completed_at is not None and match.actual_start_at is not None:
        if match.completed_at < match.actual_start_at:
            issues.append(_issue(IssueSeverity.FATAL, "completion_before_start", "completed_at precedes actual_start_at", rid, "completed_at", "temporal", provider))

    if match.provenance.source_timestamp > match.provenance.ingested_at + tolerance:
        issues.append(_issue(IssueSeverity.WARNING, "source_after_ingestion", "provider timestamp is after ingestion timestamp beyond tolerance", rid, "provenance.source_timestamp", "temporal", provider))

    effective = match.completed_at or match.actual_start_at or match.scheduled_at
    if effective is not None and effective > now + tolerance and not settings.allow_future_records:
        issues.append(_issue(IssueSeverity.ERROR, "future_historical_record", "match's effective timestamp is in the future relative to import execution", rid, "scheduled_at", "temporal", provider))

    return issues


def validate_odds_temporal(
    odds: ImportedOdds, match: ImportedMatch | None, settings: HistoricalIngestionSettings, now: datetime
) -> list[ValidationIssue]:
    rid = odds.provenance.provider_record_id
    provider = odds.provenance.provider
    tolerance = timedelta(seconds=settings.timestamp_tolerance_seconds)
    issues: list[ValidationIssue] = []

    if odds.captured_at is not None and odds.captured_at > now + tolerance and not settings.allow_future_records:
        issues.append(_issue(IssueSeverity.ERROR, "future_odds_timestamp", "odds captured_at is in the future relative to import execution", rid, "captured_at", "temporal", provider))

    if match is not None and odds.captured_at is not None:
        effective = match.completed_at or match.actual_start_at or match.scheduled_at
        if effective is not None and odds.captured_at > effective + tolerance:
            issues.append(_issue(IssueSeverity.WARNING, "odds_after_match_lifecycle", "odds captured after the match's own lifecycle ended", rid, "captured_at", "temporal", provider))

    return issues


# --------------------------------------------------------------------------
# Orchestration + policy
# --------------------------------------------------------------------------

Outcome = Literal["accept", "accept_with_warnings", "reject"]


def validate_match(
    match: ImportedMatch, settings: HistoricalIngestionSettings, now: datetime
) -> list[ValidationIssue]:
    return [
        *validate_match_structural(match),
        *validate_match_semantic(match),
        *validate_match_temporal(match, settings, now),
    ]


def validate_odds(
    odds: ImportedOdds, match: ImportedMatch | None, settings: HistoricalIngestionSettings, now: datetime
) -> list[ValidationIssue]:
    return [
        *validate_odds_structural(odds),
        *validate_odds_semantic(odds),
        *validate_odds_temporal(odds, match, settings, now),
    ]


def decide_validation_outcome(issues: list[ValidationIssue], settings: HistoricalIngestionSettings) -> Outcome:
    """Apply the configured validation/warning policy to a record's issues."""
    if any(i.severity == IssueSeverity.FATAL for i in issues):
        return "reject"
    if any(i.severity == IssueSeverity.ERROR for i in issues):
        return "reject" if settings.validation_policy == "strict" else "accept_with_warnings"
    if any(i.severity in (IssueSeverity.WARNING, IssueSeverity.INFO) for i in issues):
        return "accept_with_warnings" if settings.warning_policy == "accept" else "reject"
    return "accept"
