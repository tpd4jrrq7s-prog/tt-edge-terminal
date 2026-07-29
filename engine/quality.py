"""Data-quality assessment (0-100) for a match analysis request.

Every deduction is represented as an explicit `DataQualityIssue` so the
final score is always traceable to a concrete, named reason — never an
opaque penalty.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from config.analytics import AnalyticsSettings, get_analytics_settings
from domain.match import Match, MatchStatus
from domain.odds import Odds
from engine.models import DataQualityAssessment, DataQualityIssue, HeadToHeadRecord, HistoricalMatch, IssueSeverity

_SEVERITY_PENALTY = {
    IssueSeverity.INFO: 3.0,
    IssueSeverity.WARNING: 10.0,
    IssueSeverity.CRITICAL: 25.0,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _check_match_completeness(match: Match) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    if match.status is MatchStatus.FINISHED and not match.sets:
        issues.append(
            DataQualityIssue(
                field="match.sets",
                detail="Match is marked finished but no set scores were provided",
                severity=IssueSeverity.WARNING,
            )
        )
    if match.status is MatchStatus.FINISHED and match.sets:
        if match.sets_won_player_one == match.sets_won_player_two:
            issues.append(
                DataQualityIssue(
                    field="match.sets",
                    detail="Match is marked finished but set scores are tied, which is not possible",
                    severity=IssueSeverity.CRITICAL,
                )
            )
    return issues


def _check_odds(match: Match, odds: list[Odds], as_of: datetime, settings: AnalyticsSettings) -> tuple[
    list[DataQualityIssue], bool, bool
]:
    issues: list[DataQualityIssue] = []
    if not odds:
        issues.append(
            DataQualityIssue(
                field="odds",
                detail="No bookmaker odds were provided for this match",
                severity=IssueSeverity.WARNING,
            )
        )
        return issues, False, False

    max_age = timedelta(minutes=settings.odds_max_age_minutes)
    freshest = max(o.captured_at for o in odds)
    is_fresh = (as_of - freshest) <= max_age if freshest <= as_of else True

    for o in odds:
        if o.captured_at > as_of:
            issues.append(
                DataQualityIssue(
                    field="odds.captured_at",
                    detail=f"Odds from {o.bookmaker!r} are timestamped in the future relative to analysis time",
                    severity=IssueSeverity.WARNING,
                )
            )
    if not is_fresh:
        issues.append(
            DataQualityIssue(
                field="odds.captured_at",
                detail=f"Freshest odds are older than {settings.odds_max_age_minutes:.0f} minutes",
                severity=IssueSeverity.INFO,
            )
        )
    return issues, True, is_fresh


def _check_history(
    field_name: str,
    history: list[HistoricalMatch],
    min_sample: int,
) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    if not history:
        issues.append(
            DataQualityIssue(
                field=field_name,
                detail="No historical matches were provided for this player",
                severity=IssueSeverity.WARNING,
            )
        )
        return issues

    if len(history) < min_sample:
        issues.append(
            DataQualityIssue(
                field=field_name,
                detail=f"Only {len(history)} historical match(es) provided (below the {min_sample}-match minimum)",
                severity=IssueSeverity.INFO,
            )
        )

    seen: set[tuple[str, str]] = set()
    for entry in history:
        key = (entry.opponent_id, entry.played_at.isoformat())
        if key in seen:
            issues.append(
                DataQualityIssue(
                    field=field_name,
                    detail=f"Duplicate historical match observation against {entry.opponent_id!r}",
                    severity=IssueSeverity.WARNING,
                )
            )
        seen.add(key)
    return issues


def _check_head_to_head(
    head_to_head: HeadToHeadRecord | None,
    player_one_history: list[HistoricalMatch],
    match: Match,
) -> list[DataQualityIssue]:
    if head_to_head is None:
        return []
    observed = sum(1 for h in player_one_history if h.opponent_id == match.player_two.id)
    if observed > head_to_head.total_matches:
        return [
            DataQualityIssue(
                field="head_to_head",
                detail="Head-to-head record has fewer matches than observed directly in player history",
                severity=IssueSeverity.WARNING,
            )
        ]
    return []


def assess_data_quality(
    match: Match,
    odds: list[Odds],
    player_one_history: list[HistoricalMatch],
    player_two_history: list[HistoricalMatch],
    head_to_head: HeadToHeadRecord | None,
    as_of: datetime,
    settings: AnalyticsSettings | None = None,
) -> DataQualityAssessment:
    """Assess the completeness and reliability of the data behind an analysis."""
    settings = settings or get_analytics_settings()

    issues: list[DataQualityIssue] = []
    issues.extend(_check_match_completeness(match))

    odds_issues, odds_available, odds_fresh = _check_odds(match, odds, as_of, settings)
    issues.extend(odds_issues)

    issues.extend(_check_history("player_one_history", player_one_history, settings.min_history_sample_size))
    issues.extend(_check_history("player_two_history", player_two_history, settings.min_history_sample_size))
    issues.extend(_check_head_to_head(head_to_head, player_one_history, match))

    penalty = sum(_SEVERITY_PENALTY[issue.severity] for issue in issues)
    score = _clamp(100.0 - penalty, 0.0, 100.0)

    return DataQualityAssessment(
        score=score,
        warnings=issues,
        history_sample_size_player_one=len(player_one_history),
        history_sample_size_player_two=len(player_two_history),
        odds_available=odds_available,
        odds_fresh=odds_fresh,
    )
