"""Low-level, pure rolling-window primitives shared by `features.player` and `features.matchup`.

Everything here operates on a `PlayerMatchPerspective` — a match viewed
from exactly one participant's side — so both single-player rolling
stats and head-to-head stats can reuse the same math.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from persistence.models import COMPLETED_STATUSES, HistoricalMatchRecord


@dataclass(frozen=True)
class PlayerMatchPerspective:
    """One historical match, viewed from one specific player's perspective."""

    match_id: str
    opponent_id: str
    effective_timestamp: datetime
    won: bool | None
    sets_for: int
    sets_against: int
    points_for: int
    points_against: int
    first_set_won: bool | None
    is_deciding_set_match: bool
    deciding_set_won: bool | None
    straight_sets_win: bool
    duration_minutes: float | None
    incomplete: bool
    data_quality_score: float
    competition_id: str | None
    best_of: int | None


def build_perspective(match: HistoricalMatchRecord, player_id: str) -> PlayerMatchPerspective:
    """Build a `PlayerMatchPerspective` for `player_id` from a `HistoricalMatchRecord`."""
    if player_id == match.player_a_id:
        is_a = True
        opponent_id = match.player_b_id
    elif player_id == match.player_b_id:
        is_a = False
        opponent_id = match.player_a_id
    else:
        raise ValueError(f"player {player_id!r} did not participate in match {match.id!r}")

    sets_for = match.sets_a_won if is_a else match.sets_b_won
    sets_against = match.sets_b_won if is_a else match.sets_a_won
    points_for = sum((s.player_a_points if is_a else s.player_b_points) for s in match.sets)
    points_against = sum((s.player_b_points if is_a else s.player_a_points) for s in match.sets)
    won = None if match.winner_id is None else (match.winner_id == player_id)

    first_set_won: bool | None = None
    if match.sets:
        first = min(match.sets, key=lambda s: s.set_number)
        first_set_won = (first.winner == "player_a") if is_a else (first.winner == "player_b")

    is_deciding_set_match = False
    deciding_set_won: bool | None = None
    if len(match.sets) >= 2:
        ordered = sorted(match.sets, key=lambda s: s.set_number)
        prior = ordered[:-1]
        prior_for = sum(1 for s in prior if (s.winner == "player_a") == is_a)
        prior_against = len(prior) - prior_for
        if prior_for == prior_against:
            is_deciding_set_match = True
            last = ordered[-1]
            deciding_set_won = (last.winner == "player_a") == is_a

    straight_sets_win = bool(won) and sets_against == 0 and sets_for > 0

    duration_minutes = None
    if match.actual_start_at is not None and match.completed_at is not None:
        duration_minutes = (match.completed_at - match.actual_start_at).total_seconds() / 60.0

    return PlayerMatchPerspective(
        match_id=match.id,
        opponent_id=opponent_id,
        effective_timestamp=match.effective_timestamp,
        won=won,
        sets_for=sets_for,
        sets_against=sets_against,
        points_for=points_for,
        points_against=points_against,
        first_set_won=first_set_won,
        is_deciding_set_match=is_deciding_set_match,
        deciding_set_won=deciding_set_won,
        straight_sets_win=straight_sets_win,
        duration_minutes=duration_minutes,
        incomplete=match.status not in COMPLETED_STATUSES,
        data_quality_score=match.data_quality.completeness_score,
        competition_id=match.competition_id,
        best_of=match.best_of,
    )


def sort_most_recent_first(perspectives: list[PlayerMatchPerspective]) -> list[PlayerMatchPerspective]:
    """Deterministic ordering: newest first, ties broken by match ID for stability."""
    return sorted(perspectives, key=lambda p: (p.effective_timestamp, p.match_id), reverse=True)


def safe_rate(numerator: int, denominator: int) -> float | None:
    """A rate that is `None` (not 0.0) when the denominator is zero."""
    if denominator <= 0:
        return None
    return numerator / denominator


def recency_weight(effective_timestamp: datetime, as_of: datetime, half_life_days: float) -> float:
    """Exponential recency decay: 1.0 at `as_of`, halving every `half_life_days`."""
    days_ago = max(0.0, (as_of - effective_timestamp).total_seconds() / 86400.0)
    return 0.5 ** (days_ago / half_life_days)


def result_streak(perspectives_most_recent_first: list[PlayerMatchPerspective]) -> int:
    """Positive = current win streak length, negative = current loss streak length, 0 = no data."""
    sign: int | None = None
    streak = 0
    for p in perspectives_most_recent_first:
        if p.won is None:
            break
        current_sign = 1 if p.won else -1
        if sign is None:
            sign = current_sign
            streak = 1
        elif current_sign == sign:
            streak += 1
        else:
            break
    return streak * sign if sign is not None else 0


def _match_performance_score(p: PlayerMatchPerspective) -> float:
    if p.won is None:
        return 50.0
    outcome = 100.0 if p.won else 0.0
    total_points = p.points_for + p.points_against
    if total_points == 0:
        return outcome
    margin_ratio = (p.points_for - p.points_against) / total_points
    return max(0.0, min(100.0, outcome + margin_ratio * 15.0))


def recency_weighted_win_rate(
    perspectives_most_recent_first: list[PlayerMatchPerspective], as_of: datetime, half_life_days: float
) -> float | None:
    completed = [p for p in perspectives_most_recent_first if p.won is not None]
    if not completed:
        return None
    total_weight = 0.0
    weighted_sum = 0.0
    for p in completed:
        w = recency_weight(p.effective_timestamp, as_of, half_life_days)
        weighted_sum += (1.0 if p.won else 0.0) * w
        total_weight += w
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def volatility_score(perspectives: list[PlayerMatchPerspective]) -> float | None:
    """Normalized (0-1) standard deviation of per-match performance score. None if <2 completed matches."""
    completed = [p for p in perspectives if p.won is not None]
    if len(completed) < 2:
        return None
    scores = [_match_performance_score(p) for p in completed]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    stdev = variance**0.5
    return max(0.0, min(1.0, stdev / 50.0))
