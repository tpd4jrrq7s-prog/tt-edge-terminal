"""Match momentum analysis (0-100 per player).

Distinguishes three states, in priority order:

- IN_PLAY: point-by-point data for the current match is available.
- PRE_MATCH: no live point data, but historical form trend is available.
- NO_DATA: neither is available — both scores are the neutral 50.0.

Short streaks are deliberately not treated as high-confidence signals:
confidence is capped by sample size and dampened by lead-change volatility.
"""

from __future__ import annotations

from datetime import datetime

from config.analytics import AnalyticsSettings, get_analytics_settings
from engine.form import match_performance_score, recency_weight
from engine.models import HistoricalMatch, MomentumComponent, MomentumResult, MomentumState, PointEvent

NEUTRAL_SCORE = 50.0

_IN_PLAY_WEIGHTS = {
    "consecutive_points": 0.25,
    "consecutive_sets": 0.20,
    "recovery": 0.15,
    "pressure_points": 0.20,
    "recent_differential": 0.20,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _reconstruct_set_winners(events: list[PointEvent]) -> list[str]:
    """Derive each set's winner from the last point event recorded in that set."""
    last_event_by_set: dict[int, PointEvent] = {}
    order: list[int] = []
    for event in events:
        if event.set_number not in last_event_by_set:
            order.append(event.set_number)
        last_event_by_set[event.set_number] = event

    winners = []
    for set_number in order:
        final = last_event_by_set[set_number]
        if final.player_one_score > final.player_two_score:
            winners.append("player_one")
        elif final.player_two_score > final.player_one_score:
            winners.append("player_two")
    return winners


def _trailing_streak(sequence: list[str]) -> tuple[str, int]:
    """Return (owner, length) of the trailing run of identical values, or ("", 0) if empty."""
    if not sequence:
        return "", 0
    owner = sequence[-1]
    length = 0
    for value in reversed(sequence):
        if value != owner:
            break
        length += 1
    return owner, length


def _consecutive_points_component(events: list[PointEvent]) -> tuple[float, float, str]:
    winners = [e.winner for e in events]
    owner, streak = _trailing_streak(winners)
    if not owner:
        return NEUTRAL_SCORE, NEUTRAL_SCORE, "no points played yet"
    owner_score = _clamp(50.0 + min(streak, 10) * 5.0, 0.0, 100.0)
    other_score = 100.0 - owner_score
    detail = f"{owner} has won {streak} consecutive point(s)"
    return (owner_score, other_score, detail) if owner == "player_one" else (other_score, owner_score, detail)


def _consecutive_sets_component(events: list[PointEvent]) -> tuple[float, float, str]:
    set_winners = _reconstruct_set_winners(events)
    owner, streak = _trailing_streak(set_winners)
    if not owner:
        return NEUTRAL_SCORE, NEUTRAL_SCORE, "no completed sets yet"
    owner_score = _clamp(50.0 + streak * 15.0, 0.0, 100.0)
    other_score = 100.0 - owner_score
    detail = f"{owner} has won {streak} consecutive set(s)"
    return (owner_score, other_score, detail) if owner == "player_one" else (other_score, owner_score, detail)


def _recovery_component(events: list[PointEvent]) -> tuple[float, float, str]:
    set_winners = _reconstruct_set_winners(events)
    recoveries = {"player_one": 0, "player_two": 0}
    losses = {"player_one": 0, "player_two": 0}
    for i in range(len(set_winners) - 1):
        loser = "player_two" if set_winners[i] == "player_one" else "player_one"
        losses[loser] += 1
        if set_winners[i + 1] == loser:
            recoveries[loser] += 1

    def _score(player: str) -> float:
        if losses[player] == 0:
            return NEUTRAL_SCORE
        return _clamp(100.0 * recoveries[player] / losses[player], 0.0, 100.0)

    detail = f"player_one recovered {recoveries['player_one']}/{losses['player_one']} lost set(s); " \
        f"player_two recovered {recoveries['player_two']}/{losses['player_two']} lost set(s)"
    return _score("player_one"), _score("player_two"), detail


def _pressure_points_component(events: list[PointEvent], threshold: int) -> tuple[float, float, str]:
    pressure_events = [e for e in events if e.player_one_score >= threshold and e.player_two_score >= threshold]
    if not pressure_events:
        return NEUTRAL_SCORE, NEUTRAL_SCORE, "no pressure-point situations reached yet"
    one_wins = sum(1 for e in pressure_events if e.winner == "player_one")
    total = len(pressure_events)
    one_score = _clamp(100.0 * one_wins / total, 0.0, 100.0)
    detail = f"player_one won {one_wins}/{total} pressure point(s) (>= {threshold}-{threshold})"
    return one_score, 100.0 - one_score, detail


def _recent_differential_component(events: list[PointEvent], window: int) -> tuple[float, float, str]:
    recent = events[-window:] if window > 0 else []
    if not recent:
        return NEUTRAL_SCORE, NEUTRAL_SCORE, "no recent points to evaluate"
    one_wins = sum(1 for e in recent if e.winner == "player_one")
    two_wins = len(recent) - one_wins
    net = one_wins - two_wins
    one_score = _clamp(50.0 + net * 10.0, 0.0, 100.0)
    detail = f"player_one won {one_wins}/{len(recent)} of the last {len(recent)} point(s)"
    return one_score, 100.0 - one_score, detail


def _lead_changes(events: list[PointEvent]) -> int:
    changes = 0
    previous_leader: str | None = None
    for event in events:
        if event.player_one_score > event.player_two_score:
            leader = "player_one"
        elif event.player_two_score > event.player_one_score:
            leader = "player_two"
        else:
            leader = previous_leader
        if previous_leader is not None and leader is not None and leader != previous_leader:
            changes += 1
        previous_leader = leader
    return changes


def calculate_in_play_momentum(
    point_progression: list[PointEvent],
    settings: AnalyticsSettings,
) -> MomentumResult:
    """Compute momentum from point-by-point progression of the current match."""
    components: list[MomentumComponent] = []
    weighted_one = 0.0

    cp_one, cp_two, cp_detail = _consecutive_points_component(point_progression)
    cs_one, cs_two, cs_detail = _consecutive_sets_component(point_progression)
    rec_one, rec_two, rec_detail = _recovery_component(point_progression)
    pp_one, pp_two, pp_detail = _pressure_points_component(
        point_progression, settings.pressure_point_score_threshold
    )
    rd_one, rd_two, rd_detail = _recent_differential_component(
        point_progression, settings.momentum_recent_window * 2
    )

    parts = [
        ("consecutive_points", cp_one, cp_two, cp_detail),
        ("consecutive_sets", cs_one, cs_two, cs_detail),
        ("recovery", rec_one, rec_two, rec_detail),
        ("pressure_points", pp_one, pp_two, pp_detail),
        ("recent_differential", rd_one, rd_two, rd_detail),
    ]
    for name, one_value, _two_value, detail in parts:
        weight = _IN_PLAY_WEIGHTS[name]
        weighted_one += one_value * weight
        components.append(MomentumComponent(name=name, value=one_value, detail=detail))

    player_one_score = _clamp(weighted_one, 0.0, 100.0)
    player_two_score = 100.0 - player_one_score

    lead_changes = _lead_changes(point_progression)
    sample_ratio = _clamp(len(point_progression) / settings.momentum_min_points, 0.0, 1.0)
    volatility_penalty = _clamp(lead_changes / 10.0, 0.0, 0.5)
    confidence = _clamp(sample_ratio * (1.0 - volatility_penalty), 0.0, 1.0)

    components.append(
        MomentumComponent(
            name="lead_changes",
            value=float(lead_changes),
            detail=f"{lead_changes} lead change(s) observed so far",
        )
    )

    return MomentumResult(
        player_one_score=player_one_score,
        player_two_score=player_two_score,
        state=MomentumState.IN_PLAY,
        confidence=confidence,
        components=components,
    )


def _pre_match_player_momentum(
    history: list[HistoricalMatch],
    as_of: datetime,
    settings: AnalyticsSettings,
) -> tuple[float, float]:
    """Return (score, confidence) for one player's pre-match momentum trend."""
    if not history:
        return NEUTRAL_SCORE, 0.0

    recent = sorted(history, key=lambda m: m.played_at, reverse=True)[: settings.momentum_recent_window]

    total_weight = 0.0
    weighted_sum = 0.0
    for match in recent:
        weight = recency_weight(match.played_at, as_of, settings.form_recency_half_life_days)
        weighted_sum += match_performance_score(match) * weight
        total_weight += weight
    trend_score = weighted_sum / total_weight if total_weight > 0 else NEUTRAL_SCORE

    ordered_recent = sorted(history, key=lambda m: m.played_at, reverse=True)
    outcomes = ["win" if m.won else "loss" for m in ordered_recent]
    owner, streak = _trailing_streak(outcomes)
    streak_bonus = streak * 5.0 if owner == "win" else -streak * 5.0
    score = _clamp(trend_score + _clamp(streak_bonus, -20.0, 20.0), 0.0, 100.0)

    confidence = _clamp(len(recent) / max(settings.min_history_sample_size, 1), 0.0, 1.0)
    return score, confidence


def calculate_pre_match_momentum(
    player_one_history: list[HistoricalMatch],
    player_two_history: list[HistoricalMatch],
    as_of: datetime,
    settings: AnalyticsSettings,
) -> MomentumResult:
    """Compute momentum from each player's own recent-match trend (no live data)."""
    one_score, one_confidence = _pre_match_player_momentum(player_one_history, as_of, settings)
    two_score, two_confidence = _pre_match_player_momentum(player_two_history, as_of, settings)

    components = [
        MomentumComponent(
            name="recent_trend",
            value=one_score,
            detail=f"player_one recent-match trend score is {one_score:.1f}/100",
        ),
        MomentumComponent(
            name="recent_trend",
            value=two_score,
            detail=f"player_two recent-match trend score is {two_score:.1f}/100",
        ),
    ]

    confidence = min(one_confidence, two_confidence) if (player_one_history or player_two_history) else 0.0

    return MomentumResult(
        player_one_score=one_score,
        player_two_score=two_score,
        state=MomentumState.PRE_MATCH,
        confidence=confidence,
        components=components,
    )


def calculate_momentum(
    point_progression: list[PointEvent],
    player_one_history: list[HistoricalMatch],
    player_two_history: list[HistoricalMatch],
    as_of: datetime,
    settings: AnalyticsSettings | None = None,
) -> MomentumResult:
    """Compute match momentum, preferring live point data over historical trend."""
    settings = settings or get_analytics_settings()

    if point_progression:
        return calculate_in_play_momentum(point_progression, settings)

    if player_one_history or player_two_history:
        return calculate_pre_match_momentum(player_one_history, player_two_history, as_of, settings)

    return MomentumResult(
        player_one_score=NEUTRAL_SCORE,
        player_two_score=NEUTRAL_SCORE,
        state=MomentumState.NO_DATA,
        confidence=0.0,
        components=[],
    )
