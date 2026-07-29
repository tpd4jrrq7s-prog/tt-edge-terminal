"""Transparent behavioral pattern detection from historical match data.

Every detector reports its own sample size and a confidence derived from
that sample size — a pattern is never presented as strong evidence from
a handful of observations. Patterns whose denominator is zero (e.g. "how
often do they come back after losing the first set" when they've never
lost a first set) are omitted rather than fabricated as 0/0.
"""

from __future__ import annotations

import statistics
from typing import Literal

from config.analytics import AnalyticsSettings, get_analytics_settings
from engine.form import match_performance_score
from engine.models import HistoricalMatch, PatternSignal, PointEvent


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sample_confidence(n: int, min_sample: int) -> float:
    if n <= 0:
        return 0.0
    if min_sample <= 0:
        return 1.0
    return _clamp(n / min_sample, 0.0, 1.0)


def _matches_with_sets(history: list[HistoricalMatch]) -> list[HistoricalMatch]:
    return [m for m in history if m.sets]


def _signal(
    player_id: str,
    pattern: str,
    numerator: int,
    denominator: int,
    min_sample: int,
    observations: list[str],
) -> PatternSignal | None:
    if denominator <= 0:
        return None
    strength = numerator / denominator
    return PatternSignal(
        player_id=player_id,
        pattern=pattern,
        strength=_clamp(strength, 0.0, 1.0),
        confidence=_sample_confidence(denominator, min_sample),
        sample_size=denominator,
        supporting_observations=observations,
    )


def _slow_starter(player_id: str, with_sets: list[HistoricalMatch], min_sample: int) -> PatternSignal | None:
    lost_first = sum(1 for m in with_sets if not m.sets[0].won)
    return _signal(
        player_id, "slow_starter", lost_first, len(with_sets), min_sample,
        [f"Lost the first set in {lost_first}/{len(with_sets)} matches with set data"],
    )


def _strong_first_set(player_id: str, with_sets: list[HistoricalMatch], min_sample: int) -> PatternSignal | None:
    won_first = sum(1 for m in with_sets if m.sets[0].won)
    return _signal(
        player_id, "strong_first_set", won_first, len(with_sets), min_sample,
        [f"Won the first set in {won_first}/{len(with_sets)} matches with set data"],
    )


def _frequent_comeback(player_id: str, with_sets: list[HistoricalMatch], min_sample: int) -> PatternSignal | None:
    lost_first_matches = [m for m in with_sets if not m.sets[0].won]
    comebacks = sum(1 for m in lost_first_matches if m.won)
    return _signal(
        player_id, "frequent_comeback", comebacks, len(lost_first_matches), min_sample,
        [f"Won {comebacks}/{len(lost_first_matches)} matches after losing the first set"],
    )


def _weak_after_winning_first_set(
    player_id: str, with_sets: list[HistoricalMatch], min_sample: int
) -> PatternSignal | None:
    won_first_matches = [m for m in with_sets if m.sets[0].won]
    upsets = sum(1 for m in won_first_matches if not m.won)
    return _signal(
        player_id, "weak_after_winning_first_set", upsets, len(won_first_matches), min_sample,
        [f"Lost {upsets}/{len(won_first_matches)} matches after winning the first set"],
    )


def _high_deciding_set_performance(
    player_id: str, with_sets: list[HistoricalMatch], min_sample: int
) -> PatternSignal | None:
    deciders = []
    for m in with_sets:
        if len(m.sets) < 2:
            continue
        prior = m.sets[:-1]
        prior_won = sum(1 for s in prior if s.won)
        if prior_won == len(prior) - prior_won:
            deciders.append(m)
    won_deciders = sum(1 for m in deciders if m.sets[-1].won)
    return _signal(
        player_id, "high_deciding_set_performance", won_deciders, len(deciders), min_sample,
        [f"Won {won_deciders}/{len(deciders)} deciding sets (match tied before the final set)"],
    )


def _high_volatility(player_id: str, history: list[HistoricalMatch], min_sample: int) -> PatternSignal | None:
    if len(history) < 2:
        return None
    scores = [match_performance_score(m) for m in history]
    stdev = statistics.pstdev(scores)
    strength = _clamp(stdev / 50.0, 0.0, 1.0)
    return PatternSignal(
        player_id=player_id,
        pattern="high_volatility",
        strength=strength,
        confidence=_sample_confidence(len(history), min_sample),
        sample_size=len(history),
        supporting_observations=[
            f"Match performance score standard deviation is {stdev:.1f} across {len(history)} matches"
        ],
    )


def _straight_sets_tendency(
    player_id: str, with_sets: list[HistoricalMatch], min_sample: int
) -> PatternSignal | None:
    won_matches = [m for m in with_sets if m.won]
    straight = sum(1 for m in won_matches if m.sets_lost == 0)
    return _signal(
        player_id, "straight_sets_tendency", straight, len(won_matches), min_sample,
        [f"Won {straight}/{len(won_matches)} matches without dropping a set"],
    )


def _late_set_collapse(player_id: str, with_sets: list[HistoricalMatch], min_sample: int) -> PatternSignal | None:
    lost_sets = [s for m in with_sets for s in m.sets if not s.won]
    close_losses = sum(1 for s in lost_sets if abs(s.margin) <= 2)
    return _signal(
        player_id, "late_set_collapse", close_losses, len(lost_sets), min_sample,
        [f"{close_losses}/{len(lost_sets)} lost sets were lost by 2 points or fewer"],
    )


def _strong_pressure_point_performance(
    player_id: str,
    side: Literal["player_one", "player_two"],
    point_progression: list[PointEvent],
    threshold: int,
    min_sample: int,
) -> PatternSignal | None:
    pressure_events = [
        e for e in point_progression if e.player_one_score >= threshold and e.player_two_score >= threshold
    ]
    wins = sum(1 for e in pressure_events if e.winner == side)
    return _signal(
        player_id, "strong_pressure_point_performance", wins, len(pressure_events), min_sample,
        [f"Won {wins}/{len(pressure_events)} pressure points (>= {threshold}-{threshold}) in the current match"],
    )


def detect_patterns(
    player_id: str,
    side: Literal["player_one", "player_two"],
    history: list[HistoricalMatch],
    point_progression: list[PointEvent],
    settings: AnalyticsSettings | None = None,
) -> list[PatternSignal]:
    """Detect all transparent behavioral patterns supported by the available data."""
    settings = settings or get_analytics_settings()
    min_sample = settings.min_history_sample_size
    with_sets = _matches_with_sets(history)

    candidates = [
        _slow_starter(player_id, with_sets, min_sample),
        _strong_first_set(player_id, with_sets, min_sample),
        _frequent_comeback(player_id, with_sets, min_sample),
        _weak_after_winning_first_set(player_id, with_sets, min_sample),
        _high_deciding_set_performance(player_id, with_sets, min_sample),
        _high_volatility(player_id, history, min_sample),
        _straight_sets_tendency(player_id, with_sets, min_sample),
        _late_set_collapse(player_id, with_sets, min_sample),
        _strong_pressure_point_performance(
            player_id, side, point_progression, settings.pressure_point_score_threshold, min_sample
        ),
    ]
    return [signal for signal in candidates if signal is not None]
