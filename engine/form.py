"""Transparent recent-form scoring (0-100) for a single player.

Every historical match contributes a `match_performance_score` (result
+ margin quality) weighted by three independent, testable factors:
recency, opponent strength, and completeness (walkover/retirement).
With no history, the score is the neutral midpoint (50.0) with zero
confidence — never a fabricated observation.
"""

from __future__ import annotations

from datetime import datetime

from config.analytics import AnalyticsSettings, get_analytics_settings
from engine.models import FormResult, HistoricalMatch

NEUTRAL_SCORE = 50.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def match_performance_score(match: HistoricalMatch) -> float:
    """Score a single historical match from 0-100, blending result and set/point margin.

    A win is worth more than a loss, but margin quality nudges the score
    within a small band so a dominant win outscores a narrow one and a
    narrow loss outscores a blowout loss.
    """
    if match.walkover:
        return 100.0 if match.won else 0.0

    outcome = 100.0 if match.won else 0.0
    if not match.sets:
        return outcome

    total_points = sum(s.total_points for s in match.sets) or 1
    margin_ratio = match.point_margin / total_points  # roughly in [-1, 1]
    margin_bonus = margin_ratio * 15.0
    return _clamp(outcome + margin_bonus, 0.0, 100.0)


def recency_weight(played_at: datetime, as_of: datetime, half_life_days: float) -> float:
    """Exponential recency decay: 1.0 for `as_of`, halving every `half_life_days`."""
    days_ago = max(0.0, (as_of - played_at).total_seconds() / 86400.0)
    return 0.5 ** (days_ago / half_life_days)


def opponent_strength_weight(opponent_ranking: int | None) -> float:
    """Weight multiplier rewarding wins/losses against stronger (lower-numbered) opponents.

    Unranked opponents are neutral (1.0x) rather than assumed weak or strong.
    """
    if opponent_ranking is None:
        return 1.0
    normalized = _clamp((200 - opponent_ranking) / 200.0, 0.0, 1.0)
    return 1.0 + 0.5 * normalized


def completion_weight(match: HistoricalMatch) -> float:
    """Down-weight matches with incomplete/unreliable data (walkovers, retirements)."""
    if match.walkover:
        return 0.3
    if match.retired:
        return 0.5
    return 1.0


def _sample_confidence(n: int, min_sample: int) -> float:
    if n <= 0:
        return 0.0
    if min_sample <= 0:
        return 1.0
    return _clamp(n / min_sample, 0.0, 1.0)


def calculate_form(
    history: list[HistoricalMatch],
    as_of: datetime,
    settings: AnalyticsSettings | None = None,
) -> FormResult:
    """Compute a player's recent-form score and confidence from their match history."""
    settings = settings or get_analytics_settings()

    if not history:
        return FormResult(
            score=NEUTRAL_SCORE,
            confidence=0.0,
            matches_considered=0,
            average_recency_weight=None,
        )

    total_weight = 0.0
    weighted_score_sum = 0.0
    recency_weight_sum = 0.0

    for match in history:
        rw = recency_weight(match.played_at, as_of, settings.form_recency_half_life_days)
        ow = opponent_strength_weight(match.opponent_ranking)
        cw = completion_weight(match)
        weight = rw * ow * cw

        weighted_score_sum += match_performance_score(match) * weight
        total_weight += weight
        recency_weight_sum += rw

    if total_weight <= 0:
        score = NEUTRAL_SCORE
    else:
        score = _clamp(weighted_score_sum / total_weight, 0.0, 100.0)

    return FormResult(
        score=score,
        confidence=_sample_confidence(len(history), settings.min_history_sample_size),
        matches_considered=len(history),
        average_recency_weight=recency_weight_sum / len(history),
    )
