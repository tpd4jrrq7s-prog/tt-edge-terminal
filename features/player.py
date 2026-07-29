"""Build transparent, rolling `PlayerRollingFeatures` from a player's prior matches.

Callers are responsible for supplying only records that are already
known to be strictly before the relevant cutoff (see
`persistence.protocols.MatchRepository.list_player_matches_before`) —
this module does no cutoff filtering of its own, by design, so a single
obvious call site owns leakage safety.
"""

from __future__ import annotations

from datetime import datetime

from config.historical import HistoricalIntelligenceSettings, get_historical_intelligence_settings
from features.models import PlayerRollingFeatures, WindowFeatures
from features.rolling import (
    PlayerMatchPerspective,
    build_perspective,
    recency_weighted_win_rate,
    result_streak,
    safe_rate,
    sort_most_recent_first,
    volatility_score,
)
from persistence.models import HistoricalMatchRecord


def _build_window(label: str, window: list[PlayerMatchPerspective]) -> WindowFeatures:
    n = len(window)
    completed = [p for p in window if p.won is not None]
    wins = sum(1 for p in completed if p.won)
    losses = len(completed) - wins

    sets_won = sum(p.sets_for for p in window)
    sets_lost = sum(p.sets_against for p in window)
    points_won = sum(p.points_for for p in window)
    points_lost = sum(p.points_against for p in window)

    set_margins = [p.sets_for - p.sets_against for p in window if (p.sets_for + p.sets_against) > 0]
    point_margins = [p.points_for - p.points_against for p in window if (p.points_for + p.points_against) > 0]

    winning_matches = [p for p in completed if p.won]
    straight_wins = sum(1 for p in winning_matches if p.straight_sets_win)

    deciders = [p for p in window if p.is_deciding_set_match]
    decider_results = [p for p in deciders if p.deciding_set_won is not None]

    first_set_known = [p for p in window if p.first_set_won is not None]
    lost_first_with_result = [p for p in first_set_known if not p.first_set_won and p.won is not None]
    won_first_with_result = [p for p in first_set_known if p.first_set_won and p.won is not None]

    durations = [p.duration_minutes for p in window if p.duration_minutes is not None]

    return WindowFeatures(
        window_label=label,
        matches_played=n,
        wins=wins,
        losses=losses,
        win_rate=safe_rate(wins, len(completed)),
        sets_won=sets_won,
        sets_lost=sets_lost,
        set_win_rate=safe_rate(sets_won, sets_won + sets_lost),
        points_won=points_won,
        points_lost=points_lost,
        point_win_rate=safe_rate(points_won, points_won + points_lost),
        average_set_margin=(sum(set_margins) / len(set_margins)) if set_margins else None,
        average_point_margin=(sum(point_margins) / len(point_margins)) if point_margins else None,
        straight_sets_win_rate=safe_rate(straight_wins, len(winning_matches)) if winning_matches else None,
        straight_sets_win_rate_n=len(winning_matches),
        deciding_set_appearance_rate=safe_rate(len(deciders), n) if n > 0 else None,
        deciding_set_win_rate=(
            safe_rate(sum(1 for p in decider_results if p.deciding_set_won), len(decider_results))
            if decider_results
            else None
        ),
        deciding_set_win_rate_n=len(decider_results),
        first_set_win_rate=(
            safe_rate(sum(1 for p in first_set_known if p.first_set_won), len(first_set_known))
            if first_set_known
            else None
        ),
        first_set_win_rate_n=len(first_set_known),
        comeback_win_rate=(
            safe_rate(sum(1 for p in lost_first_with_result if p.won), len(lost_first_with_result))
            if lost_first_with_result
            else None
        ),
        comeback_win_rate_n=len(lost_first_with_result),
        loss_rate_after_winning_first_set=(
            safe_rate(sum(1 for p in won_first_with_result if not p.won), len(won_first_with_result))
            if won_first_with_result
            else None
        ),
        loss_rate_after_winning_first_set_n=len(won_first_with_result),
        average_match_duration_minutes=(sum(durations) / len(durations)) if durations else None,
        average_match_duration_n=len(durations),
        incomplete_match_count=sum(1 for p in window if p.incomplete),
    )


def build_player_rolling_features(
    player_id: str,
    prior_matches: list[HistoricalMatchRecord],
    as_of: datetime,
    settings: HistoricalIntelligenceSettings | None = None,
    latest_ranking: int | None = None,
) -> PlayerRollingFeatures:
    """Build rolling features for `player_id` from matches already filtered to before `as_of`.

    `latest_ranking`, if supplied, must already be the player's most
    recent ranking strictly before `as_of` (e.g. from
    `RankingRepository.latest_before`) — never a "current" ranking
    backfilled into a historical snapshot.
    """
    settings = settings or get_historical_intelligence_settings()

    perspectives = sort_most_recent_first([build_perspective(m, player_id) for m in prior_matches])

    windows = [_build_window("all_time", perspectives)]
    for size in settings.rolling_window_sizes:
        windows.append(_build_window(f"last_{size}", perspectives[:size]))

    rest_hours = None
    if perspectives:
        rest_hours = (as_of - perspectives[0].effective_timestamp).total_seconds() / 3600.0

    matches_24h = sum(1 for p in perspectives if (as_of - p.effective_timestamp).total_seconds() <= 86400)
    matches_7d = sum(1 for p in perspectives if (as_of - p.effective_timestamp).total_seconds() <= 7 * 86400)

    quality_scores = [p.data_quality_score for p in perspectives]

    return PlayerRollingFeatures(
        player_id=player_id,
        windows=windows,
        rest_hours_since_previous_match=rest_hours,
        matches_in_previous_24h=matches_24h,
        matches_in_previous_7d=matches_7d,
        result_streak=result_streak(perspectives),
        recency_weighted_win_rate=recency_weighted_win_rate(
            perspectives, as_of, settings.form_recency_half_life_days
        ),
        opponent_adjusted_win_rate=None,
        ranking=latest_ranking,
        volatility_score=volatility_score(perspectives),
        historical_data_quality_average=(sum(quality_scores) / len(quality_scores)) if quality_scores else None,
        observation_count=len(perspectives),
    )
