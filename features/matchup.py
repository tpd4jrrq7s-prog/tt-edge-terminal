"""Build leakage-safe `MatchupFeatures` (head-to-head) between two players.

As with `features.player`, callers must supply only records already
known to be strictly before the relevant cutoff. An empty history is
represented as all-`None`/zero fields — never a fabricated 50/50 split.
"""

from __future__ import annotations

from datetime import datetime

from config.historical import HistoricalIntelligenceSettings, get_historical_intelligence_settings
from features.models import MatchupFeatures, PlayerRollingFeatures
from features.rolling import build_perspective, safe_rate, sort_most_recent_first
from persistence.models import HistoricalMatchRecord


def build_matchup_features(
    player_a_id: str,
    player_b_id: str,
    head_to_head_matches: list[HistoricalMatchRecord],
    player_a_features: PlayerRollingFeatures,
    player_b_features: PlayerRollingFeatures,
    as_of: datetime,
    target_competition_id: str | None = None,
    target_best_of: int | None = None,
    settings: HistoricalIntelligenceSettings | None = None,
) -> MatchupFeatures:
    """Build matchup features from player A's perspective; player B's rate is the complement."""
    settings = settings or get_historical_intelligence_settings()

    perspectives = sort_most_recent_first(
        [build_perspective(m, player_a_id) for m in head_to_head_matches]
    )

    form_differential = None
    a_all_time = player_a_features.window("all_time")
    b_all_time = player_b_features.window("all_time")
    if a_all_time and b_all_time and a_all_time.win_rate is not None and b_all_time.win_rate is not None:
        form_differential = a_all_time.win_rate - b_all_time.win_rate

    rest_differential = None
    if (
        player_a_features.rest_hours_since_previous_match is not None
        and player_b_features.rest_hours_since_previous_match is not None
    ):
        rest_differential = (
            player_a_features.rest_hours_since_previous_match - player_b_features.rest_hours_since_previous_match
        )

    workload_differential = float(
        player_a_features.matches_in_previous_7d - player_b_features.matches_in_previous_7d
    )

    volatility_differential = None
    if player_a_features.volatility_score is not None and player_b_features.volatility_score is not None:
        volatility_differential = player_a_features.volatility_score - player_b_features.volatility_score

    ranking_differential = None
    if player_a_features.ranking is not None and player_b_features.ranking is not None:
        # Positive favors player_a (a lower ranking number is better).
        ranking_differential = float(player_b_features.ranking - player_a_features.ranking)

    if not perspectives:
        return MatchupFeatures(
            player_a_id=player_a_id,
            player_b_id=player_b_id,
            head_to_head_matches=0,
            rest_differential_hours=rest_differential,
            workload_differential=workload_differential,
            form_differential=form_differential,
            volatility_differential=volatility_differential,
            ranking_differential=ranking_differential,
        )

    completed = [p for p in perspectives if p.won is not None]
    a_win_rate = safe_rate(sum(1 for p in completed if p.won), len(completed)) if completed else None
    b_win_rate = (1.0 - a_win_rate) if a_win_rate is not None else None

    recent_window = settings.rolling_window_sizes[0]
    recent = [p for p in perspectives[:recent_window] if p.won is not None]
    recent_win_rate = safe_rate(sum(1 for p in recent if p.won), len(recent)) if recent else None

    margins = [p.sets_for - p.sets_against for p in perspectives if (p.sets_for + p.sets_against) > 0]
    average_margin = (sum(margins) / len(margins)) if margins else None

    deciders = [p for p in perspectives if p.is_deciding_set_match]
    decider_results = [p for p in deciders if p.deciding_set_won is not None]
    decider_win_rate = (
        safe_rate(sum(1 for p in decider_results if p.deciding_set_won), len(decider_results))
        if decider_results
        else None
    )

    days_since_last_meeting = (as_of - perspectives[0].effective_timestamp).total_seconds() / 86400.0

    competition_matches = (
        [p for p in perspectives if target_competition_id is not None and p.competition_id == target_competition_id]
    )
    competition_completed = [p for p in competition_matches if p.won is not None]
    competition_win_rate = (
        safe_rate(sum(1 for p in competition_completed if p.won), len(competition_completed))
        if competition_completed
        else None
    )

    format_matches = [p for p in perspectives if target_best_of is not None and p.best_of == target_best_of]
    format_completed = [p for p in format_matches if p.won is not None]
    format_win_rate = (
        safe_rate(sum(1 for p in format_completed if p.won), len(format_completed)) if format_completed else None
    )

    return MatchupFeatures(
        player_a_id=player_a_id,
        player_b_id=player_b_id,
        head_to_head_matches=len(perspectives),
        player_a_win_rate=a_win_rate,
        player_b_win_rate=b_win_rate,
        recent_head_to_head_win_rate_player_a=recent_win_rate,
        recent_head_to_head_n=len(recent),
        average_set_margin_player_a=average_margin,
        deciding_set_h2h_matches=len(deciders),
        deciding_set_h2h_win_rate_player_a=decider_win_rate,
        days_since_last_meeting=days_since_last_meeting,
        competition_specific_h2h_matches=len(competition_matches),
        competition_specific_h2h_win_rate_player_a=competition_win_rate,
        format_specific_h2h_matches=len(format_matches),
        format_specific_h2h_win_rate_player_a=format_win_rate,
        rest_differential_hours=rest_differential,
        workload_differential=workload_differential,
        form_differential=form_differential,
        volatility_differential=volatility_differential,
        ranking_differential=ranking_differential,
    )
