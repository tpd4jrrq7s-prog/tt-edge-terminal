"""Typed contracts for rolling player features, matchup features, and the
leakage-safe FeatureSnapshot that bundles them.

Every rate/average feature is `float | None`: `None` means "not enough
data to compute this", never a fabricated zero or a fabricated 0.5.
Every rate feature that has a natural observation count is paired with
an explicit `..._n` count field so consumers can judge reliability
themselves rather than guessing from the rate alone.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class WindowFeatures(BaseModel):
    """Rolling features for one player over one specific window (e.g. "last_5", "all_time")."""

    window_label: str
    matches_played: int = Field(..., ge=0)
    wins: int = Field(..., ge=0)
    losses: int = Field(..., ge=0)
    win_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    sets_won: int = Field(default=0, ge=0)
    sets_lost: int = Field(default=0, ge=0)
    set_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    points_won: int = Field(default=0, ge=0)
    points_lost: int = Field(default=0, ge=0)
    point_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    average_set_margin: float | None = None
    average_point_margin: float | None = None

    straight_sets_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    straight_sets_win_rate_n: int = Field(default=0, ge=0)

    deciding_set_appearance_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    deciding_set_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    deciding_set_win_rate_n: int = Field(default=0, ge=0)

    first_set_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    first_set_win_rate_n: int = Field(default=0, ge=0)

    comeback_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    comeback_win_rate_n: int = Field(default=0, ge=0)

    loss_rate_after_winning_first_set: float | None = Field(default=None, ge=0.0, le=1.0)
    loss_rate_after_winning_first_set_n: int = Field(default=0, ge=0)

    average_match_duration_minutes: float | None = None
    average_match_duration_n: int = Field(default=0, ge=0)

    incomplete_match_count: int = Field(default=0, ge=0)


class PlayerRollingFeatures(BaseModel):
    """A player's rolling features across every configured window, plus non-windowed signals."""

    player_id: str
    windows: list[WindowFeatures] = Field(default_factory=list)

    rest_hours_since_previous_match: float | None = Field(default=None, ge=0.0)
    matches_in_previous_24h: int = Field(default=0, ge=0)
    matches_in_previous_7d: int = Field(default=0, ge=0)

    result_streak: int = Field(default=0, description="Positive = win streak length, negative = loss streak")
    recency_weighted_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    opponent_adjusted_win_rate: float | None = Field(
        default=None,
        description="Reserved for future use: requires historical ranking/strength data not modeled in Phase 3",
    )
    ranking: int | None = None

    volatility_score: float | None = Field(default=None, ge=0.0, le=1.0)
    historical_data_quality_average: float | None = Field(default=None, ge=0.0, le=100.0)

    observation_count: int = Field(default=0, ge=0)

    def window(self, label: str) -> WindowFeatures | None:
        return next((w for w in self.windows if w.window_label == label), None)


class MatchupFeatures(BaseModel):
    """Head-to-head / matchup features between two specific players."""

    player_a_id: str
    player_b_id: str

    head_to_head_matches: int = Field(default=0, ge=0)
    player_a_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    player_b_win_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    recent_head_to_head_win_rate_player_a: float | None = Field(default=None, ge=0.0, le=1.0)
    recent_head_to_head_n: int = Field(default=0, ge=0)

    average_set_margin_player_a: float | None = None

    deciding_set_h2h_matches: int = Field(default=0, ge=0)
    deciding_set_h2h_win_rate_player_a: float | None = Field(default=None, ge=0.0, le=1.0)

    days_since_last_meeting: float | None = Field(default=None, ge=0.0)

    competition_specific_h2h_matches: int = Field(default=0, ge=0)
    competition_specific_h2h_win_rate_player_a: float | None = Field(default=None, ge=0.0, le=1.0)

    format_specific_h2h_matches: int = Field(default=0, ge=0)
    format_specific_h2h_win_rate_player_a: float | None = Field(default=None, ge=0.0, le=1.0)

    ranking_differential: float | None = Field(
        default=None, description="Reserved: requires historical ranking data not modeled in Phase 3"
    )
    rest_differential_hours: float | None = None
    workload_differential: float | None = None
    form_differential: float | None = None
    volatility_differential: float | None = None


class ProvenanceMetadata(BaseModel):
    """Full provenance for one FeatureSnapshot: what was used, and what wasn't."""

    player_a_source_match_ids: list[str] = Field(default_factory=list)
    player_b_source_match_ids: list[str] = Field(default_factory=list)
    head_to_head_source_match_ids: list[str] = Field(default_factory=list)

    cutoff: datetime
    repository_fingerprint: str
    input_fingerprint: str

    feature_schema_version: str
    builder_version: str

    player_a_observation_count: int = Field(default=0, ge=0)
    player_b_observation_count: int = Field(default=0, ge=0)
    head_to_head_observation_count: int = Field(default=0, ge=0)

    warnings: list[str] = Field(default_factory=list)
    missing_feature_names: list[str] = Field(default_factory=list)
    data_quality_score: float = Field(..., ge=0.0, le=100.0)


class FeatureSnapshot(BaseModel):
    """A leakage-safe bundle of features available immediately before `as_of`."""

    id: str = Field(..., min_length=1)
    target_match_id: str = Field(..., min_length=1)
    as_of: datetime
    player_a_id: str = Field(..., min_length=1)
    player_b_id: str = Field(..., min_length=1)
    player_a_features: PlayerRollingFeatures
    player_b_features: PlayerRollingFeatures
    matchup_features: MatchupFeatures
    provenance: ProvenanceMetadata
    feature_schema_version: str
    builder_version: str
