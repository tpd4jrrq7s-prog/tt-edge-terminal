"""Typed input and output contracts for the analytics engine.

Every public engine function accepts and returns these Pydantic models —
never loose dictionaries. Input models (HistoricalMatch, PointEvent,
HeadToHeadRecord, CompetitionContext, MatchAnalysisRequest) describe
optional, caller-supplied context; every field the engine cannot derive
from real input is left `None`/empty rather than fabricated.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from domain.match import Match
from domain.odds import Odds


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class MomentumState(str, Enum):
    """Which kind of momentum signal is available for a match."""

    PRE_MATCH = "pre_match"
    IN_PLAY = "in_play"
    NO_DATA = "no_data"


class RiskLabel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class ConfidenceLabel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ValueDecision(str, Enum):
    NO_SIGNAL = "no_signal"
    OBSERVE = "observe"
    POSSIBLE_VALUE = "possible_value"
    STRONG_VALUE = "strong_value"


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# --------------------------------------------------------------------------
# Engine-level input models
# --------------------------------------------------------------------------


class SetResult(BaseModel):
    """A single completed set score, from the subject player's perspective."""

    player_points: int = Field(..., ge=0)
    opponent_points: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _not_a_tie(self) -> "SetResult":
        if self.player_points == self.opponent_points:
            raise ValueError("A completed table tennis set cannot end in a tie")
        return self

    @property
    def won(self) -> bool:
        return self.player_points > self.opponent_points

    @property
    def margin(self) -> int:
        return self.player_points - self.opponent_points

    @property
    def total_points(self) -> int:
        return self.player_points + self.opponent_points


class HistoricalMatch(BaseModel):
    """One completed historical match, from one player's point of view.

    `won` is the authoritative outcome (also covers walkovers, where no
    set-level detail exists). `sets`, when provided, must not contradict
    it. Nothing here is inferred beyond what the caller supplies.
    """

    player_id: str = Field(..., min_length=1)
    opponent_id: str = Field(..., min_length=1)
    opponent_ranking: int | None = Field(default=None, ge=1)
    played_at: datetime
    won: bool
    sets: list[SetResult] = Field(default_factory=list)
    walkover: bool = False
    retired: bool = False

    @model_validator(mode="after")
    def _player_and_opponent_differ(self) -> "HistoricalMatch":
        if self.player_id == self.opponent_id:
            raise ValueError("A historical match cannot have the same player as their own opponent")
        return self

    @model_validator(mode="after")
    def _sets_consistent_with_result(self) -> "HistoricalMatch":
        if self.sets and not self.walkover:
            if self.sets_won > self.sets_lost and not self.won:
                raise ValueError("Set scores indicate a win but 'won' is False")
            if self.sets_lost > self.sets_won and self.won:
                raise ValueError("Set scores indicate a loss but 'won' is True")
        return self

    @property
    def sets_won(self) -> int:
        return sum(1 for s in self.sets if s.won)

    @property
    def sets_lost(self) -> int:
        return sum(1 for s in self.sets if not s.won)

    @property
    def point_margin(self) -> int:
        return sum(s.margin for s in self.sets)


class PointEvent(BaseModel):
    """One point played within the current match, in chronological (list) order."""

    set_number: int = Field(..., ge=1)
    winner: Literal["player_one", "player_two"]
    player_one_score: int = Field(..., ge=0, description="Cumulative score after this point")
    player_two_score: int = Field(..., ge=0, description="Cumulative score after this point")


class HeadToHeadRecord(BaseModel):
    """Aggregate head-to-head history between the two players in the current match."""

    player_one_wins: int = Field(default=0, ge=0)
    player_two_wins: int = Field(default=0, ge=0)
    last_played_at: datetime | None = None

    @property
    def total_matches(self) -> int:
        return self.player_one_wins + self.player_two_wins


class CompetitionContext(BaseModel):
    """Optional context about the current match not carried by the domain Match model."""

    surface: str | None = None
    competition_name: str | None = None
    best_of_sets: int | None = Field(default=None, ge=1)


class MatchAnalysisRequest(BaseModel):
    """Everything the analytics engine needs for a single match analysis.

    Only `match` is required. Every other field is optional — the engine
    must produce a valid (if lower-confidence) analysis without them.
    """

    match: Match
    odds: list[Odds] = Field(default_factory=list)
    player_one_history: list[HistoricalMatch] = Field(default_factory=list)
    player_two_history: list[HistoricalMatch] = Field(default_factory=list)
    point_progression: list[PointEvent] = Field(default_factory=list)
    head_to_head: HeadToHeadRecord | None = None
    context: CompetitionContext | None = None

    @model_validator(mode="after")
    def _inputs_reference_this_match(self) -> "MatchAnalysisRequest":
        for odds in self.odds:
            if odds.match_id != self.match.id:
                raise ValueError(
                    f"Odds match_id {odds.match_id!r} does not match request match id {self.match.id!r}"
                )
        for entry in self.player_one_history:
            if entry.player_id != self.match.player_one.id:
                raise ValueError("player_one_history contains an entry that does not belong to player_one")
        for entry in self.player_two_history:
            if entry.player_id != self.match.player_two.id:
                raise ValueError("player_two_history contains an entry that does not belong to player_two")
        return self


# --------------------------------------------------------------------------
# Small typed components (used inside the larger result models below)
# --------------------------------------------------------------------------


class MomentumComponent(BaseModel):
    """One named contributor to a momentum score, for transparency/explanations."""

    name: str
    value: float
    detail: str


class ProbabilityFactor(BaseModel):
    """One named contributor to the probability engine's logistic combination."""

    name: str
    weight: float
    raw_signal: float = Field(..., ge=-1.0, le=1.0)
    weighted_contribution: float
    description: str


class ConfidenceReason(BaseModel):
    """One factor that increased or decreased the overall confidence score."""

    factor: str
    direction: Literal["increases", "decreases"]
    detail: str


class RiskFactor(BaseModel):
    """One named contributor (0-100 severity) to the overall risk score."""

    name: str
    severity: float = Field(..., ge=0.0, le=100.0)
    detail: str


class DataQualityIssue(BaseModel):
    """A single, specific data-quality problem found in the request."""

    field: str
    detail: str
    severity: IssueSeverity


# --------------------------------------------------------------------------
# Per-stage result models
# --------------------------------------------------------------------------


class FormResult(BaseModel):
    """Output of the form engine for a single player."""

    score: float = Field(..., ge=0.0, le=100.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    matches_considered: int = Field(..., ge=0)
    average_recency_weight: float | None = None


class MomentumResult(BaseModel):
    """Output of the momentum engine for a match (both players)."""

    player_one_score: float = Field(..., ge=0.0, le=100.0)
    player_two_score: float = Field(..., ge=0.0, le=100.0)
    state: MomentumState
    confidence: float = Field(..., ge=0.0, le=1.0)
    components: list[MomentumComponent] = Field(default_factory=list)


class PlayerMetrics(BaseModel):
    """Derived, per-player metrics feeding into match-level features."""

    player_id: str
    player_name: str
    ranking: int | None = None
    form_score: float = Field(..., ge=0.0, le=100.0)
    form_confidence: float = Field(..., ge=0.0, le=1.0)
    matches_considered: int = Field(..., ge=0)
    momentum_score: float = Field(..., ge=0.0, le=100.0)
    momentum_state: MomentumState


class MatchFeatures(BaseModel):
    """Comparable, derived features between the two players for this match."""

    player_one: PlayerMetrics
    player_two: PlayerMetrics
    ranking_differential: float | None = Field(default=None, ge=-1.0, le=1.0)
    form_differential: float = Field(..., ge=-1.0, le=1.0)
    momentum_differential: float = Field(..., ge=-1.0, le=1.0)
    head_to_head: HeadToHeadRecord | None = None
    head_to_head_signal: float = Field(..., ge=-1.0, le=1.0)
    match_state_signal: float = Field(..., ge=-1.0, le=1.0)
    context: CompetitionContext | None = None


class ProbabilityResult(BaseModel):
    """Output of the probability engine."""

    player_one_probability: float = Field(..., gt=0.0, lt=1.0)
    player_two_probability: float = Field(..., gt=0.0, lt=1.0)
    factors: list[ProbabilityFactor] = Field(default_factory=list)
    data_quality_penalty: float = Field(..., ge=0.0, le=1.0)
    calibration_ready: bool
    method: str = "weighted_logistic_v1"


class ConfidenceAssessment(BaseModel):
    """Output of the confidence engine. Independent of ProbabilityResult's values."""

    score: float = Field(..., ge=0.0, le=1.0)
    label: ConfidenceLabel
    reasons: list[ConfidenceReason] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """Output of the risk engine."""

    score: float = Field(..., ge=0.0, le=100.0)
    label: RiskLabel
    factors: list[RiskFactor] = Field(default_factory=list)


class PlayerValueAssessment(BaseModel):
    """Value assessment for a single player's side of the market."""

    player_id: str
    bookmaker: str
    decimal_odds: float = Field(..., gt=1.0)
    model_probability: float = Field(..., gt=0.0, lt=1.0)
    implied_probability: float = Field(..., gt=0.0, lt=1.0)
    fair_odds: float = Field(..., gt=1.0)
    probability_edge: float
    expected_value: float
    value_score: float = Field(..., ge=0.0, le=100.0)
    decision: ValueDecision


class ValueAssessment(BaseModel):
    """Output of the value engine, covering both sides of the market."""

    player_one: PlayerValueAssessment | None = None
    player_two: PlayerValueAssessment | None = None
    odds_considered: int = Field(default=0, ge=0)
    market_disagreement: float | None = Field(default=None, ge=0.0)


class PatternSignal(BaseModel):
    """A single detected behavioral pattern for one player."""

    player_id: str
    pattern: str
    strength: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    sample_size: int = Field(..., ge=0)
    supporting_observations: list[str] = Field(default_factory=list)


class DataQualityAssessment(BaseModel):
    """Output of the data-quality engine."""

    score: float = Field(..., ge=0.0, le=100.0)
    warnings: list[DataQualityIssue] = Field(default_factory=list)
    history_sample_size_player_one: int = Field(..., ge=0)
    history_sample_size_player_two: int = Field(..., ge=0)
    odds_available: bool
    odds_fresh: bool


class MatchAnalysis(BaseModel):
    """The single, top-level typed result returned by the analytics engine."""

    match_id: str
    generated_at: datetime
    match_features: MatchFeatures
    probability: ProbabilityResult
    confidence: ConfidenceAssessment
    risk: RiskAssessment
    value: ValueAssessment
    patterns: list[PatternSignal] = Field(default_factory=list)
    data_quality: DataQualityAssessment
    explanations: list[str] = Field(default_factory=list)
