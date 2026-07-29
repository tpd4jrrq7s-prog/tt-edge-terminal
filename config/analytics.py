"""Typed, validated configuration for the Phase 2B analytics engine.

All weights and thresholds are read from environment variables (prefixed
`ANALYTICS_`, nested with `__`) or a local `.env` file, falling back to
the documented defaults below. Invalid configuration (weights that don't
sum to 1, thresholds out of order, etc.) fails fast with a clear
`pydantic.ValidationError` at construction time.
"""

from __future__ import annotations

import math
from functools import lru_cache

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _validate_weights_sum_to_one(weights: dict[str, float], *, label: str) -> None:
    for name, value in weights.items():
        if value < 0:
            raise ValueError(f"{label} weight '{name}' must be >= 0 (got {value})")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"{label} weights must sum to 1.0 (got {total:.6f})")


class ProbabilityWeights(BaseModel):
    """Weights for each factor in the probability engine's logistic combination."""

    form: float = 0.30
    ranking: float = 0.25
    momentum: float = 0.20
    head_to_head: float = 0.15
    match_state: float = 0.10
    context: float = 0.00

    @model_validator(mode="after")
    def _validate(self) -> "ProbabilityWeights":
        _validate_weights_sum_to_one(self.model_dump(), label="Probability")
        return self


class RiskWeights(BaseModel):
    """Weights for each contributing component of the overall risk score."""

    data_quality: float = 0.20
    conflicting_signals: float = 0.15
    volatility: float = 0.15
    momentum_reversal: float = 0.10
    market_disagreement: float = 0.15
    market_movement: float = 0.10
    missing_data: float = 0.10
    short_format: float = 0.05

    @model_validator(mode="after")
    def _validate(self) -> "RiskWeights":
        _validate_weights_sum_to_one(self.model_dump(), label="Risk")
        return self


class AnalyticsSettings(BaseSettings):
    """Central, validated configuration for every engine module."""

    model_config = SettingsConfigDict(
        env_prefix="ANALYTICS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    probability_weights: ProbabilityWeights = Field(default_factory=ProbabilityWeights)
    risk_weights: RiskWeights = Field(default_factory=RiskWeights)

    # Sample size / recency
    min_history_sample_size: int = Field(default=3, ge=0)
    momentum_recent_window: int = Field(default=5, ge=1)
    momentum_min_points: int = Field(default=8, ge=1)
    pressure_point_score_threshold: int = Field(default=9, ge=1)
    form_recency_half_life_days: float = Field(default=90.0, gt=0)
    odds_max_age_minutes: float = Field(default=30.0, gt=0)
    ranking_scale: float = Field(default=50.0, gt=0)

    # Risk classification thresholds (0-100 scale), strictly increasing
    risk_low_threshold: float = Field(default=25.0, ge=0, le=100)
    risk_medium_threshold: float = Field(default=50.0, ge=0, le=100)
    risk_high_threshold: float = Field(default=75.0, ge=0, le=100)

    # Confidence classification thresholds (0-1 scale), strictly increasing
    confidence_low_threshold: float = Field(default=0.35, ge=0, le=1)
    confidence_medium_threshold: float = Field(default=0.6, ge=0, le=1)
    confidence_high_threshold: float = Field(default=0.8, ge=0, le=1)

    # Value engine gating thresholds
    min_probability_edge: float = Field(default=0.03, ge=0.0, le=1.0)
    min_expected_value: float = Field(default=0.02)
    min_confidence_for_value: float = Field(default=0.5, ge=0.0, le=1.0)
    max_risk_for_value: float = Field(default=70.0, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "AnalyticsSettings":
        if not (0 <= self.risk_low_threshold < self.risk_medium_threshold < self.risk_high_threshold <= 100):
            raise ValueError(
                "risk thresholds must satisfy "
                "0 <= risk_low_threshold < risk_medium_threshold < risk_high_threshold <= 100"
            )
        if not (
            0 <= self.confidence_low_threshold
            < self.confidence_medium_threshold
            < self.confidence_high_threshold
            <= 1
        ):
            raise ValueError(
                "confidence thresholds must satisfy "
                "0 <= confidence_low_threshold < confidence_medium_threshold "
                "< confidence_high_threshold <= 1"
            )
        return self


@lru_cache
def get_analytics_settings() -> AnalyticsSettings:
    """Return a cached, validated AnalyticsSettings instance."""
    return AnalyticsSettings()
