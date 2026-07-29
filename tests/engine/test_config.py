"""Tests for typed, validated analytics settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.analytics import AnalyticsSettings, ProbabilityWeights, RiskWeights, get_analytics_settings


def test_default_settings_construct_without_error():
    settings = AnalyticsSettings()
    assert settings.min_history_sample_size == 3


def test_get_analytics_settings_is_cached():
    assert get_analytics_settings() is get_analytics_settings()


def test_probability_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        ProbabilityWeights(form=0.5, ranking=0.5, momentum=0.5, head_to_head=0.0, match_state=0.0, context=0.0)


def test_probability_weights_reject_negative_values():
    with pytest.raises(ValidationError):
        ProbabilityWeights(form=-0.1, ranking=0.3, momentum=0.3, head_to_head=0.2, match_state=0.2, context=0.1)


def test_risk_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        RiskWeights(
            data_quality=0.5,
            conflicting_signals=0.5,
            volatility=0.5,
            momentum_reversal=0.0,
            market_disagreement=0.0,
            market_movement=0.0,
            missing_data=0.0,
            short_format=0.0,
        )


def test_risk_thresholds_must_be_strictly_increasing():
    with pytest.raises(ValidationError):
        AnalyticsSettings(risk_low_threshold=50.0, risk_medium_threshold=25.0, risk_high_threshold=75.0)


def test_confidence_thresholds_must_be_strictly_increasing():
    with pytest.raises(ValidationError):
        AnalyticsSettings(
            confidence_low_threshold=0.8, confidence_medium_threshold=0.6, confidence_high_threshold=0.9
        )
