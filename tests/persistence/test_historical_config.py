"""Tests for typed, validated historical-intelligence settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.historical import HistoricalIntelligenceSettings, get_historical_intelligence_settings


def test_defaults_construct_without_error():
    settings = HistoricalIntelligenceSettings()
    assert settings.rolling_window_sizes == [5, 10, 20]


def test_settings_are_cached():
    assert get_historical_intelligence_settings() is get_historical_intelligence_settings()


def test_rolling_window_sizes_must_be_positive():
    with pytest.raises(ValidationError):
        HistoricalIntelligenceSettings(rolling_window_sizes=[5, -1])


def test_rolling_window_sizes_must_be_sorted():
    with pytest.raises(ValidationError):
        HistoricalIntelligenceSettings(rolling_window_sizes=[10, 5])


def test_rolling_window_sizes_must_not_have_duplicates():
    with pytest.raises(ValidationError):
        HistoricalIntelligenceSettings(rolling_window_sizes=[5, 5, 10])


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ValidationError):
        HistoricalIntelligenceSettings(split_train_ratio=0.5, split_validation_ratio=0.3, split_test_ratio=0.3)


def test_version_identifiers_must_look_like_semver():
    with pytest.raises(ValidationError):
        HistoricalIntelligenceSettings(feature_schema_version="not-a-version")


def test_identity_thresholds_are_bounded():
    with pytest.raises(ValidationError):
        HistoricalIntelligenceSettings(identity_match_threshold=1.5)
