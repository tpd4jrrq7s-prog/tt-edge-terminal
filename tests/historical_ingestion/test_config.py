"""Tests for HistoricalIngestionSettings validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.historical_ingestion import HistoricalIngestionSettings, get_historical_ingestion_settings


def test_defaults_construct_without_error():
    settings = HistoricalIngestionSettings()
    assert settings.batch_size == 100
    assert settings.validation_policy == "strict"


def test_settings_are_cached():
    assert get_historical_ingestion_settings() is get_historical_ingestion_settings()


def test_batch_size_must_be_positive():
    with pytest.raises(ValidationError):
        HistoricalIngestionSettings(batch_size=0)


def test_mapping_version_must_look_like_semver():
    with pytest.raises(ValidationError):
        HistoricalIngestionSettings(mapping_version="not-a-version")


def test_invalid_literal_policy_rejected():
    with pytest.raises(ValidationError):
        HistoricalIngestionSettings(validation_policy="loose")
