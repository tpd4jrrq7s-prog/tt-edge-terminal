"""Tests for ProviderMappingConfig validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from providers.models import ProviderMappingConfig


def test_valid_config_constructs():
    config = ProviderMappingConfig(provider="mock", mapping_version="1.0.0", status_map={"OK": "finished"})
    assert config.provider == "mock"


def test_invalid_status_map_target_rejected():
    with pytest.raises(ValidationError):
        ProviderMappingConfig(provider="mock", mapping_version="1.0.0", status_map={"OK": "not-a-real-status"})


def test_defaults_are_sensible():
    config = ProviderMappingConfig(provider="mock", mapping_version="1.0.0")
    assert config.sets_key == "sets"
    assert config.unknown_status_policy == "warn"
