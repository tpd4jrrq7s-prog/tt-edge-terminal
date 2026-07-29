"""Tests for deterministic player-name normalization."""

from __future__ import annotations

import pytest

from identity.errors import InvalidPlayerNameError
from identity.normalizer import build_normalized_identity, normalize_player_name


def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize_player_name("  Ma   Long  ") == "ma long"


def test_normalize_folds_accents():
    assert normalize_player_name("Timo Böll") == normalize_player_name("Timo Boll")


def test_normalize_strips_punctuation():
    assert normalize_player_name("O'Brien-Smith") == "o brien smith"


def test_normalize_is_deterministic():
    assert normalize_player_name("Fan Zhendong") == normalize_player_name("Fan Zhendong")


def test_normalize_rejects_blank_name():
    with pytest.raises(InvalidPlayerNameError):
        normalize_player_name("   ")


def test_build_normalized_identity_carries_hints():
    identity = build_normalized_identity("Ma Long", country="CHN", external_provider="mock", external_player_id="1")
    assert identity.original_name == "Ma Long"
    assert identity.normalized_name == "ma long"
    assert identity.country == "CHN"
    assert identity.external_provider == "mock"
