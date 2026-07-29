"""Tests for odds normalization."""

from __future__ import annotations

import pytest

from ingestion.models import RawOdds
from normalization.odds_normalizer import OddsNormalizationError, normalize_odds


def _raw_odds(**overrides) -> RawOdds:
    defaults = dict(
        provider_match_id="m1",
        bookmaker="Pinnacle",
        player_one_odds=1.5,
        player_two_odds=2.5,
        captured_at="2026-08-01T13:00:00+00:00",
    )
    defaults.update(overrides)
    return RawOdds(**defaults)


def test_normalize_odds_maps_fields_correctly():
    odds = normalize_odds(_raw_odds())
    assert odds.match_id == "m1"
    assert odds.bookmaker == "Pinnacle"
    assert odds.player_one_odds == 1.5
    assert odds.player_two_odds == 2.5
    assert odds.captured_at.isoformat() == "2026-08-01T13:00:00+00:00"


def test_normalize_odds_rejects_non_decimal_odds():
    with pytest.raises(OddsNormalizationError):
        normalize_odds(_raw_odds(player_one_odds=1.0))


def test_normalize_odds_rejects_odds_below_one():
    with pytest.raises(OddsNormalizationError):
        normalize_odds(_raw_odds(player_two_odds=0.9))


def test_normalize_odds_rejects_blank_bookmaker():
    with pytest.raises(OddsNormalizationError):
        normalize_odds(_raw_odds(bookmaker="   "))


def test_normalize_odds_rejects_blank_match_id():
    with pytest.raises(OddsNormalizationError):
        normalize_odds(_raw_odds(provider_match_id="   "))


def test_normalize_odds_rejects_invalid_timestamp():
    with pytest.raises(OddsNormalizationError):
        normalize_odds(_raw_odds(captured_at="not-a-timestamp"))
