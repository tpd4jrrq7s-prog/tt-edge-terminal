"""Tests for the domain Odds model."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from domain.odds import Odds


def test_odds_construction_and_implied_probability():
    odds = Odds(match_id="m1", bookmaker="Pinnacle", player_one_odds=1.5, player_two_odds=2.75)
    assert odds.match_id == "m1"
    assert odds.bookmaker == "Pinnacle"
    assert odds.implied_probability_player_one == pytest.approx(1.0 / 1.5)
    assert odds.implied_probability_player_two == pytest.approx(1.0 / 2.75)


def test_odds_captured_at_defaults_to_utc_now():
    before = datetime.now(timezone.utc)
    odds = Odds(match_id="m1", bookmaker="Pinnacle", player_one_odds=1.5, player_two_odds=2.75)
    after = datetime.now(timezone.utc)
    assert before <= odds.captured_at <= after


def test_odds_rejects_non_decimal_odds():
    with pytest.raises(ValidationError):
        Odds(match_id="m1", bookmaker="Pinnacle", player_one_odds=1.0, player_two_odds=2.0)


def test_odds_requires_match_id_and_bookmaker():
    with pytest.raises(ValidationError):
        Odds(player_one_odds=1.5, player_two_odds=2.5)  # type: ignore[call-arg]
