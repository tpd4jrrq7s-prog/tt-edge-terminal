"""Tests for raw ingestion model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ingestion.models import RawMatch, RawOdds, RawPlayer, RawSetScore


def test_raw_player_requires_id_and_name():
    with pytest.raises(ValidationError):
        RawPlayer(full_name="Ma Long")  # type: ignore[call-arg]


def test_raw_set_score_defaults_to_zero_zero():
    s = RawSetScore()
    assert s.player_one_points == 0
    assert s.player_two_points == 0


def test_raw_set_score_rejects_negative_points():
    with pytest.raises(ValidationError):
        RawSetScore(player_one_points=-1, player_two_points=0)


def test_raw_match_requires_players_status_and_schedule():
    player_one = RawPlayer(provider_player_id="p1", full_name="Ma Long")
    player_two = RawPlayer(provider_player_id="p2", full_name="Fan Zhendong")
    match = RawMatch(
        provider_match_id="m1",
        player_one=player_one,
        player_two=player_two,
        status="scheduled",
        scheduled_at="2026-08-01T14:00:00+00:00",
    )
    assert match.sets == []


def test_raw_match_missing_status_is_rejected():
    player_one = RawPlayer(provider_player_id="p1", full_name="Ma Long")
    player_two = RawPlayer(provider_player_id="p2", full_name="Fan Zhendong")
    with pytest.raises(ValidationError):
        RawMatch(  # type: ignore[call-arg]
            provider_match_id="m1",
            player_one=player_one,
            player_two=player_two,
            scheduled_at="2026-08-01T14:00:00+00:00",
        )


def test_raw_odds_accepts_out_of_range_odds_without_domain_validation():
    # The raw layer is intentionally permissive; normalization enforces
    # that odds are valid decimal odds (> 1.0).
    odds = RawOdds(
        provider_match_id="m1",
        bookmaker="Pinnacle",
        player_one_odds=0.5,
        player_two_odds=2.0,
        captured_at="2026-08-01T13:00:00+00:00",
    )
    assert odds.player_one_odds == 0.5
