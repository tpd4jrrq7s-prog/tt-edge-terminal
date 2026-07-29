"""Tests for match normalization."""

from __future__ import annotations

import pytest

from domain.match import MatchStatus
from ingestion.models import RawMatch, RawPlayer, RawSetScore
from normalization.match_normalizer import MatchNormalizationError, normalize_match


def _raw_match(**overrides) -> RawMatch:
    defaults = dict(
        provider_match_id="m1",
        player_one=RawPlayer(provider_player_id="p1", full_name="Ma Long"),
        player_two=RawPlayer(provider_player_id="p2", full_name="Fan Zhendong"),
        status="scheduled",
        sets=[],
        scheduled_at="2026-08-01T14:00:00+00:00",
    )
    defaults.update(overrides)
    return RawMatch(**defaults)


def test_normalize_match_maps_fields_correctly():
    match = normalize_match(_raw_match())
    assert match.id == "m1"
    assert match.player_one.name == "Ma Long"
    assert match.player_two.name == "Fan Zhendong"
    assert match.status is MatchStatus.SCHEDULED


@pytest.mark.parametrize(
    "raw_status,expected",
    [
        ("scheduled", MatchStatus.SCHEDULED),
        ("LIVE", MatchStatus.LIVE),
        ("in_progress", MatchStatus.LIVE),
        ("finished", MatchStatus.FINISHED),
        ("cancelled", MatchStatus.CANCELLED),
    ],
)
def test_normalize_match_status_mapping(raw_status, expected):
    match = normalize_match(_raw_match(status=raw_status))
    assert match.status is expected


def test_normalize_match_rejects_unknown_status():
    with pytest.raises(MatchNormalizationError):
        normalize_match(_raw_match(status="postponed"))


def test_normalize_match_rejects_same_player_on_both_sides():
    same_player = RawPlayer(provider_player_id="p1", full_name="Ma Long")
    with pytest.raises(MatchNormalizationError):
        normalize_match(_raw_match(player_one=same_player, player_two=same_player))


def test_normalize_match_rejects_blank_player_name():
    blank_named = RawPlayer(provider_player_id="p2", full_name="   ")
    with pytest.raises(MatchNormalizationError):
        normalize_match(_raw_match(player_two=blank_named))


def test_normalize_match_rejects_invalid_scheduled_at():
    with pytest.raises(MatchNormalizationError):
        normalize_match(_raw_match(scheduled_at="not-a-timestamp"))


def test_normalize_match_converts_sets():
    sets = [RawSetScore(player_one_points=11, player_two_points=7)]
    match = normalize_match(_raw_match(sets=sets))
    assert len(match.sets) == 1
    assert match.sets[0].player_one_points == 11
    assert match.sets[0].player_two_points == 7
