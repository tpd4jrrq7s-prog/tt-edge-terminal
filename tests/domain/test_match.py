"""Tests for the domain Match model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.match import Match, MatchStatus, SetScore
from domain.player import Player


def _players() -> tuple[Player, Player]:
    return Player(id="p1", name="Ma Long"), Player(id="p2", name="Fan Zhendong")


def test_match_default_status_is_scheduled():
    p1, p2 = _players()
    match = Match(id="m1", player_one=p1, player_two=p2)
    assert match.status is MatchStatus.SCHEDULED
    assert match.sets == []


def test_match_is_live_property():
    p1, p2 = _players()
    match = Match(id="m1", player_one=p1, player_two=p2, status=MatchStatus.LIVE)
    assert match.is_live is True

    match.status = MatchStatus.FINISHED
    assert match.is_live is False


def test_sets_won_counts_correctly():
    p1, p2 = _players()
    sets = [
        SetScore(player_one_points=11, player_two_points=7),
        SetScore(player_one_points=9, player_two_points=11),
        SetScore(player_one_points=11, player_two_points=5),
    ]
    match = Match(id="m1", player_one=p1, player_two=p2, sets=sets)
    assert match.sets_won_player_one == 2
    assert match.sets_won_player_two == 1


def test_set_score_rejects_negative_points():
    with pytest.raises(ValidationError):
        SetScore(player_one_points=-1, player_two_points=0)


def test_match_requires_id_and_players():
    with pytest.raises(ValidationError):
        Match()  # type: ignore[call-arg]
