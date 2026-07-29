"""Tests for the domain Player model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.player import Player


def test_player_minimal_construction():
    player = Player(id="p1", name="Ma Long")
    assert player.id == "p1"
    assert player.name == "Ma Long"
    assert player.country is None
    assert player.ranking is None


def test_player_full_construction():
    player = Player(id="p2", name="Fan Zhendong", country="CHN", ranking=2)
    assert player.country == "CHN"
    assert player.ranking == 2


def test_player_str_returns_name():
    player = Player(id="p1", name="Ma Long")
    assert str(player) == "Ma Long"


def test_player_requires_id_and_name():
    with pytest.raises(ValidationError):
        Player(name="Ma Long")  # type: ignore[call-arg]


def test_player_ranking_must_be_at_least_one():
    with pytest.raises(ValidationError):
        Player(id="p1", name="Ma Long", ranking=0)
