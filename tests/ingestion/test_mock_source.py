"""Tests for the deterministic MockTableTennisSource."""

from __future__ import annotations

from ingestion.sources.mock_source import MockTableTennisSource


def test_mock_source_returns_expected_matches():
    source = MockTableTennisSource()
    matches = source.fetch_matches()
    assert len(matches) == 3
    ids = {m.provider_match_id for m in matches}
    assert ids == {"match-001", "match-002", "match-003"}


def test_mock_source_is_deterministic_across_calls():
    source = MockTableTennisSource()
    first = [m.model_dump() for m in source.fetch_matches()]
    second = [m.model_dump() for m in source.fetch_matches()]
    assert first == second


def test_mock_source_is_deterministic_across_instances():
    first = [m.model_dump() for m in MockTableTennisSource().fetch_matches()]
    second = [m.model_dump() for m in MockTableTennisSource().fetch_matches()]
    assert first == second


def test_mock_source_returns_odds_for_known_match():
    source = MockTableTennisSource()
    odds = source.fetch_odds("match-001")
    assert len(odds) == 2
    assert all(o.provider_match_id == "match-001" for o in odds)


def test_mock_source_returns_empty_odds_for_unknown_match():
    source = MockTableTennisSource()
    assert source.fetch_odds("does-not-exist") == []


def test_mock_source_has_a_name():
    source = MockTableTennisSource()
    assert source.name == "mock-table-tennis-source"
