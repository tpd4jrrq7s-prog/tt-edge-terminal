"""Tests for IngestionService orchestration."""

from __future__ import annotations

from ingestion.models import RawMatch, RawOdds, RawPlayer
from ingestion.service import IngestionResult, IngestionService
from ingestion.sources.mock_source import MockTableTennisSource


class _StubSource:
    """A minimal MatchSource stub for testing orchestration in isolation."""

    name = "stub-source"

    def __init__(self) -> None:
        self._match = RawMatch(
            provider_match_id="m1",
            player_one=RawPlayer(provider_player_id="p1", full_name="Ma Long"),
            player_two=RawPlayer(provider_player_id="p2", full_name="Fan Zhendong"),
            status="scheduled",
            scheduled_at="2026-08-01T14:00:00+00:00",
        )
        self._odds = RawOdds(
            provider_match_id="m1",
            bookmaker="Pinnacle",
            player_one_odds=1.5,
            player_two_odds=2.5,
            captured_at="2026-08-01T13:00:00+00:00",
        )

    def fetch_matches(self) -> list[RawMatch]:
        return [self._match]

    def fetch_odds(self, provider_match_id: str) -> list[RawOdds]:
        return [self._odds] if provider_match_id == "m1" else []


def test_ingestion_service_normalizes_matches_and_odds():
    service = IngestionService(source=_StubSource())
    result = service.run_once()

    assert isinstance(result, IngestionResult)
    assert len(result.matches) == 1
    assert result.matches[0].id == "m1"
    assert len(result.odds) == 1
    assert result.odds[0].bookmaker == "Pinnacle"


def test_ingestion_service_works_with_mock_source():
    service = IngestionService(source=MockTableTennisSource())
    result = service.run_once()

    assert len(result.matches) == 3
    assert len(result.odds) == 4


def test_ingestion_service_is_stateless_across_runs():
    service = IngestionService(source=MockTableTennisSource())
    first = service.run_once()
    second = service.run_once()

    assert [m.id for m in first.matches] == [m.id for m in second.matches]
    assert [o.bookmaker for o in first.odds] == [o.bookmaker for o in second.odds]
