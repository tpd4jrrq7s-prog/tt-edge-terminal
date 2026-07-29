"""Deterministic, in-memory mock table tennis data source.

Returns fixed, realistic match and odds data. No randomness, no network
calls, no external APIs, and no database — everything lives in memory
for the lifetime of the `MockTableTennisSource` instance. This stands in
for a real provider until one is integrated in a later phase.
"""

from __future__ import annotations

from ingestion.models import RawMatch, RawOdds, RawPlayer, RawSetScore


def _build_players() -> dict[str, RawPlayer]:
    return {
        "ma_long": RawPlayer(
            provider_player_id="ma_long", full_name="Ma Long", country_code="CHN", world_ranking=1
        ),
        "fan_zhendong": RawPlayer(
            provider_player_id="fan_zhendong",
            full_name="Fan Zhendong",
            country_code="CHN",
            world_ranking=2,
        ),
        "timo_boll": RawPlayer(
            provider_player_id="timo_boll",
            full_name="Timo Boll",
            country_code="DEU",
            world_ranking=12,
        ),
        "dimitrij_ovtcharov": RawPlayer(
            provider_player_id="dimitrij_ovtcharov",
            full_name="Dimitrij Ovtcharov",
            country_code="DEU",
            world_ranking=15,
        ),
        "tomokazu_harimoto": RawPlayer(
            provider_player_id="tomokazu_harimoto",
            full_name="Tomokazu Harimoto",
            country_code="JPN",
            world_ranking=5,
        ),
        "hugo_calderano": RawPlayer(
            provider_player_id="hugo_calderano",
            full_name="Hugo Calderano",
            country_code="BRA",
            world_ranking=4,
        ),
    }


class MockTableTennisSource:
    """A deterministic in-memory `MatchSource` used until a real provider exists."""

    name = "mock-table-tennis-source"

    def __init__(self) -> None:
        players = _build_players()
        self._matches: list[RawMatch] = [
            RawMatch(
                provider_match_id="match-001",
                player_one=players["ma_long"],
                player_two=players["fan_zhendong"],
                status="scheduled",
                sets=[],
                scheduled_at="2026-08-01T14:00:00+00:00",
            ),
            RawMatch(
                provider_match_id="match-002",
                player_one=players["timo_boll"],
                player_two=players["dimitrij_ovtcharov"],
                status="live",
                sets=[
                    RawSetScore(player_one_points=11, player_two_points=7),
                    RawSetScore(player_one_points=9, player_two_points=11),
                    RawSetScore(player_one_points=6, player_two_points=4),
                ],
                scheduled_at="2026-07-29T10:00:00+00:00",
            ),
            RawMatch(
                provider_match_id="match-003",
                player_one=players["tomokazu_harimoto"],
                player_two=players["hugo_calderano"],
                status="finished",
                sets=[
                    RawSetScore(player_one_points=11, player_two_points=9),
                    RawSetScore(player_one_points=11, player_two_points=6),
                    RawSetScore(player_one_points=8, player_two_points=11),
                    RawSetScore(player_one_points=11, player_two_points=5),
                ],
                scheduled_at="2026-07-28T09:00:00+00:00",
            ),
        ]
        self._odds_by_match: dict[str, list[RawOdds]] = {
            "match-001": [
                RawOdds(
                    provider_match_id="match-001",
                    bookmaker="Pinnacle",
                    player_one_odds=1.55,
                    player_two_odds=2.45,
                    captured_at="2026-08-01T13:00:00+00:00",
                ),
                RawOdds(
                    provider_match_id="match-001",
                    bookmaker="Betfair",
                    player_one_odds=1.58,
                    player_two_odds=2.40,
                    captured_at="2026-08-01T13:05:00+00:00",
                ),
            ],
            "match-002": [
                RawOdds(
                    provider_match_id="match-002",
                    bookmaker="Pinnacle",
                    player_one_odds=2.10,
                    player_two_odds=1.75,
                    captured_at="2026-07-29T10:05:00+00:00",
                ),
            ],
            "match-003": [
                RawOdds(
                    provider_match_id="match-003",
                    bookmaker="Betfair",
                    player_one_odds=1.40,
                    player_two_odds=3.00,
                    captured_at="2026-07-28T08:55:00+00:00",
                ),
            ],
        }

    def fetch_matches(self) -> list[RawMatch]:
        """Return the fixed set of mock matches."""
        return list(self._matches)

    def fetch_odds(self, provider_match_id: str) -> list[RawOdds]:
        """Return the fixed odds quotes for a mock match, or an empty list if unknown."""
        return list(self._odds_by_match.get(provider_match_id, []))
