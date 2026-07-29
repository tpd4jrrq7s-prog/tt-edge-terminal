"""Repository Protocol interfaces for the persistence layer.

Cutoff semantics (critical): unless a method name/doc says "at or
before", every cutoff query is **strictly before** — a query for
records "before T" never returns a record whose relevant timestamp is
>= T. `OddsRepository.list_for_match_at_or_before` is the one
documented exception (inclusive), matching real-world "latest known
price" semantics.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

from features.models import FeatureSnapshot
from persistence.models import (
    HistoricalCompetitionRecord,
    HistoricalMatchRecord,
    HistoricalOddsRecord,
    HistoricalPlayerRecord,
    HistoricalRankingRecord,
)


@runtime_checkable
class PlayerRepository(Protocol):
    def add(self, player: HistoricalPlayerRecord) -> None: ...

    def add_many(self, players: Iterable[HistoricalPlayerRecord]) -> None: ...

    def get(self, player_id: str) -> HistoricalPlayerRecord | None: ...

    def has_provider_id(self, provider: str, provider_player_id: str) -> bool: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


@runtime_checkable
class MatchRepository(Protocol):
    def add(self, match: HistoricalMatchRecord) -> None: ...

    def add_many(self, matches: Iterable[HistoricalMatchRecord]) -> None: ...

    def get(self, match_id: str) -> HistoricalMatchRecord | None: ...

    def replace(self, match: HistoricalMatchRecord) -> None:
        """Overwrite an existing record with the same `id` (e.g. scheduled -> completed progression).

        Raises `persistence.errors.RecordNotFoundError` if no record
        with that `id` exists yet — this is an update primitive, not an
        upsert. Used only for safe, non-conflicting merges; see
        `historical_ingestion.deduplication`.
        """
        ...

    def has_provider_id(self, provider: str, provider_match_id: str) -> bool: ...

    def list_player_matches_before(self, player_id: str, cutoff: datetime) -> list[HistoricalMatchRecord]:
        """Return this player's matches with effective_timestamp strictly before cutoff."""
        ...

    def list_head_to_head_before(
        self, player_a_id: str, player_b_id: str, cutoff: datetime
    ) -> list[HistoricalMatchRecord]:
        """Return matches between these two players with effective_timestamp strictly before cutoff."""
        ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


@runtime_checkable
class OddsRepository(Protocol):
    def add(self, odds: HistoricalOddsRecord) -> None: ...

    def add_many(self, odds: Iterable[HistoricalOddsRecord]) -> None: ...

    def get(self, odds_id: str) -> HistoricalOddsRecord | None: ...

    def list_for_match_at_or_before(self, match_id: str, cutoff: datetime) -> list[HistoricalOddsRecord]:
        """Return odds for a match with captured_at <= cutoff (inclusive)."""
        ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


@runtime_checkable
class FeatureSnapshotRepository(Protocol):
    def add(self, snapshot: FeatureSnapshot) -> None: ...

    def get(self, snapshot_id: str) -> FeatureSnapshot | None: ...

    def latest_before(
        self, player_a_id: str, player_b_id: str, cutoff: datetime
    ) -> FeatureSnapshot | None:
        """Return the most recent snapshot for this pair with as_of strictly before cutoff."""
        ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


@runtime_checkable
class RankingRepository(Protocol):
    def add(self, ranking: HistoricalRankingRecord) -> None: ...

    def add_many(self, rankings: Iterable[HistoricalRankingRecord]) -> None: ...

    def get(self, ranking_id: str) -> HistoricalRankingRecord | None: ...

    def latest_before(self, player_id: str, cutoff: datetime) -> HistoricalRankingRecord | None:
        """Return the most recent ranking observation for player_id strictly before cutoff."""
        ...

    def list_for_player_before(self, player_id: str, cutoff: datetime) -> list[HistoricalRankingRecord]:
        """Return this player's ranking observations with effective_at strictly before cutoff."""
        ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


@runtime_checkable
class CompetitionRepository(Protocol):
    def add(self, competition: HistoricalCompetitionRecord) -> None: ...

    def get(self, competition_id: str) -> HistoricalCompetitionRecord | None: ...

    def has_provider_id(self, provider: str, provider_competition_id: str) -> bool: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...
