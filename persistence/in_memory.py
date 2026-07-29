"""Deterministic, in-memory repository implementations for development and tests.

No database. No global mutable state — every repository instance owns
its own dictionaries. Insertion order is preserved; iteration methods
apply deterministic secondary sorting so results never depend on dict
iteration quirks. Every getter returns a deep copy so callers can never
mutate stored state through the returned object.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from features.models import FeatureSnapshot
from persistence.errors import DuplicateRecordError, ProviderMappingConflictError
from persistence.models import HistoricalMatchRecord, HistoricalOddsRecord, HistoricalPlayerRecord


class InMemoryPlayerRepository:
    """In-memory `PlayerRepository` implementation."""

    def __init__(self) -> None:
        self._by_id: dict[str, HistoricalPlayerRecord] = {}
        self._provider_index: dict[tuple[str, str], str] = {}

    def add(self, player: HistoricalPlayerRecord) -> None:
        if player.id in self._by_id:
            raise DuplicateRecordError(f"Player {player.id!r} already exists")
        key = (player.provider, player.provider_player_id)
        existing = self._provider_index.get(key)
        if existing is not None and existing != player.id:
            raise ProviderMappingConflictError(
                f"Provider mapping {key} is already assigned to internal ID {existing!r}"
            )
        self._by_id[player.id] = player.model_copy(deep=True)
        self._provider_index[key] = player.id

    def add_many(self, players: Iterable[HistoricalPlayerRecord]) -> None:
        for player in players:
            self.add(player)

    def get(self, player_id: str) -> HistoricalPlayerRecord | None:
        record = self._by_id.get(player_id)
        return record.model_copy(deep=True) if record is not None else None

    def has_provider_id(self, provider: str, provider_player_id: str) -> bool:
        return (provider, provider_player_id) in self._provider_index

    def count(self) -> int:
        return len(self._by_id)

    def clear(self) -> None:
        self._by_id.clear()
        self._provider_index.clear()


class InMemoryMatchRepository:
    """In-memory `MatchRepository` implementation."""

    def __init__(self) -> None:
        self._by_id: dict[str, HistoricalMatchRecord] = {}
        self._provider_index: dict[tuple[str, str], str] = {}

    def add(self, match: HistoricalMatchRecord) -> None:
        if match.id in self._by_id:
            raise DuplicateRecordError(f"Match {match.id!r} already exists")
        key = (match.provider, match.provider_match_id)
        existing = self._provider_index.get(key)
        if existing is not None and existing != match.id:
            raise ProviderMappingConflictError(
                f"Provider mapping {key} is already assigned to internal ID {existing!r}"
            )
        self._by_id[match.id] = match.model_copy(deep=True)
        self._provider_index[key] = match.id

    def add_many(self, matches: Iterable[HistoricalMatchRecord]) -> None:
        for match in matches:
            self.add(match)

    def get(self, match_id: str) -> HistoricalMatchRecord | None:
        record = self._by_id.get(match_id)
        return record.model_copy(deep=True) if record is not None else None

    def has_provider_id(self, provider: str, provider_match_id: str) -> bool:
        return (provider, provider_match_id) in self._provider_index

    def _sorted(self, matches: list[HistoricalMatchRecord]) -> list[HistoricalMatchRecord]:
        return sorted(matches, key=lambda m: (m.effective_timestamp, m.id))

    def list_player_matches_before(self, player_id: str, cutoff: datetime) -> list[HistoricalMatchRecord]:
        matches = [
            m
            for m in self._by_id.values()
            if player_id in (m.player_a_id, m.player_b_id) and m.effective_timestamp < cutoff
        ]
        return [m.model_copy(deep=True) for m in self._sorted(matches)]

    def list_head_to_head_before(
        self, player_a_id: str, player_b_id: str, cutoff: datetime
    ) -> list[HistoricalMatchRecord]:
        pair = {player_a_id, player_b_id}
        matches = [
            m
            for m in self._by_id.values()
            if {m.player_a_id, m.player_b_id} == pair and m.effective_timestamp < cutoff
        ]
        return [m.model_copy(deep=True) for m in self._sorted(matches)]

    def count(self) -> int:
        return len(self._by_id)

    def clear(self) -> None:
        self._by_id.clear()
        self._provider_index.clear()


class InMemoryOddsRepository:
    """In-memory `OddsRepository` implementation."""

    def __init__(self) -> None:
        self._by_id: dict[str, HistoricalOddsRecord] = {}
        self._dedup_index: set[tuple[str, str, str, datetime]] = set()

    def add(self, odds: HistoricalOddsRecord) -> None:
        if odds.id in self._by_id:
            raise DuplicateRecordError(f"Odds observation {odds.id!r} already exists")
        dedup_key = (odds.match_id, odds.bookmaker, odds.selection_id, odds.captured_at)
        if dedup_key in self._dedup_index:
            raise DuplicateRecordError(
                f"Duplicate odds observation for match={odds.match_id!r} "
                f"bookmaker={odds.bookmaker!r} selection={odds.selection_id!r} "
                f"captured_at={odds.captured_at.isoformat()!r}"
            )
        self._by_id[odds.id] = odds.model_copy(deep=True)
        self._dedup_index.add(dedup_key)

    def add_many(self, odds: Iterable[HistoricalOddsRecord]) -> None:
        for o in odds:
            self.add(o)

    def get(self, odds_id: str) -> HistoricalOddsRecord | None:
        record = self._by_id.get(odds_id)
        return record.model_copy(deep=True) if record is not None else None

    def list_for_match_at_or_before(self, match_id: str, cutoff: datetime) -> list[HistoricalOddsRecord]:
        records = [o for o in self._by_id.values() if o.match_id == match_id and o.captured_at <= cutoff]
        ordered = sorted(records, key=lambda o: (o.captured_at, o.bookmaker, o.id))
        return [o.model_copy(deep=True) for o in ordered]

    def count(self) -> int:
        return len(self._by_id)

    def clear(self) -> None:
        self._by_id.clear()
        self._dedup_index.clear()


class InMemoryFeatureSnapshotRepository:
    """In-memory `FeatureSnapshotRepository` implementation."""

    def __init__(self) -> None:
        self._by_id: dict[str, FeatureSnapshot] = {}

    def add(self, snapshot: FeatureSnapshot) -> None:
        if snapshot.id in self._by_id:
            raise DuplicateRecordError(f"Feature snapshot {snapshot.id!r} already exists")
        self._by_id[snapshot.id] = snapshot.model_copy(deep=True)

    def get(self, snapshot_id: str) -> FeatureSnapshot | None:
        record = self._by_id.get(snapshot_id)
        return record.model_copy(deep=True) if record is not None else None

    def latest_before(
        self, player_a_id: str, player_b_id: str, cutoff: datetime
    ) -> FeatureSnapshot | None:
        pair = {player_a_id, player_b_id}
        candidates = [
            s
            for s in self._by_id.values()
            if {s.player_a_id, s.player_b_id} == pair and s.as_of < cutoff
        ]
        if not candidates:
            return None
        latest = sorted(candidates, key=lambda s: (s.as_of, s.id))[-1]
        return latest.model_copy(deep=True)

    def count(self) -> int:
        return len(self._by_id)

    def clear(self) -> None:
        self._by_id.clear()
