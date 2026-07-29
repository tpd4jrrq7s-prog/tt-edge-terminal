"""A deterministic, in-memory mock provider source — illustrative data only.

Deliberately includes, across two batches: valid records, an exact
duplicate re-import, an ambiguous player identity, a conflicting match
result, odds history (multiple observations), and ranking history —
everything `python -m historical_ingestion.demo` needs to demonstrate.
Makes no network or file-system calls.
"""

from __future__ import annotations

from datetime import datetime, timezone

from historical_ingestion.models import RawRecord, SourceBatch, SourceHealth
from historical_ingestion.sources.base import compute_batch_checksum

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class MockTableTennisProviderSource:
    """A fixed, deterministic sequence of illustrative raw records from the 'mock' provider."""

    provider = "mock"
    version = "1.0.0"

    def __init__(self, batch_size: int = 8, source_name: str = "mock-provider-source") -> None:
        self.name = source_name
        self._batch_size = batch_size
        self._records = self._build_records()

    def _build_records(self) -> list[RawRecord]:
        now = datetime.now(timezone.utc)

        def rec(record_id: str, record_type: str, payload: dict, source_ts: datetime) -> RawRecord:
            return RawRecord(
                provider=self.provider, provider_record_id=record_id, record_type=record_type,
                payload=payload, source_batch_id="mock-source", source_timestamp=source_ts, fetched_at=now,
            )

        records: list[RawRecord] = [
            rec("px1", "player", {"name": "Ma Long", "country": "CHN", "ranking_hint": 2}, _BASE),
            rec("px2", "player", {"name": "Fan Zhendong", "country": "CHN", "ranking_hint": 5}, _BASE),
            rec(
                "demo-open", "competition",
                {
                    "name": "Demo Open", "country": "CHN", "level": "tier-1", "format": "singles",
                    "season": "2026", "indoor": True, "active_from": _iso(_BASE), "active_to": _iso(_BASE.replace(day=15)),
                },
                _BASE,
            ),
            rec(
                "mp-1", "match",
                {
                    "competition_id": "demo-open", "competition_name": "Demo Open",
                    "player_a_id": "px1", "player_b_id": "px2",
                    "scheduled_at": _iso(_BASE), "actual_start_at": _iso(_BASE),
                    "completed_at": _iso(_BASE.replace(hour=1)),
                    "status": "COMPLETED", "best_of": 5, "winner_id": "px1",
                    "sets": [
                        {"set_number": 1, "a": 11, "b": 7},
                        {"set_number": 2, "a": 9, "b": 11},
                        {"set_number": 3, "a": 11, "b": 8},
                    ],
                },
                _BASE.replace(hour=1),
            ),
            rec(
                "odds-1", "odds",
                {"match_id": "mp-1", "bookmaker": "MockBook", "selection_id": "px1", "decimal_odds": 1.8,
                 "captured_at": _iso(_BASE.replace(minute=0)), "market_id": "winner"},
                _BASE.replace(minute=0),
            ),
            rec(
                "odds-2", "odds",
                {"match_id": "mp-1", "bookmaker": "MockBook", "selection_id": "px1", "decimal_odds": 1.75,
                 "captured_at": _iso(_BASE.replace(minute=30)), "market_id": "winner"},
                _BASE.replace(minute=30),
            ),
            rec(
                "rank-1", "ranking",
                {"player_id": "px1", "ranking": 2, "ranking_points": 6500.0, "effective_at": _iso(_BASE)},
                _BASE,
            ),
            rec(
                "rank-2", "ranking",
                {"player_id": "px2", "ranking": 5, "ranking_points": 6000.0, "effective_at": _iso(_BASE)},
                _BASE,
            ),
            # A second, easily-confused player identity — deliberately similar to "Ma Long".
            # The demo pre-seeds this identity's known record (with its px3 external ID) so this
            # record resolves via exact external-ID match rather than emergent fuzzy matching,
            # which would otherwise risk incorrectly auto-merging it into "Ma Long".
            rec("px3", "player", {"name": "Mao Long", "country": "CHN"}, _BASE),
            # --- batch boundary (with default batch_size=8) ---
            # Exact re-import of mp-1, identical content: should be skipped as idempotent.
            rec(
                "mp-1", "match",
                {
                    "competition_id": "demo-open", "competition_name": "Demo Open",
                    "player_a_id": "px1", "player_b_id": "px2",
                    "scheduled_at": _iso(_BASE), "actual_start_at": _iso(_BASE),
                    "completed_at": _iso(_BASE.replace(hour=1)),
                    "status": "COMPLETED", "best_of": 5, "winner_id": "px1",
                    "sets": [
                        {"set_number": 1, "a": 11, "b": 7},
                        {"set_number": 2, "a": 9, "b": 11},
                        {"set_number": 3, "a": 11, "b": 8},
                    ],
                },
                _BASE.replace(hour=1),
            ),
            # Conflicting re-import of mp-1: same provider match id, different winner.
            rec(
                "mp-1", "match",
                {
                    "competition_id": "demo-open", "competition_name": "Demo Open",
                    "player_a_id": "px1", "player_b_id": "px2",
                    "scheduled_at": _iso(_BASE), "actual_start_at": _iso(_BASE),
                    "completed_at": _iso(_BASE.replace(hour=1)),
                    "status": "COMPLETED", "best_of": 5, "winner_id": "px2",
                    "sets": [
                        {"set_number": 1, "a": 7, "b": 11},
                        {"set_number": 2, "a": 11, "b": 9},
                        {"set_number": 3, "a": 8, "b": 11},
                    ],
                },
                _BASE.replace(hour=1, minute=5),
            ),
            # Ambiguous identity: no external id, name close to both "Ma Long" and "Mao Long"
            # (genuinely ambiguous when the resolver's ambiguity margin is >= ~0.06 — see
            # historical_ingestion.demo, which configures identity resolution accordingly).
            rec("px-ambiguous", "player", {"name": "Cma Long", "country": "CHN"}, _BASE.replace(day=5)),
            # A second, distinct completed match — normal insert.
            rec(
                "mp-2", "match",
                {
                    "competition_id": "demo-open", "competition_name": "Demo Open",
                    "player_a_id": "px1", "player_b_id": "px2",
                    "scheduled_at": _iso(_BASE.replace(day=10)), "actual_start_at": _iso(_BASE.replace(day=10)),
                    "completed_at": _iso(_BASE.replace(day=10, hour=1)),
                    "status": "COMPLETED", "best_of": 5, "winner_id": "px2",
                    "sets": [
                        {"set_number": 1, "a": 8, "b": 11},
                        {"set_number": 2, "a": 11, "b": 9},
                        {"set_number": 3, "a": 6, "b": 11},
                    ],
                },
                _BASE.replace(day=10, hour=1),
            ),
            rec(
                "odds-3", "odds",
                {"match_id": "mp-2", "bookmaker": "MockBook", "selection_id": "px1", "decimal_odds": 1.9,
                 "captured_at": _iso(_BASE.replace(day=10, minute=0)), "market_id": "winner"},
                _BASE.replace(day=10, minute=0),
            ),
            rec(
                "rank-3", "ranking",
                {"player_id": "px1", "ranking": 3, "ranking_points": 6400.0, "effective_at": _iso(_BASE.replace(day=10))},
                _BASE.replace(day=10),
            ),
        ]
        return records

    def fetch_batch(self, cursor: str | None) -> SourceBatch:
        offset = int(cursor) if cursor else 0
        window = self._records[offset : offset + self._batch_size]
        next_offset = offset + len(window)
        next_cursor = str(next_offset) if next_offset < len(self._records) else None
        now = datetime.now(timezone.utc)

        return SourceBatch(
            source_name=self.name,
            provider=self.provider,
            source_version=self.version,
            batch_id=f"{self.name}:{offset}",
            records=window,
            next_cursor=next_cursor,
            source_timestamp=now,
            fetched_at=now,
            source_metadata={"row_offset": str(offset)},
            checksum=compute_batch_checksum(window),
        )

    def health(self) -> SourceHealth:
        return SourceHealth(
            source_name=self.name, provider=self.provider, healthy=True,
            detail="deterministic in-memory mock source", checked_at=datetime.now(timezone.utc),
        )
