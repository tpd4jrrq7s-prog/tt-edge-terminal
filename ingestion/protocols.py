"""Provider-independent interface that ingestion sources must implement."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ingestion.models import RawMatch, RawOdds


@runtime_checkable
class MatchSource(Protocol):
    """A provider-independent source of raw table tennis match and odds data.

    Any concrete source (mock, scraped, API-backed, etc.) implements this
    interface. `ingestion.service.IngestionService` depends only on this
    protocol, never on a concrete provider.
    """

    name: str

    def fetch_matches(self) -> list[RawMatch]:
        """Return the current set of known raw matches."""
        ...

    def fetch_odds(self, provider_match_id: str) -> list[RawOdds]:
        """Return raw odds quotes for a single provider match id."""
        ...
