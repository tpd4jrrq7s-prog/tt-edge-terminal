"""Ingestion service: coordinates fetching from a source and normalizing results."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from domain.match import Match
from domain.odds import Odds
from ingestion.protocols import MatchSource
from normalization.match_normalizer import normalize_match
from normalization.odds_normalizer import normalize_odds

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Typed, normalized output of a single ingestion pass."""

    matches: list[Match] = field(default_factory=list)
    odds: list[Odds] = field(default_factory=list)


class IngestionService:
    """Fetches raw data from any `MatchSource` and normalizes it into domain models.

    The source is injected rather than constructed internally, so any
    implementation of `MatchSource` (mock, future real providers, etc.)
    can be used without changing this class.
    """

    def __init__(self, source: MatchSource) -> None:
        self._source = source

    def run_once(self) -> IngestionResult:
        """Fetch and normalize matches and their odds in a single pass."""
        logger.info("ingestion.run_once.start source=%s", self._source.name)

        matches: list[Match] = []
        odds: list[Odds] = []

        for raw_match in self._source.fetch_matches():
            match = normalize_match(raw_match)
            matches.append(match)

            for raw_odds in self._source.fetch_odds(raw_match.provider_match_id):
                odds.append(normalize_odds(raw_odds))

        logger.info(
            "ingestion.run_once.complete source=%s matches=%d odds=%d",
            self._source.name,
            len(matches),
            len(odds),
        )
        return IngestionResult(matches=matches, odds=odds)
