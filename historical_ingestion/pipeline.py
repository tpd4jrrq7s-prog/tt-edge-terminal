"""Pure, stateless per-record processing functions.

Each function does exactly one conversion/decision step and touches no
repository or other mutable state — `historical_ingestion.service`
orchestrates them, injects dependencies, and owns all mutation. This
split keeps every step here trivially unit-testable in isolation.
"""

from __future__ import annotations

from historical_ingestion.models import (
    ImportedCompetition,
    ImportedMatch,
    ImportedOdds,
    ImportedPlayer,
    ImportedRanking,
    RawRecord,
)
from historical_ingestion.protocols import HistoricalProviderAdapter
from identity.models import IdentityResolution, NormalizedPlayerIdentity, PlayerIdentityRecord
from identity.normalizer import normalize_player_name
from identity.resolver import IdentityResolver
from persistence.models import (
    DataQualityMetadata,
    HistoricalCompetitionRecord,
    HistoricalMatchRecord,
    HistoricalOddsRecord,
    HistoricalPlayerRecord,
    HistoricalRankingRecord,
    HistoricalSetRecord,
    MatchRecordStatus,
)

_STATUS_MAP = {
    "scheduled": MatchRecordStatus.SCHEDULED,
    "live": MatchRecordStatus.LIVE,
    "finished": MatchRecordStatus.FINISHED,
    "retired": MatchRecordStatus.RETIRED,
    "cancelled": MatchRecordStatus.CANCELLED,
}


def adapt_record(
    raw: RawRecord, adapter: HistoricalProviderAdapter
) -> ImportedPlayer | ImportedMatch | ImportedOdds | ImportedCompetition | ImportedRanking:
    """Adapt one raw record via the provider's adapter. May raise a `providers.errors.ProviderError`."""
    return adapter.adapt(raw)


def resolve_player_identity(
    imported_player: ImportedPlayer,
    known_identities: list[PlayerIdentityRecord],
    resolver: IdentityResolver,
) -> IdentityResolution:
    """Resolve one imported player against known identities (external ID, then alias, then similarity)."""
    candidate = NormalizedPlayerIdentity(
        original_name=imported_player.name,
        normalized_name=normalize_player_name(imported_player.name),
        country=imported_player.country,
        external_provider=imported_player.provenance.provider if imported_player.external_player_id else None,
        external_player_id=imported_player.external_player_id,
    )
    return resolver.resolve(candidate, known_identities)


def build_historical_player_record(imported_player: ImportedPlayer, internal_id: str) -> HistoricalPlayerRecord:
    """Convert an accepted `ImportedPlayer` into a persisted `HistoricalPlayerRecord`."""
    return HistoricalPlayerRecord(
        id=internal_id,
        name=imported_player.name,
        country=imported_player.country,
        provider=imported_player.provenance.provider,
        provider_player_id=imported_player.provenance.provider_record_id,
        ingested_at=imported_player.provenance.ingested_at,
    )


def build_historical_match_record(
    imported_match: ImportedMatch, internal_id: str, player_a_id: str, player_b_id: str
) -> HistoricalMatchRecord:
    """Convert an accepted `ImportedMatch` into a persisted `HistoricalMatchRecord`.

    Callers must only call this after validation has accepted the
    record — the strict `persistence.models` validators still apply and
    will raise if something is nonetheless inconsistent.
    """
    sets = [
        HistoricalSetRecord(set_number=s.set_number, player_a_points=s.player_a_points, player_b_points=s.player_b_points)
        for s in imported_match.sets
        if s.set_number is not None and s.player_a_points is not None and s.player_b_points is not None
    ]
    winner_id = None
    if imported_match.winner_external_id is not None:
        winner_id = player_a_id if imported_match.winner_external_id == imported_match.player_a_external_id else player_b_id

    return HistoricalMatchRecord(
        id=internal_id,
        provider=imported_match.provenance.provider,
        provider_match_id=imported_match.provenance.provider_record_id,
        competition_id=imported_match.competition_id,
        competition_name=imported_match.competition_name,
        player_a_id=player_a_id,
        player_b_id=player_b_id,
        scheduled_at=imported_match.scheduled_at,
        actual_start_at=imported_match.actual_start_at,
        completed_at=imported_match.completed_at,
        status=_STATUS_MAP[imported_match.status],
        best_of=imported_match.best_of,
        winner_id=winner_id,
        sets=sets,
        provider_timestamp=imported_match.provenance.source_timestamp,
        ingested_at=imported_match.provenance.ingested_at,
        data_quality=DataQualityMetadata(warnings=list(imported_match.provenance.warnings)),
    )


def build_historical_odds_record(
    imported_odds: ImportedOdds, internal_id: str, internal_match_id: str, internal_selection_id: str
) -> HistoricalOddsRecord:
    """Convert an accepted `ImportedOdds` into a persisted `HistoricalOddsRecord`."""
    return HistoricalOddsRecord(
        id=internal_id,
        match_id=internal_match_id,
        bookmaker=imported_odds.bookmaker,
        selection_id=internal_selection_id,
        decimal_odds=imported_odds.decimal_odds,
        captured_at=imported_odds.captured_at,
        provider=imported_odds.provenance.provider,
        market_id=imported_odds.market_id,
    )


def build_historical_ranking_record(
    imported_ranking: ImportedRanking, internal_id: str, internal_player_id: str
) -> HistoricalRankingRecord:
    """Convert an accepted `ImportedRanking` into a persisted `HistoricalRankingRecord`."""
    return HistoricalRankingRecord(
        id=internal_id,
        player_id=internal_player_id,
        ranking=imported_ranking.ranking,
        ranking_points=imported_ranking.ranking_points,
        effective_at=imported_ranking.effective_at,
        provider=imported_ranking.provenance.provider,
        provider_record_id=imported_ranking.provenance.provider_record_id,
        ingested_at=imported_ranking.provenance.ingested_at,
    )


def build_historical_competition_record(
    imported_competition: ImportedCompetition, internal_id: str
) -> HistoricalCompetitionRecord:
    """Convert an accepted `ImportedCompetition` into a persisted `HistoricalCompetitionRecord`."""
    return HistoricalCompetitionRecord(
        id=internal_id,
        provider=imported_competition.provenance.provider,
        provider_competition_id=imported_competition.provenance.provider_record_id,
        name=imported_competition.name,
        country=imported_competition.country,
        level=imported_competition.level,
        format=imported_competition.format,
        season=imported_competition.season,
        indoor=imported_competition.indoor,
        active_from=imported_competition.active_from,
        active_to=imported_competition.active_to,
        ingested_at=imported_competition.provenance.ingested_at,
    )
