"""End-to-end integration: provider source -> adapter -> import service -> repositories
-> Phase 3 feature snapshot -> training dataset.
"""

from __future__ import annotations

from config.historical import HistoricalIntelligenceSettings
from datasets.builder import DatasetBuilder
from datasets.splits import chronological_holdout_split
from features.builder import HistoricalFeatureBuilder
from historical_ingestion.checkpoints import InMemoryCheckpointStore
from historical_ingestion.service import HistoricalImportService
from historical_ingestion.sources.mock_provider import MockTableTennisProviderSource
from identity.models import ExternalIdentifier, PlayerIdentityRecord
from identity.resolver import IdentityResolver
from persistence.in_memory import (
    InMemoryCompetitionRepository,
    InMemoryMatchRepository,
    InMemoryOddsRepository,
    InMemoryPlayerRepository,
    InMemoryRankingRepository,
)
from providers.generic.adapter import GenericProviderAdapter
from providers.generic.mappings import mock_provider_mapping


def _seed_identities() -> list[PlayerIdentityRecord]:
    return [
        PlayerIdentityRecord(
            id="player:mock:px1", canonical_name="Ma Long", normalized_name="ma long", country="CHN",
            external_ids=[ExternalIdentifier(provider="mock", provider_player_id="px1")],
        ),
        PlayerIdentityRecord(
            id="player:mock:px2", canonical_name="Fan Zhendong", normalized_name="fan zhendong", country="CHN",
            external_ids=[ExternalIdentifier(provider="mock", provider_player_id="px2")],
        ),
    ]


def test_full_pipeline_source_to_dataset():
    player_repo = InMemoryPlayerRepository()
    match_repo = InMemoryMatchRepository()
    odds_repo = InMemoryOddsRepository()
    ranking_repo = InMemoryRankingRepository()
    competition_repo = InMemoryCompetitionRepository()

    service = HistoricalImportService(
        source=MockTableTennisProviderSource(),
        adapter=GenericProviderAdapter(mock_provider_mapping()),
        identity_resolver=IdentityResolver(settings=HistoricalIntelligenceSettings(identity_ambiguity_margin=0.07)),
        player_repository=player_repo, match_repository=match_repo, odds_repository=odds_repo,
        ranking_repository=ranking_repo, competition_repository=competition_repo,
        checkpoint_store=InMemoryCheckpointStore(), known_identities=_seed_identities(),
    )

    # 1. Source -> adapter -> service -> repositories.
    report = service.run()
    assert report.total_inserted > 0
    assert match_repo.count() == 2
    assert odds_repo.count() == 3
    assert ranking_repo.count() == 3

    # 2. Repositories -> leakage-safe feature snapshot.
    feature_builder = HistoricalFeatureBuilder(match_repo, ranking_repository=ranking_repo)
    target = match_repo.get("mock:mp-2")
    snapshot = feature_builder.build(target.id, target.scheduled_at)
    assert snapshot.provenance.player_a_observation_count == 1
    assert snapshot.provenance.head_to_head_observation_count == 1
    assert snapshot.target_match_id not in snapshot.provenance.player_a_source_match_ids

    # 3. Snapshot -> training dataset.
    dataset_builder = DatasetBuilder(match_repo, feature_builder)
    match_ids = ["mock:mp-1", "mock:mp-2"]
    examples, manifest = dataset_builder.build(match_ids, dataset_id="integration-test")
    assert manifest.training_example_count == 2
    assert manifest.skipped_count == 0
    for example in examples:
        assert example.player_a_won in (0, 1)
        assert example.features.target_match_id == example.target_match_id


def test_full_pipeline_is_deterministic_across_runs():
    def _run_once():
        match_repo = InMemoryMatchRepository()
        service = HistoricalImportService(
            source=MockTableTennisProviderSource(),
            adapter=GenericProviderAdapter(mock_provider_mapping()),
            identity_resolver=IdentityResolver(settings=HistoricalIntelligenceSettings(identity_ambiguity_margin=0.07)),
            player_repository=InMemoryPlayerRepository(), match_repository=match_repo,
            odds_repository=InMemoryOddsRepository(), ranking_repository=InMemoryRankingRepository(),
            competition_repository=InMemoryCompetitionRepository(), checkpoint_store=InMemoryCheckpointStore(),
            known_identities=_seed_identities(),
        )
        service.run()
        return match_repo.count(), match_repo.get("mock:mp-1").winner_id

    assert _run_once() == _run_once()
