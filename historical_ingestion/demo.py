"""Read-only demonstration of the Phase 4 historical ingestion platform.

Run with:
    python -m historical_ingestion.demo

All data is deterministic and illustrative/mock (see
`historical_ingestion.sources.mock_provider`) — not real match results.
Makes no network calls and never suggests or places a wager.
"""

from __future__ import annotations

from config.historical import HistoricalIntelligenceSettings
from config.historical_ingestion import HistoricalIngestionSettings
from datasets.builder import DatasetBuilder
from datasets.splits import chronological_holdout_split
from features.builder import HistoricalFeatureBuilder
from historical_ingestion.checkpoints import InMemoryCheckpointStore
from historical_ingestion.reports import format_report_text
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

# A slightly wider ambiguity margin than the Phase 3 default so the deliberately close
# "Cma Long" vs "Ma Long"/"Mao Long" demo case actually lands as ambiguous, not auto-matched.
_IDENTITY_SETTINGS = HistoricalIntelligenceSettings(identity_ambiguity_margin=0.07)


def _seed_known_identities() -> list[PlayerIdentityRecord]:
    """Pre-seed already-known identities with real external IDs.

    This lets px1/px2/px3 resolve via exact external-ID match (never
    fuzzy), so the *only* fuzzy/ambiguous resolution in this demo is the
    deliberately close, external-id-less "Cma Long" player later on.
    """
    return [
        PlayerIdentityRecord(
            id="player:mock:px1", canonical_name="Ma Long", normalized_name="ma long",
            country="CHN", external_ids=[ExternalIdentifier(provider="mock", provider_player_id="px1")],
        ),
        PlayerIdentityRecord(
            id="player:mock:px2", canonical_name="Fan Zhendong", normalized_name="fan zhendong",
            country="CHN", external_ids=[ExternalIdentifier(provider="mock", provider_player_id="px2")],
        ),
        PlayerIdentityRecord(
            id="player:mock:px3", canonical_name="Mao Long", normalized_name="mao long",
            country="CHN", external_ids=[ExternalIdentifier(provider="mock", provider_player_id="px3")],
        ),
    ]


def _build_service(
    settings: HistoricalIngestionSettings,
    checkpoint_store: InMemoryCheckpointStore,
    player_repo, match_repo, odds_repo, ranking_repo, competition_repo,
) -> HistoricalImportService:
    return HistoricalImportService(
        source=MockTableTennisProviderSource(),
        adapter=GenericProviderAdapter(mock_provider_mapping()),
        identity_resolver=IdentityResolver(settings=_IDENTITY_SETTINGS),
        player_repository=player_repo,
        match_repository=match_repo,
        odds_repository=odds_repo,
        ranking_repository=ranking_repo,
        competition_repository=competition_repo,
        checkpoint_store=checkpoint_store,
        settings=settings,
        known_identities=_seed_known_identities(),
    )


def main() -> None:
    settings = HistoricalIngestionSettings()

    print("=" * 70)
    print("TT Edge Terminal — Historical Ingestion Demo (illustrative mock data)")
    print("=" * 70)

    # --- Step-by-step batch processing + checkpointing -----------------------------
    player_repo = InMemoryPlayerRepository()
    match_repo = InMemoryMatchRepository()
    odds_repo = InMemoryOddsRepository()
    ranking_repo = InMemoryRankingRepository()
    competition_repo = InMemoryCompetitionRepository()
    checkpoint_store = InMemoryCheckpointStore(checkpoint_version=settings.checkpoint_version)
    service = _build_service(settings, checkpoint_store, player_repo, match_repo, odds_repo, ranking_repo, competition_repo)

    print("\n-- Batch 1 --")
    cursor_after_1, batch1_report = service.process_batch(None)
    print(f"  read={batch1_report.records_read} accepted={batch1_report.records_accepted} "
          f"inserted={batch1_report.records_inserted} skipped={batch1_report.records_skipped} "
          f"quarantined={batch1_report.records_quarantined} rejected={batch1_report.records_rejected}")
    print(f"  checkpoint cursor advanced to: {cursor_after_1!r}")

    print("\n-- Batch 2 --")
    cursor_after_2, batch2_report = service.process_batch(cursor_after_1)
    print(f"  read={batch2_report.records_read} accepted={batch2_report.records_accepted} "
          f"inserted={batch2_report.records_inserted} skipped={batch2_report.records_skipped} "
          f"quarantined={batch2_report.records_quarantined} rejected={batch2_report.records_rejected}")
    print(f"  checkpoint cursor advanced to: {cursor_after_2!r} (None = source fully consumed)")

    print("\n-- Quarantine records --")
    for q in service.list_quarantine_records():
        print(f"  - [{q.reason}] {q.record_type} {q.provider_record_id}: {q.issues}")
    if not service.list_quarantine_records():
        print("  (none)")

    print("\n-- Conflict records --")
    for c in service.list_conflict_records():
        print(f"  - {c.record_type} {c.provider_record_id}: conflicting field={c.conflicting_field!r}")
    if not service.list_conflict_records():
        print("  (none)")

    print("\n-- Repository counts --")
    print(f"  players={player_repo.count()} matches={match_repo.count()} odds={odds_repo.count()} "
          f"rankings={ranking_repo.count()} competitions={competition_repo.count()}")

    # --- A full run() over a fresh checkpoint, for one clean cumulative report -----
    print("\n-- Full run() over a fresh repository/checkpoint --")
    fresh_report = _build_service(
        settings, InMemoryCheckpointStore(checkpoint_version=settings.checkpoint_version),
        InMemoryPlayerRepository(), InMemoryMatchRepository(), InMemoryOddsRepository(),
        InMemoryRankingRepository(), InMemoryCompetitionRepository(),
    ).run()
    print(format_report_text(fresh_report))

    # --- Integration: build a Phase 3 feature snapshot from the imported data ------
    print("\n-- Building a Phase 3 feature snapshot from imported data --")
    feature_builder = HistoricalFeatureBuilder(match_repo, ranking_repository=ranking_repo)
    target_match = match_repo.get("mock:mp-2")
    snapshot = feature_builder.build(target_match.id, target_match.scheduled_at)
    print(f"  snapshot as_of={snapshot.as_of.isoformat()} "
          f"player_a_obs={snapshot.provenance.player_a_observation_count} "
          f"player_b_obs={snapshot.provenance.player_b_observation_count} "
          f"h2h={snapshot.provenance.head_to_head_observation_count} "
          f"data_quality={snapshot.provenance.data_quality_score:.1f}/100")

    print("\n-- Building a small chronological training dataset --")
    dataset_builder = DatasetBuilder(match_repo, feature_builder)
    match_ids = [m.id for m in [match_repo.get("mock:mp-1"), match_repo.get("mock:mp-2")] if m is not None]
    examples, manifest = dataset_builder.build(match_ids, dataset_id="ingestion-demo-dataset")
    print(f"  training examples: {manifest.training_example_count} (skipped: {manifest.skipped_count})")
    if len(examples) >= 3:
        plan = chronological_holdout_split(examples)
        for split in plan.splits:
            print(f"  split {split.label}: {split.size} example(s)")
    else:
        print("  (too few examples for a chronological holdout split in this small demo)")

    print(
        "\nAll data above is illustrative/mock. This demo performs no external calls, trains no "
        "ML model, and never suggests or places a wager."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
