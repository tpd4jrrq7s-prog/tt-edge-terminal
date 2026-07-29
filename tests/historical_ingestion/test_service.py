"""Tests for HistoricalImportService: batches, transactions, dry-run, quarantine, conflicts."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from config.historical import HistoricalIntelligenceSettings
from config.historical_ingestion import HistoricalIngestionSettings
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

IDENTITY_SETTINGS = HistoricalIntelligenceSettings(identity_ambiguity_margin=0.07)


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
        PlayerIdentityRecord(
            id="player:mock:px3", canonical_name="Mao Long", normalized_name="mao long", country="CHN",
            external_ids=[ExternalIdentifier(provider="mock", provider_player_id="px3")],
        ),
    ]


def _build_service(settings=None, checkpoint_store=None, **repo_overrides):
    repos = dict(
        player_repository=InMemoryPlayerRepository(), match_repository=InMemoryMatchRepository(),
        odds_repository=InMemoryOddsRepository(), ranking_repository=InMemoryRankingRepository(),
        competition_repository=InMemoryCompetitionRepository(),
    )
    repos.update(repo_overrides)
    return HistoricalImportService(
        source=MockTableTennisProviderSource(),
        adapter=GenericProviderAdapter(mock_provider_mapping()),
        identity_resolver=IdentityResolver(settings=IDENTITY_SETTINGS),
        checkpoint_store=checkpoint_store or InMemoryCheckpointStore(),
        settings=settings or HistoricalIngestionSettings(),
        known_identities=_seed_identities(),
        **repos,
    ), repos


def test_run_processes_two_batches_and_persists():
    service, repos = _build_service()
    report = service.run()
    assert len(report.batches) == 2
    assert all(b.succeeded for b in report.batches)
    assert repos["match_repository"].count() == 2
    assert repos["odds_repository"].count() == 3
    assert repos["ranking_repository"].count() == 3


def test_exact_duplicate_is_skipped_idempotent():
    service, repos = _build_service()
    service.run()
    # match count should reflect only the two distinct matches, not the re-imported duplicate
    assert repos["match_repository"].count() == 2


def test_conflicting_result_produces_conflict_record_not_persisted():
    service, repos = _build_service()
    service.run()
    conflicts = service.list_conflict_records()
    assert len(conflicts) == 1
    assert conflicts[0].record_type == "match"
    # The original (px1-won) version must still be the one persisted.
    stored = repos["match_repository"].get("mock:mp-1")
    assert stored.winner_id == "player:mock:px1"


def test_ambiguous_identity_is_quarantined_not_persisted():
    service, repos = _build_service()
    service.run()
    quarantined = service.list_quarantine_records()
    assert len(quarantined) == 1
    assert quarantined[0].reason == "ambiguous_identity"
    # Only the 3 unambiguous, pre-seeded identities' players should be persisted.
    assert repos["player_repository"].count() == 3


def test_checkpoint_advances_after_each_successful_batch():
    checkpoint_store = InMemoryCheckpointStore()
    service, _ = _build_service(checkpoint_store=checkpoint_store)
    cursor1, _ = service.process_batch(None)
    checkpoint = checkpoint_store.get("mock-provider-source", "mock")
    assert checkpoint.cursor == cursor1
    cursor2, _ = service.process_batch(cursor1)
    checkpoint = checkpoint_store.get("mock-provider-source", "mock")
    assert checkpoint.cursor == cursor2


def test_dry_run_persists_nothing_and_does_not_advance_checkpoint():
    checkpoint_store = InMemoryCheckpointStore()
    service, repos = _build_service(checkpoint_store=checkpoint_store)
    service.run(dry_run=True)
    assert repos["match_repository"].count() == 0
    assert repos["player_repository"].count() == 0
    assert checkpoint_store.get("mock-provider-source", "mock") is None


def test_dry_run_still_produces_a_full_report():
    service, _ = _build_service()
    report = service.run(dry_run=True)
    assert report.total_records_read == 15
    assert report.dry_run is True


def test_fatal_error_mid_batch_aborts_with_no_partial_persistence():
    service, repos = _build_service()

    class _ExplodingAdapter:
        provider = "mock"
        mapping_version = "1.0.0"
        calls = 0

        def adapt(self, raw):
            _ExplodingAdapter.calls += 1
            if _ExplodingAdapter.calls == 5:
                raise RuntimeError("boom")
            return GenericProviderAdapter(mock_provider_mapping()).adapt(raw)

    service._adapter = _ExplodingAdapter()
    cursor, report = service.process_batch(None)
    assert report.succeeded is False
    assert "boom" in report.failure_reason
    assert repos["player_repository"].count() == 0
    assert repos["match_repository"].count() == 0
    assert cursor is None  # checkpoint cursor unchanged (was None before this batch)


def test_checkpoint_not_advanced_after_fatal_error():
    checkpoint_store = InMemoryCheckpointStore()
    service, _ = _build_service(checkpoint_store=checkpoint_store)

    class _ExplodingAdapter:
        provider = "mock"
        mapping_version = "1.0.0"

        def adapt(self, raw):
            raise RuntimeError("boom")

    service._adapter = _ExplodingAdapter()
    service.process_batch(None)
    assert checkpoint_store.get("mock-provider-source", "mock") is None


def test_repository_fingerprint_changes_after_run():
    service, _ = _build_service()
    report = service.run()
    assert report.repository_fingerprint_before != report.repository_fingerprint_after


def test_run_is_resumable_via_checkpoint():
    checkpoint_store = InMemoryCheckpointStore()
    service, repos = _build_service(checkpoint_store=checkpoint_store)
    service.process_batch(None)
    # A second service instance picks up from the saved checkpoint and finishes the run.
    service2, _ = _build_service(checkpoint_store=checkpoint_store, **repos)
    report = service2.run()
    assert len(report.batches) == 1  # only the remaining batch
    assert repos["match_repository"].count() == 2
