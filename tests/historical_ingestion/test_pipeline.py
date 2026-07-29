"""Tests for pure per-record conversion functions."""

from __future__ import annotations

from datetime import datetime, timezone

from historical_ingestion.models import ImportedMatch, ImportedOdds, ImportedPlayer, ImportedSet, ImportProvenance
from historical_ingestion.pipeline import (
    build_historical_match_record,
    build_historical_odds_record,
    build_historical_player_record,
    resolve_player_identity,
)
from identity.models import IdentityOutcome, PlayerIdentityRecord
from identity.resolver import IdentityResolver
from persistence.models import MatchRecordStatus

AWARE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _provenance(**overrides) -> ImportProvenance:
    defaults = dict(
        provider="mock", provider_record_id="r1", source_batch_id="b1",
        source_timestamp=AWARE, ingested_at=AWARE, raw_fingerprint="fp", mapping_version="1.0.0",
    )
    defaults.update(overrides)
    return ImportProvenance(**defaults)


def test_resolve_player_identity_via_exact_external_id():
    known = [
        PlayerIdentityRecord(
            id="p1", canonical_name="Ma Long", normalized_name="ma long",
            external_ids=[{"provider": "mock", "provider_player_id": "px1"}],
        )
    ]
    player = ImportedPlayer(provenance=_provenance(provider_record_id="px1"), name="Ma Long", external_player_id="px1")
    resolution = resolve_player_identity(player, known, IdentityResolver())
    assert resolution.outcome is IdentityOutcome.MATCHED
    assert resolution.identity_id == "p1"


def test_build_historical_player_record():
    player = ImportedPlayer(provenance=_provenance(), name="Ma Long", country="CHN")
    record = build_historical_player_record(player, "internal-1")
    assert record.id == "internal-1"
    assert record.name == "Ma Long"
    assert record.provider == "mock"


def test_build_historical_match_record_maps_winner_correctly():
    match = ImportedMatch(
        provenance=_provenance(),
        player_a_external_id="pa", player_b_external_id="pb",
        scheduled_at=AWARE, actual_start_at=AWARE, completed_at=AWARE,
        status="finished", winner_external_id="pa",
        sets=[ImportedSet(set_number=1, player_a_points=11, player_b_points=7)],
    )
    record = build_historical_match_record(match, "internal-match-1", "internal-pa", "internal-pb")
    assert record.winner_id == "internal-pa"
    assert record.status is MatchRecordStatus.FINISHED
    assert record.sets[0].player_a_points == 11


def test_build_historical_match_record_skips_incomplete_sets():
    match = ImportedMatch(
        provenance=_provenance(),
        player_a_external_id="pa", player_b_external_id="pb",
        scheduled_at=AWARE, status="scheduled",
        sets=[ImportedSet(set_number=None, player_a_points=11, player_b_points=7)],
    )
    record = build_historical_match_record(match, "internal-match-1", "internal-pa", "internal-pb")
    assert record.sets == []


def test_build_historical_odds_record():
    odds = ImportedOdds(
        provenance=_provenance(), provider_match_id="m1", bookmaker="Pinnacle",
        selection_external_id="pa", decimal_odds=1.8, captured_at=AWARE,
    )
    record = build_historical_odds_record(odds, "odds-1", "internal-match-1", "internal-pa")
    assert record.match_id == "internal-match-1"
    assert record.selection_id == "internal-pa"
    assert record.decimal_odds == 1.8
