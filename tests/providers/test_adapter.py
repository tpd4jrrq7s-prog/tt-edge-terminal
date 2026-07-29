"""Tests for the generic, configurable provider adapter."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from historical_ingestion.models import RawRecord
from providers.errors import ProviderMappingError, UnknownProviderStatusError
from providers.generic.adapter import GenericProviderAdapter, raw_fingerprint
from providers.generic.mappings import mock_provider_mapping

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _raw(record_type: str, payload: dict, provider="mock", record_id="r1") -> RawRecord:
    return RawRecord(
        provider=provider, provider_record_id=record_id, record_type=record_type, payload=payload,
        source_batch_id="b1", source_timestamp=NOW, fetched_at=NOW,
    )


def test_adapt_match_maps_all_scalar_fields():
    adapter = GenericProviderAdapter(mock_provider_mapping())
    raw = _raw("match", {
        "competition_id": "demo-open", "competition_name": "Demo Open",
        "player_a_id": "px1", "player_b_id": "px2",
        "scheduled_at": "2026-01-01T00:00:00+00:00", "actual_start_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T01:00:00+00:00", "status": "COMPLETED", "best_of": 5, "winner_id": "px1",
        "sets": [{"set_number": 1, "a": 11, "b": 7}],
    })
    match = adapter.adapt(raw)
    assert match.player_a_external_id == "px1"
    assert match.status == "finished"
    assert match.status_raw == "COMPLETED"
    assert len(match.sets) == 1
    assert match.sets[0].player_a_points == 11


def test_adapt_match_leaves_missing_fields_as_none_not_fabricated():
    adapter = GenericProviderAdapter(mock_provider_mapping())
    raw = _raw("match", {"status": "SCHEDULED"})
    match = adapter.adapt(raw)
    assert match.player_a_external_id is None
    assert match.scheduled_at is None


def test_unknown_status_warns_by_default():
    adapter = GenericProviderAdapter(mock_provider_mapping())
    raw = _raw("match", {"status": "SOMETHING_WEIRD"})
    match = adapter.adapt(raw)
    assert match.status is None
    assert match.status_raw == "SOMETHING_WEIRD"
    assert any("unknown status" in w for w in match.provenance.warnings)


def test_unknown_status_rejects_when_configured():
    config = mock_provider_mapping()
    config = config.model_copy(update={"unknown_status_policy": "reject"})
    adapter = GenericProviderAdapter(config)
    raw = _raw("match", {"status": "SOMETHING_WEIRD"})
    with pytest.raises(UnknownProviderStatusError):
        adapter.adapt(raw)


def test_adapt_odds_maps_fields():
    adapter = GenericProviderAdapter(mock_provider_mapping())
    raw = _raw("odds", {
        "match_id": "mp-1", "bookmaker": "MockBook", "selection_id": "px1",
        "decimal_odds": 1.8, "captured_at": "2026-01-01T00:00:00+00:00", "market_id": "winner",
    })
    odds = adapter.adapt(raw)
    assert odds.provider_match_id == "mp-1"
    assert odds.decimal_odds == 1.8
    assert odds.captured_at is not None


def test_adapt_player_sets_external_id_from_provenance():
    adapter = GenericProviderAdapter(mock_provider_mapping())
    raw = _raw("player", {"name": "Ma Long", "country": "CHN"}, record_id="px1")
    player = adapter.adapt(raw)
    assert player.name == "Ma Long"
    assert player.external_player_id == "px1"


def test_adapt_ranking_and_competition():
    adapter = GenericProviderAdapter(mock_provider_mapping())
    ranking = adapter.adapt(_raw("ranking", {"player_id": "px1", "ranking": 2, "ranking_points": 6500.0, "effective_at": "2026-01-01T00:00:00+00:00"}))
    assert ranking.ranking == 2
    competition = adapter.adapt(_raw("competition", {"name": "Demo Open", "country": "CHN"}))
    assert competition.name == "Demo Open"


def test_adapter_rejects_mismatched_provider():
    adapter = GenericProviderAdapter(mock_provider_mapping())
    raw = _raw("match", {}, provider="other-provider")
    with pytest.raises(ProviderMappingError):
        adapter.adapt(raw)


def test_raw_fingerprint_is_deterministic():
    payload = {"b": 2, "a": 1}
    assert raw_fingerprint(payload) == raw_fingerprint({"a": 1, "b": 2})


def test_raw_fingerprint_differs_for_different_payloads():
    assert raw_fingerprint({"a": 1}) != raw_fingerprint({"a": 2})


def test_malformed_set_entries_are_skipped_with_warning():
    adapter = GenericProviderAdapter(mock_provider_mapping())
    raw = _raw("match", {"status": "SCHEDULED", "sets": ["not-a-dict"]})
    match = adapter.adapt(raw)
    assert match.sets == []
    assert any("non-dict set entry" in w for w in match.provenance.warnings)
