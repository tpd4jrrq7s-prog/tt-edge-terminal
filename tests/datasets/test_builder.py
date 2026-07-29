"""Tests for DatasetBuilder: converting completed matches into training examples."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config.historical import HistoricalIntelligenceSettings
from datasets.builder import DatasetBuilder
from features.builder import HistoricalFeatureBuilder
from persistence.in_memory import InMemoryMatchRepository
from persistence.models import HistoricalMatchRecord, HistoricalSetRecord, MatchRecordStatus

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _finished(mid, day, winner) -> HistoricalMatchRecord:
    scheduled = BASE + timedelta(days=day)
    return HistoricalMatchRecord(
        id=mid, provider="mock", provider_match_id=mid, player_a_id="p1", player_b_id="p2",
        scheduled_at=scheduled, actual_start_at=scheduled, completed_at=scheduled + timedelta(hours=1),
        status=MatchRecordStatus.FINISHED, winner_id=winner,
        sets=[HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)]
        if winner == "p1" else [HistoricalSetRecord(set_number=1, player_a_points=7, player_b_points=11)],
        provider_timestamp=scheduled, ingested_at=scheduled,
    )


def _scheduled(mid, day) -> HistoricalMatchRecord:
    scheduled = BASE + timedelta(days=day)
    return HistoricalMatchRecord(
        id=mid, provider="mock", provider_match_id=mid, player_a_id="p1", player_b_id="p2",
        scheduled_at=scheduled, status=MatchRecordStatus.SCHEDULED,
        provider_timestamp=scheduled, ingested_at=scheduled,
    )


def _setup(n=6):
    repo = InMemoryMatchRepository()
    matches = [_finished(f"m{i}", i, winner=("p1" if i % 2 == 0 else "p2")) for i in range(n)]
    repo.add_many(matches)
    feature_builder = HistoricalFeatureBuilder(repo)
    dataset_builder = DatasetBuilder(repo, feature_builder)
    return repo, matches, dataset_builder


def test_builds_one_example_per_completed_match():
    _, matches, builder = _setup(6)
    examples, manifest = builder.build([m.id for m in matches], dataset_id="ds1")
    assert manifest.training_example_count == 6
    assert manifest.skipped_count == 0
    assert len(examples) == 6


def test_valid_labels_match_recorded_winner():
    _, matches, builder = _setup(4)
    examples, _ = builder.build([m.id for m in matches])
    by_id = {e.target_match_id: e for e in examples}
    assert by_id["m0"].player_a_won == 1  # m0 winner is p1 == player_a
    assert by_id["m1"].player_a_won == 0  # m1 winner is p2


def test_examples_are_in_chronological_order():
    _, matches, builder = _setup(6)
    ids_shuffled = [m.id for m in matches][::-1]
    examples, _ = builder.build(ids_shuffled)
    as_of_values = [e.as_of for e in examples]
    assert as_of_values == sorted(as_of_values)


def test_scheduled_match_is_skipped_not_labeled():
    repo = InMemoryMatchRepository()
    repo.add(_finished("m0", 0, winner="p1"))
    repo.add(_scheduled("m1", 5))
    feature_builder = HistoricalFeatureBuilder(repo)
    builder = DatasetBuilder(repo, feature_builder)
    examples, manifest = builder.build(["m0", "m1"])
    assert manifest.training_example_count == 1
    assert manifest.skipped_count == 1
    assert any("m1" in w for w in manifest.warnings)


def test_missing_match_id_is_skipped_with_warning():
    _, matches, builder = _setup(2)
    examples, manifest = builder.build([*[m.id for m in matches], "does-not-exist"])
    assert manifest.skipped_count == 1
    assert any("does-not-exist" in w for w in manifest.warnings)


def test_target_fields_are_excluded_from_snapshot_features():
    _, matches, builder = _setup(3)
    examples, _ = builder.build([m.id for m in matches])
    for example in examples:
        dumped = example.features.model_dump()
        assert "winner_id" not in dumped
        assert "player_a_won" not in dumped


def test_manifest_label_distribution_and_date_range():
    _, matches, builder = _setup(6)
    examples, manifest = builder.build([m.id for m in matches])
    assert manifest.label_distribution.player_a_wins + manifest.label_distribution.player_b_wins == 6
    assert manifest.date_range_start <= manifest.date_range_end


def test_repository_fingerprint_is_deterministic():
    _, matches, builder = _setup(4)
    ids = [m.id for m in matches]
    _, manifest_one = builder.build(ids)
    _, manifest_two = builder.build(list(reversed(ids)))
    assert manifest_one.repository_fingerprint == manifest_two.repository_fingerprint
