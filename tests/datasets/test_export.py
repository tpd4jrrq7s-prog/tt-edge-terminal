"""Tests for deterministic dataset export (JSONL, CSV, manifest)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone

from datasets.builder import DatasetBuilder
from datasets.export import export_csv, export_jsonl, export_manifest
from features.builder import HistoricalFeatureBuilder
from persistence.in_memory import InMemoryMatchRepository
from persistence.models import HistoricalMatchRecord, HistoricalSetRecord, MatchRecordStatus

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _finished(mid, day, winner="p1") -> HistoricalMatchRecord:
    scheduled = BASE + timedelta(days=day)
    return HistoricalMatchRecord(
        id=mid, provider="mock", provider_match_id=mid, player_a_id="p1", player_b_id="p2",
        scheduled_at=scheduled, actual_start_at=scheduled, completed_at=scheduled + timedelta(hours=1),
        status=MatchRecordStatus.FINISHED, winner_id=winner,
        sets=[HistoricalSetRecord(set_number=1, player_a_points=11, player_b_points=7)],
        provider_timestamp=scheduled, ingested_at=scheduled,
    )


def _examples(n=4):
    repo = InMemoryMatchRepository()
    matches = [_finished(f"m{i}", i, winner=("p1" if i % 2 == 0 else "p2")) for i in range(n)]
    repo.add_many(matches)
    feature_builder = HistoricalFeatureBuilder(repo)
    builder = DatasetBuilder(repo, feature_builder)
    examples, manifest = builder.build([m.id for m in matches], dataset_id="ds")
    return examples, manifest


def test_jsonl_export_is_valid_and_one_object_per_line(tmp_path):
    examples, _ = _examples()
    path = tmp_path / "out.jsonl"
    export_jsonl(examples, str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(examples)
    for line in lines:
        json.loads(line)  # must parse


def test_jsonl_row_keys_are_alphabetically_sorted():
    from datasets.export import _row_for

    examples, _ = _examples()
    row = _row_for(examples[0], 6)
    serialized = json.dumps(row, sort_keys=True)
    assert list(json.loads(serialized).keys()) == sorted(row.keys())


def test_csv_export_has_stable_alphabetical_header(tmp_path):
    examples, _ = _examples()
    path = tmp_path / "out.csv"
    export_csv(examples, str(path))
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    assert header == sorted(header)


def test_csv_row_count_matches_examples(tmp_path):
    examples, _ = _examples()
    path = tmp_path / "out.csv"
    export_csv(examples, str(path))
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == len(examples)


def test_repeated_jsonl_export_is_byte_identical(tmp_path):
    examples, _ = _examples()
    path_one = tmp_path / "one.jsonl"
    path_two = tmp_path / "two.jsonl"
    export_jsonl(examples, str(path_one))
    export_jsonl(examples, str(path_two))
    assert path_one.read_bytes() == path_two.read_bytes()


def test_repeated_csv_export_is_byte_identical(tmp_path):
    examples, _ = _examples()
    path_one = tmp_path / "one.csv"
    path_two = tmp_path / "two.csv"
    export_csv(examples, str(path_one))
    export_csv(examples, str(path_two))
    assert path_one.read_bytes() == path_two.read_bytes()


def test_manifest_export_round_trips(tmp_path):
    _, manifest = _examples()
    path = tmp_path / "manifest.json"
    export_manifest(manifest, str(path))
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["dataset_id"] == manifest.dataset_id
    assert loaded["training_example_count"] == manifest.training_example_count


def test_export_uses_utc_iso_timestamps(tmp_path):
    examples, _ = _examples()
    path = tmp_path / "out.jsonl"
    export_jsonl(examples, str(path))
    first_row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first_row["as_of"].endswith("+00:00") or first_row["as_of"].endswith("Z")


def test_no_pickle_or_arbitrary_code_formats_used():
    import datasets.export as export_module

    source = open(export_module.__file__, encoding="utf-8").read()
    assert "import pickle" not in source
    assert "eval(" not in source
    assert "exec(" not in source
