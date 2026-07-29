"""Tests for the JSONL file source."""

from __future__ import annotations

import pytest

from historical_ingestion.errors import SourceReadError
from historical_ingestion.sources.jsonl_source import JSONLFileSource


def _write(tmp_path, lines):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_reads_records_deterministically(tmp_path):
    path = _write(tmp_path, ['{"id": "m1", "status": "COMPLETED"}', '{"id": "m2", "status": "SCHEDULED"}'])
    source = JSONLFileSource(path, provider="mock", record_type="match")
    batch = source.fetch_batch(None)
    assert len(batch.records) == 2
    assert batch.records[0].provider_record_id == "m1"
    assert batch.next_cursor is None


def test_pagination_cursor_resumes_correctly(tmp_path):
    path = _write(tmp_path, [f'{{"id": "m{i}"}}' for i in range(5)])
    source = JSONLFileSource(path, provider="mock", record_type="match", batch_size=2)
    first = source.fetch_batch(None)
    assert len(first.records) == 2
    assert first.next_cursor == "2"
    second = source.fetch_batch(first.next_cursor)
    assert len(second.records) == 2
    assert [r.provider_record_id for r in second.records] == ["m2", "m3"]


def test_blank_lines_are_skipped(tmp_path):
    path = _write(tmp_path, ['{"id": "m1"}', '', '   ', '{"id": "m2"}'])
    source = JSONLFileSource(path, provider="mock", record_type="match")
    batch = source.fetch_batch(None)
    assert len(batch.records) == 2


def test_malformed_json_line_raises_clear_error(tmp_path):
    path = _write(tmp_path, ['{"id": "m1"}', 'not valid json'])
    source = JSONLFileSource(path, provider="mock", record_type="match")
    with pytest.raises(SourceReadError, match=r":2:"):
        source.fetch_batch(None)


def test_non_object_line_raises_clear_error(tmp_path):
    path = _write(tmp_path, ['[1, 2, 3]'])
    source = JSONLFileSource(path, provider="mock", record_type="match")
    with pytest.raises(SourceReadError):
        source.fetch_batch(None)


def test_checksum_is_stable_for_identical_batches(tmp_path):
    path = _write(tmp_path, ['{"id": "m1"}'])
    source = JSONLFileSource(path, provider="mock", record_type="match")
    first = source.fetch_batch(None)
    second = source.fetch_batch(None)
    assert first.checksum == second.checksum


def test_health_reports_missing_file(tmp_path):
    source = JSONLFileSource(str(tmp_path / "nope.jsonl"), provider="mock", record_type="match")
    health = source.health()
    assert health.healthy is False
