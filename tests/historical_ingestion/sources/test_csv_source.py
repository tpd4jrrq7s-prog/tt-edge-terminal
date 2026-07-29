"""Tests for the CSV file source."""

from __future__ import annotations

import pytest

from historical_ingestion.errors import SourceReadError
from historical_ingestion.sources.csv_source import CSVFileSource


def _write(tmp_path, content: str, name="data.csv"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_reads_rows_with_header(tmp_path):
    path = _write(tmp_path, "id,name,country\np1,Ma Long,CHN\np2,Fan Zhendong,CHN\n")
    source = CSVFileSource(path, provider="mock", record_type="player")
    batch = source.fetch_batch(None)
    assert len(batch.records) == 2
    assert batch.records[0].payload["name"] == "Ma Long"
    assert batch.records[0].provider_record_id == "p1"


def test_configurable_delimiter(tmp_path):
    path = _write(tmp_path, "id;name\np1;Ma Long\n")
    source = CSVFileSource(path, provider="mock", record_type="player", delimiter=";")
    batch = source.fetch_batch(None)
    assert batch.records[0].payload["name"] == "Ma Long"


def test_pagination_cursor_resumes(tmp_path):
    rows = "\n".join(f"id{i},v{i}" for i in range(5))
    path = _write(tmp_path, f"id,value\n{rows}\n")
    source = CSVFileSource(path, provider="mock", record_type="player", batch_size=2)
    first = source.fetch_batch(None)
    assert len(first.records) == 2
    second = source.fetch_batch(first.next_cursor)
    assert [r.provider_record_id for r in second.records] == ["id2", "id3"]


def test_missing_header_raises_clear_error(tmp_path):
    path = _write(tmp_path, "")
    source = CSVFileSource(path, provider="mock", record_type="player")
    with pytest.raises(SourceReadError):
        source.fetch_batch(None)


def test_row_with_extra_columns_raises_clear_error(tmp_path):
    path = _write(tmp_path, "id,name\np1,Ma Long,extra\n")
    source = CSVFileSource(path, provider="mock", record_type="player")
    with pytest.raises(SourceReadError):
        source.fetch_batch(None)


def test_no_pickle_or_pandas_used():
    import historical_ingestion.sources.csv_source as module

    source = open(module.__file__, encoding="utf-8").read()
    assert "import pickle" not in source
    assert "import pandas" not in source
