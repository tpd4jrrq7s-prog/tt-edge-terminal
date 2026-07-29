"""Tests for raw/canonical import model validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from historical_ingestion.models import ImportProvenance, RawRecord, SourceBatch

AWARE = datetime(2026, 1, 1, tzinfo=timezone.utc)
NAIVE = datetime(2026, 1, 1)


def test_raw_record_rejects_naive_timestamps():
    with pytest.raises(ValidationError):
        RawRecord(
            provider="mock", provider_record_id="r1", record_type="match", payload={},
            source_batch_id="b1", source_timestamp=NAIVE, fetched_at=AWARE,
        )


def test_raw_record_accepts_opaque_payload():
    record = RawRecord(
        provider="mock", provider_record_id="r1", record_type="match", payload={"anything": [1, 2, {"x": "y"}]},
        source_batch_id="b1", source_timestamp=AWARE, fetched_at=AWARE,
    )
    assert record.payload["anything"] == [1, 2, {"x": "y"}]


def test_source_batch_rejects_naive_timestamps():
    with pytest.raises(ValidationError):
        SourceBatch(
            source_name="s", provider="mock", source_version="1.0.0", batch_id="b1",
            records=[], next_cursor=None, source_timestamp=NAIVE, fetched_at=AWARE, checksum="abc",
        )


def test_import_provenance_requires_non_blank_ids():
    with pytest.raises(ValidationError):
        ImportProvenance(
            provider="", provider_record_id="r1", source_batch_id="b1",
            source_timestamp=AWARE, ingested_at=AWARE, raw_fingerprint="fp", mapping_version="1.0.0",
        )


def test_imported_match_allows_all_optional_fields_missing():
    from historical_ingestion.models import ImportedMatch

    provenance = ImportProvenance(
        provider="mock", provider_record_id="r1", source_batch_id="b1",
        source_timestamp=AWARE, ingested_at=AWARE, raw_fingerprint="fp", mapping_version="1.0.0",
    )
    match = ImportedMatch(provenance=provenance)
    assert match.scheduled_at is None
    assert match.status is None
    assert match.sets == []
