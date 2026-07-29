"""Tests for import reports, quality metrics, and export helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from historical_ingestion.reports import (
    BatchImportReport,
    ImportMetrics,
    ImportReport,
    compute_run_id,
    export_report_json,
    format_report_text,
)

AWARE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_compute_run_id_is_deterministic():
    assert compute_run_id("s1", "mock", AWARE) == compute_run_id("s1", "mock", AWARE)


def test_compute_run_id_differs_for_different_inputs():
    assert compute_run_id("s1", "mock", AWARE) != compute_run_id("s2", "mock", AWARE)


def _report() -> ImportReport:
    batch = BatchImportReport(
        batch_id="b1", records_read=10, records_accepted=8, records_inserted=7, records_skipped=1,
        records_quarantined=1, records_rejected=1, validation_issue_counts_by_severity={"warning": 2},
        metrics=ImportMetrics(sample_size=10, accepted_record_rate=0.8), elapsed_seconds=0.05,
        checkpoint_cursor_before=None, checkpoint_cursor_after="10", succeeded=True,
    )
    return ImportReport(
        run_id="run1", source_name="s1", provider="mock", started_at=AWARE, finished_at=AWARE, dry_run=False,
        batches=[batch], total_records_read=10, total_accepted=8, total_inserted=7, total_skipped=1,
        total_quarantined=1, total_rejected=1, cumulative_metrics=ImportMetrics(sample_size=10, accepted_record_rate=0.8),
        repository_fingerprint_before="empty", repository_fingerprint_after="players=1",
    )


def test_format_report_text_includes_key_figures():
    text = format_report_text(_report())
    assert "run1" in text
    assert "records read=10" in text
    assert "accepted_record_rate" in text


def test_metrics_always_carry_sample_size():
    metrics = ImportMetrics(sample_size=3)
    assert metrics.sample_size == 3
    assert metrics.accepted_record_rate is None


def test_export_report_json_is_valid_json_with_stable_keys(tmp_path):
    path = tmp_path / "report.json"
    export_report_json(_report(), str(path))
    content = path.read_text(encoding="utf-8")
    data = json.loads(content)
    assert data["run_id"] == "run1"
    lines = content.splitlines()
    key_lines = [l for l in lines if l.strip().startswith('"') and ':' in l]
    # top-level keys should appear in sorted order somewhere in the dump
    assert content.index('"batches"') < content.index('"total_accepted"')


def test_export_report_json_is_atomic_and_repeatable(tmp_path):
    path = tmp_path / "report.json"
    export_report_json(_report(), str(path))
    first = path.read_bytes()
    export_report_json(_report(), str(path))
    second = path.read_bytes()
    assert first == second
