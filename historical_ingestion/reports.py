"""Typed import reports, quarantine/conflict records, quality metrics, and export helpers.

Metrics are always reported alongside the sample size they're computed
from (`ImportMetrics.sample_size`) — a rate is never presented without
the count backing it, so a 1/1 "100%" rate can't be mistaken for a
well-supported one.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

RecordOutcome = Literal[
    "inserted", "merged_safe", "skipped_idempotent", "quarantined", "rejected_conflict", "rejected_invalid"
]
QuarantineReason = Literal[
    "ambiguous_identity",
    "conflicting_result",
    "impossible_timestamps",
    "invalid_score",
    "unsupported_status",
    "provider_mapping_failure",
    "duplicate_conflict",
    "missing_critical_field",
]
QuarantineStatus = Literal["pending", "resolved", "rejected"]


def compute_run_id(source_name: str, provider: str, started_at: datetime) -> str:
    """A stable, deterministic run ID (no random UUIDs) — the same inputs always produce the same ID."""
    joined = f"{source_name}|{provider}|{started_at.isoformat()}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


class RecordImportResult(BaseModel):
    record_type: str
    provider_record_id: str
    internal_id: str | None = None
    outcome: RecordOutcome
    reasons: list[str] = Field(default_factory=list)


class QuarantineRecord(BaseModel):
    id: str
    record_type: str
    provider: str
    provider_record_id: str
    raw_fingerprint: str
    reason: QuarantineReason
    issues: list[str] = Field(default_factory=list)
    candidate_identity_matches: list[str] = Field(default_factory=list)
    conflict_detail: str | None = None
    created_at: datetime
    status: QuarantineStatus = "pending"


class ConflictRecord(BaseModel):
    id: str
    record_type: str
    internal_id: str
    provider: str
    provider_record_id: str
    conflicting_field: str
    incoming_summary: str
    existing_summary: str
    created_at: datetime


class ImportMetrics(BaseModel):
    """Quality metrics for one batch or a cumulative run. Always paired with `sample_size`."""

    sample_size: int = Field(..., ge=0)
    structural_validity_rate: float | None = None
    semantic_validity_rate: float | None = None
    identity_resolution_rate: float | None = None
    ambiguity_rate: float | None = None
    exact_duplicate_rate: float | None = None
    conflict_rate: float | None = None
    accepted_record_rate: float | None = None
    timestamp_completeness: float | None = None
    score_completeness: float | None = None
    ranking_coverage: float | None = None
    odds_coverage: float | None = None


class BatchImportReport(BaseModel):
    batch_id: str
    records_read: int = Field(..., ge=0)
    records_accepted: int = Field(..., ge=0)
    records_inserted: int = Field(..., ge=0)
    records_skipped: int = Field(..., ge=0)
    records_quarantined: int = Field(..., ge=0)
    records_rejected: int = Field(..., ge=0)
    validation_issue_counts_by_severity: dict[str, int] = Field(default_factory=dict)
    metrics: ImportMetrics
    elapsed_seconds: float = Field(..., ge=0.0)
    checkpoint_cursor_before: str | None = None
    checkpoint_cursor_after: str | None = None
    succeeded: bool
    failure_reason: str | None = None


class ImportReport(BaseModel):
    run_id: str
    source_name: str
    provider: str
    started_at: datetime
    finished_at: datetime
    dry_run: bool
    batches: list[BatchImportReport] = Field(default_factory=list)
    total_records_read: int = Field(default=0, ge=0)
    total_accepted: int = Field(default=0, ge=0)
    total_inserted: int = Field(default=0, ge=0)
    total_skipped: int = Field(default=0, ge=0)
    total_quarantined: int = Field(default=0, ge=0)
    total_rejected: int = Field(default=0, ge=0)
    cumulative_metrics: ImportMetrics
    repository_fingerprint_before: str
    repository_fingerprint_after: str


def format_report_text(report: ImportReport) -> str:
    """A human-readable, deterministic text summary of an import report."""
    lines = [
        f"Import run {report.run_id} — source={report.source_name!r} provider={report.provider!r}"
        f"{' (dry-run)' if report.dry_run else ''}",
        f"  started_at={report.started_at.isoformat()} finished_at={report.finished_at.isoformat()}",
        f"  batches processed: {len(report.batches)}",
        f"  records read={report.total_records_read} accepted={report.total_accepted} "
        f"inserted={report.total_inserted} skipped={report.total_skipped} "
        f"quarantined={report.total_quarantined} rejected={report.total_rejected}",
        f"  repository fingerprint: {report.repository_fingerprint_before} -> {report.repository_fingerprint_after}",
        f"  cumulative metrics (n={report.cumulative_metrics.sample_size}):",
    ]
    metrics = report.cumulative_metrics.model_dump(exclude={"sample_size"})
    for name, value in sorted(metrics.items()):
        if value is not None:
            lines.append(f"    {name}: {value:.3f}")
    for batch in report.batches:
        status = "OK" if batch.succeeded else f"FAILED ({batch.failure_reason})"
        lines.append(
            f"  batch {batch.batch_id}: {status} read={batch.records_read} "
            f"accepted={batch.records_accepted} rejected={batch.records_rejected} "
            f"quarantined={batch.records_quarantined} ({batch.elapsed_seconds:.3f}s)"
        )
    return "\n".join(lines)


def _atomic_write(path: str, content: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-report-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def export_report_json(report: ImportReport, path: str) -> None:
    """Export an ImportReport as pretty-printed, alphabetically-keyed, atomically-written JSON."""
    content = json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
    _atomic_write(path, content)
