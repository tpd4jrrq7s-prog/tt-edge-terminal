"""Domain-specific exceptions for the historical ingestion platform."""

from __future__ import annotations


class HistoricalIngestionError(Exception):
    """Base class for all historical-ingestion errors."""


class SourceReadError(HistoricalIngestionError):
    """Raised when a source fails to read/parse a batch (malformed file, bad row, etc.)."""


class CheckpointVersionError(HistoricalIngestionError):
    """Raised when a stored checkpoint's version is incompatible with the expected version."""


class BatchProcessingError(HistoricalIngestionError):
    """Raised for a fatal, unexpected error mid-batch — triggers a full-batch rollback (no partial writes)."""
