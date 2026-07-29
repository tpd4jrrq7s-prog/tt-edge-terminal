"""Domain-specific exceptions for dataset building, splitting, and leakage detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets.leakage import LeakageReport


class DatasetError(Exception):
    """Base class for all dataset-layer errors."""


class LeakageViolation(DatasetError):
    """Raised when a leakage check fails. Carries the structured `LeakageReport`."""

    def __init__(self, message: str, report: "LeakageReport") -> None:
        super().__init__(message)
        self.report = report


class InsufficientDataForSplitError(DatasetError):
    """Raised when there is not enough data to satisfy a requested split."""


class InvalidLabelError(DatasetError):
    """Raised when a training example's label cannot be derived validly."""
