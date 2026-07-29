"""Domain-specific exceptions for feature building."""

from __future__ import annotations


class FeatureBuildError(Exception):
    """Base class for all feature-building errors."""


class TargetMatchNotFoundError(FeatureBuildError):
    """Raised when the target match for a snapshot cannot be found in the repository."""


class SnapshotLeakageError(FeatureBuildError):
    """Raised when a computed snapshot would violate a leakage invariant.

    This is the per-snapshot safety net used by `features.snapshots`;
    dataset-level leakage checks (train/test overlap, split chronology)
    live in `datasets.leakage.LeakageViolation`.
    """
