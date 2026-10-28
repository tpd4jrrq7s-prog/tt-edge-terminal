"""Domain-specific exceptions for the ML foundation layer."""

from __future__ import annotations


class MLError(Exception):
    """Base class for all ml-layer errors."""


class MissingTargetValueError(MLError):
    """Raised when a requested target cannot be built without fabricating a value."""


class SchemaMismatchError(MLError):
    """Raised when a FeatureMatrix/model pair have incompatible feature schema versions."""


class NotFittedError(MLError):
    """Raised when `.transform()` or a prediction is attempted before `.fit()`."""


class InsufficientDataError(MLError):
    """Raised when there isn't enough data to build a requested split or fold."""
