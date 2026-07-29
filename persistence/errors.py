"""Domain-specific exceptions for the persistence layer."""

from __future__ import annotations


class PersistenceError(Exception):
    """Base class for all persistence-layer errors."""


class DuplicateRecordError(PersistenceError):
    """Raised when a record with the same stable ID (or unique key) already exists.

    Policy: exact-duplicate stable IDs are rejected rather than silently
    ignored or overwritten — callers that want idempotent inserts should
    check `get()`/`has_provider_id()` first.
    """


class ProviderMappingConflictError(PersistenceError):
    """Raised when the same (provider, provider_id) is mapped to two different internal IDs."""


class RecordNotFoundError(PersistenceError):
    """Raised when an operation requires a record that does not exist."""
