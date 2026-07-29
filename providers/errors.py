"""Domain-specific exceptions for provider adapters."""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for all provider-adapter errors."""


class UnknownProviderStatusError(ProviderError):
    """Raised when a raw provider status has no configured mapping and the policy is 'reject'."""


class ProviderMappingError(ProviderError):
    """Raised when a raw record cannot be mapped at all (e.g. unknown record_type)."""


class ProviderNotRegisteredError(ProviderError):
    """Raised when `ProviderRegistry.get` is called for an unregistered provider name."""
