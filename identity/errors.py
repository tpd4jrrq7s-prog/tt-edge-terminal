"""Domain-specific exceptions for player identity resolution."""

from __future__ import annotations


class IdentityError(Exception):
    """Base class for all identity-resolution errors."""


class InvalidPlayerNameError(IdentityError):
    """Raised when a player name is blank or otherwise unusable for identity resolution."""
