"""Typed contracts for player identity resolution."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class ExternalIdentifier(BaseModel):
    """One provider's identifier for a player, preserved as provenance."""

    provider: str = Field(..., min_length=1)
    provider_player_id: str = Field(..., min_length=1)


class NormalizedPlayerIdentity(BaseModel):
    """A deterministically normalized view of one incoming player observation.

    This is the *input* to resolution — it is never itself a stored,
    canonical identity.
    """

    original_name: str = Field(..., min_length=1)
    normalized_name: str
    country: str | None = None
    birth_date: date | None = None
    external_provider: str | None = None
    external_player_id: str | None = None


class PlayerIdentityRecord(BaseModel):
    """A stored, canonical player identity, with known aliases and external ID provenance."""

    id: str = Field(..., min_length=1)
    canonical_name: str = Field(..., min_length=1)
    normalized_name: str
    aliases: list[str] = Field(default_factory=list)
    country: str | None = None
    birth_date: date | None = None
    external_ids: list[ExternalIdentifier] = Field(default_factory=list)


class IdentityCandidate(BaseModel):
    """One scored candidate match against a known `PlayerIdentityRecord`."""

    identity_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class IdentityOutcome(str, Enum):
    MATCHED = "matched"
    CREATED = "created"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"


class IdentityResolution(BaseModel):
    """The result of resolving one `NormalizedPlayerIdentity` against known identities."""

    outcome: IdentityOutcome
    identity_id: str | None = None
    candidates: list[IdentityCandidate] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
