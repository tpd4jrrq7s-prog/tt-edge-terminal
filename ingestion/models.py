"""Typed models for raw, provider-independent ingestion data.

These models describe match/odds data the way an external provider might
supply it: identifiers and status/timestamp strings that are structurally
valid but not yet guaranteed to be *semantically* correct (e.g. a status
string might not map to any known `MatchStatus`, odds might not be valid
decimal odds). That validation is the job of the `normalization` layer,
not this module — raw models stay deliberately permissive so they can
represent imperfect provider data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RawPlayer(BaseModel):
    """A player as reported by a raw data provider."""

    provider_player_id: str = Field(..., description="Provider-assigned player identifier")
    full_name: str = Field(..., description="Player's full name as reported by the provider")
    country_code: str | None = Field(default=None, description="Country code, if provided")
    world_ranking: int | None = Field(default=None, description="World ranking, if provided")


class RawSetScore(BaseModel):
    """A single set score as reported by a raw data provider."""

    player_one_points: int = Field(default=0, ge=0)
    player_two_points: int = Field(default=0, ge=0)


class RawMatch(BaseModel):
    """A match as reported by a raw data provider, prior to normalization."""

    provider_match_id: str = Field(..., description="Provider-assigned match identifier")
    player_one: RawPlayer
    player_two: RawPlayer
    status: str = Field(..., description="Raw provider match status, e.g. 'scheduled'")
    sets: list[RawSetScore] = Field(default_factory=list)
    scheduled_at: str = Field(..., description="ISO 8601 timestamp string")


class RawOdds(BaseModel):
    """A single odds quote as reported by a raw data provider.

    Odds values are intentionally not constrained here (unlike the
    domain `Odds` model) — normalization is responsible for rejecting
    non-decimal odds with a clear, domain-specific error.
    """

    provider_match_id: str = Field(..., description="Identifier of the match these odds relate to")
    bookmaker: str = Field(..., description="Name of the bookmaker/odds source")
    player_one_odds: float = Field(..., description="Raw decimal odds for player one, unvalidated")
    player_two_odds: float = Field(..., description="Raw decimal odds for player two, unvalidated")
    captured_at: str = Field(..., description="ISO 8601 timestamp string")
