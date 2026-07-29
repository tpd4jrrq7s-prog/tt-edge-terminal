"""Domain model representing a table tennis player."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Player(BaseModel):
    """A table tennis player.

    This is a pure domain entity: no database concerns, no external
    source formatting. Ingestion/normalization layers (added in a later
    phase) are responsible for mapping raw source data onto this model.
    """

    id: str = Field(..., description="Unique identifier for the player")
    name: str = Field(..., description="Full name of the player")
    country: str | None = Field(
        default=None, description="Player's country code, e.g. 'NL'"
    )
    ranking: int | None = Field(
        default=None, ge=1, description="Current world ranking, if known"
    )

    def __str__(self) -> str:
        return self.name
