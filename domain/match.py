"""Domain model representing a table tennis match."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from domain.player import Player


class MatchStatus(str, Enum):
    """Lifecycle status of a match."""

    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class SetScore(BaseModel):
    """Score of a single set within a match."""

    player_one_points: int = Field(default=0, ge=0)
    player_two_points: int = Field(default=0, ge=0)


class Match(BaseModel):
    """A table tennis match between two players.

    Pure domain entity: holds match state and simple derived properties
    only. Ingestion, persistence, and analytics live in separate layers
    added in later phases.
    """

    id: str = Field(..., description="Unique identifier for the match")
    player_one: Player
    player_two: Player
    status: MatchStatus = Field(default=MatchStatus.SCHEDULED)
    sets: list[SetScore] = Field(default_factory=list)

    @property
    def is_live(self) -> bool:
        """Whether the match is currently in progress."""
        return self.status is MatchStatus.LIVE

    @property
    def sets_won_player_one(self) -> int:
        """Number of sets won by player one so far."""
        return sum(1 for s in self.sets if s.player_one_points > s.player_two_points)

    @property
    def sets_won_player_two(self) -> int:
        """Number of sets won by player two so far."""
        return sum(1 for s in self.sets if s.player_two_points > s.player_one_points)
