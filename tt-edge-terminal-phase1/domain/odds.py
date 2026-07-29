"""Domain model representing bookmaker odds for a match.

Note: this model represents odds purely as *data* for analytics and
display. Nothing in this codebase places bets or otherwise acts on
these values — this is a decision-support platform only.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Odds(BaseModel):
    """A single odds quote for a match at a point in time."""

    match_id: str = Field(..., description="Identifier of the match these odds relate to")
    bookmaker: str = Field(..., description="Name of the odds source/bookmaker")
    player_one_odds: float = Field(..., gt=1.0, description="Decimal odds for player one to win")
    player_two_odds: float = Field(..., gt=1.0, description="Decimal odds for player two to win")
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when these odds were captured",
    )

    @property
    def implied_probability_player_one(self) -> float:
        """Implied win probability for player one, derived from decimal odds."""
        return 1.0 / self.player_one_odds

    @property
    def implied_probability_player_two(self) -> float:
        """Implied win probability for player two, derived from decimal odds."""
        return 1.0 / self.player_two_odds
