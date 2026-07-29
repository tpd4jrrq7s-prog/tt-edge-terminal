"""Normalize raw provider odds data into the domain `Odds` model."""

from __future__ import annotations

from datetime import datetime

from domain.odds import Odds
from ingestion.models import RawOdds


class OddsNormalizationError(ValueError):
    """Raised when raw provider odds data cannot be normalized into a domain Odds quote."""


def _parse_captured_at(raw_value: str, *, provider_match_id: str) -> datetime:
    try:
        return datetime.fromisoformat(raw_value)
    except (TypeError, ValueError) as exc:
        raise OddsNormalizationError(
            f"Odds for match {provider_match_id!r} have an invalid captured_at "
            f"timestamp: {raw_value!r}"
        ) from exc


def normalize_odds(raw: RawOdds) -> Odds:
    """Convert a `RawOdds` into a validated domain `Odds` quote.

    Raises `OddsNormalizationError` for a blank match id or bookmaker,
    odds that are not valid decimal odds (i.e. not greater than 1.0), or
    an unparseable `captured_at` timestamp.
    """
    if not raw.provider_match_id.strip():
        raise OddsNormalizationError("Odds must reference a non-blank match id")
    if not raw.bookmaker.strip():
        raise OddsNormalizationError(
            f"Odds for match {raw.provider_match_id!r} must specify a bookmaker"
        )
    if raw.player_one_odds <= 1.0 or raw.player_two_odds <= 1.0:
        raise OddsNormalizationError(
            f"Odds for match {raw.provider_match_id!r} must be decimal odds greater than "
            f"1.0 (got player_one_odds={raw.player_one_odds}, "
            f"player_two_odds={raw.player_two_odds})"
        )

    captured_at = _parse_captured_at(raw.captured_at, provider_match_id=raw.provider_match_id)

    return Odds(
        match_id=raw.provider_match_id,
        bookmaker=raw.bookmaker,
        player_one_odds=raw.player_one_odds,
        player_two_odds=raw.player_two_odds,
        captured_at=captured_at,
    )
