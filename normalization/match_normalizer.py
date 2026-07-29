"""Normalize raw provider match data into the domain `Match` model."""

from __future__ import annotations

from datetime import datetime

from domain.match import Match, MatchStatus, SetScore
from domain.player import Player
from ingestion.models import RawMatch, RawPlayer, RawSetScore


class MatchNormalizationError(ValueError):
    """Raised when raw provider match data cannot be normalized into a domain Match."""


_STATUS_MAP: dict[str, MatchStatus] = {
    "scheduled": MatchStatus.SCHEDULED,
    "not_started": MatchStatus.SCHEDULED,
    "live": MatchStatus.LIVE,
    "in_progress": MatchStatus.LIVE,
    "finished": MatchStatus.FINISHED,
    "ended": MatchStatus.FINISHED,
    "cancelled": MatchStatus.CANCELLED,
    "canceled": MatchStatus.CANCELLED,
}


def _normalize_player(raw: RawPlayer) -> Player:
    if not raw.provider_player_id.strip():
        raise MatchNormalizationError("Player id must not be blank")
    if not raw.full_name.strip():
        raise MatchNormalizationError(f"Player {raw.provider_player_id!r} has a blank name")
    return Player(
        id=raw.provider_player_id,
        name=raw.full_name,
        country=raw.country_code,
        ranking=raw.world_ranking,
    )


def _normalize_status(raw_status: str) -> MatchStatus:
    key = raw_status.strip().lower()
    try:
        return _STATUS_MAP[key]
    except KeyError as exc:
        raise MatchNormalizationError(f"Unknown match status: {raw_status!r}") from exc


def _normalize_sets(raw_sets: list[RawSetScore]) -> list[SetScore]:
    return [
        SetScore(player_one_points=s.player_one_points, player_two_points=s.player_two_points)
        for s in raw_sets
    ]


def _validate_scheduled_at(raw_value: str, *, provider_match_id: str) -> None:
    try:
        datetime.fromisoformat(raw_value)
    except (TypeError, ValueError) as exc:
        raise MatchNormalizationError(
            f"Match {provider_match_id!r} has an invalid scheduled_at timestamp: {raw_value!r}"
        ) from exc


def normalize_match(raw: RawMatch) -> Match:
    """Convert a `RawMatch` into a validated domain `Match`.

    Raises `MatchNormalizationError` for a blank match id, invalid or
    duplicate player mapping, an unrecognized status, or an unparseable
    `scheduled_at` timestamp.
    """
    if not raw.provider_match_id.strip():
        raise MatchNormalizationError("Match id must not be blank")

    _validate_scheduled_at(raw.scheduled_at, provider_match_id=raw.provider_match_id)

    player_one = _normalize_player(raw.player_one)
    player_two = _normalize_player(raw.player_two)
    if player_one.id == player_two.id:
        raise MatchNormalizationError(
            f"Match {raw.provider_match_id!r} cannot have the same player on both sides"
        )

    status = _normalize_status(raw.status)
    sets = _normalize_sets(raw.sets)

    return Match(
        id=raw.provider_match_id,
        player_one=player_one,
        player_two=player_two,
        status=status,
        sets=sets,
    )
