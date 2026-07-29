"""Deterministic player-name normalization.

Applies, in order: Unicode NFKD decomposition + combining-mark removal
(a safe, reversible-in-intent way to fold accented Latin characters,
e.g. "Ovtcharov" variants), case folding, punctuation normalization,
and whitespace collapsing.

Limitation (documented, not silently attempted): this does **not**
perform cross-script transliteration (e.g. Cyrillic/CJK to Latin) — that
requires a curated, locale-aware mapping to be safe, and guessing one
would risk silently merging distinct players. Only accent-folding within
already-Latin text is performed.
"""

from __future__ import annotations

import unicodedata

from identity.errors import InvalidPlayerNameError
from identity.models import NormalizedPlayerIdentity

# Known alias -> canonical-form mapping, applied after normalization.
# Deliberately small and explicit rather than a fuzzy/learned mapping.
KNOWN_ALIASES: dict[str, str] = {}


def normalize_player_name(name: str) -> str:
    """Deterministically normalize a player name into a comparable canonical form."""
    if not name or not name.strip():
        raise InvalidPlayerNameError("Player name must not be blank")

    decomposed = unicodedata.normalize("NFKD", name)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    folded = without_marks.casefold()
    punctuation_stripped = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in folded)
    collapsed = " ".join(punctuation_stripped.split())

    if not collapsed:
        raise InvalidPlayerNameError(f"Player name {name!r} normalized to an empty string")

    return KNOWN_ALIASES.get(collapsed, collapsed)


def build_normalized_identity(
    name: str,
    *,
    country: str | None = None,
    birth_date=None,
    external_provider: str | None = None,
    external_player_id: str | None = None,
) -> NormalizedPlayerIdentity:
    """Build a `NormalizedPlayerIdentity` from raw provider-supplied fields."""
    return NormalizedPlayerIdentity(
        original_name=name,
        normalized_name=normalize_player_name(name),
        country=country,
        birth_date=birth_date,
        external_provider=external_provider,
        external_player_id=external_player_id,
    )
