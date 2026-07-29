"""Deterministic player identity resolution.

Scoring is a transparent, dependency-free similarity computation
(stdlib `difflib.SequenceMatcher`) plus explicit, documented adjustments
for exact external IDs, country/birth-date agreement, and short-name
risk — never a heavyweight fuzzy-matching library, and never an
auto-merge of an ambiguous case.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from config.historical import HistoricalIntelligenceSettings, get_historical_intelligence_settings
from identity.models import (
    IdentityCandidate,
    IdentityOutcome,
    IdentityResolution,
    NormalizedPlayerIdentity,
    PlayerIdentityRecord,
)

_EXTERNAL_ID_CONFLICT_SANITY_FLOOR = 0.3


def _name_similarity(a: str, b: str) -> float:
    """Transparent, deterministic string similarity in [0, 1] (stdlib difflib)."""
    return SequenceMatcher(None, a, b).ratio()


def _best_name_similarity(candidate_name: str, identity: PlayerIdentityRecord) -> float:
    names = [identity.normalized_name, *identity.aliases]
    return max(_name_similarity(candidate_name, n) for n in names)


def _find_exact_external_match(
    candidate: NormalizedPlayerIdentity, known_identities: list[PlayerIdentityRecord]
) -> PlayerIdentityRecord | None:
    if candidate.external_provider is None or candidate.external_player_id is None:
        return None
    for identity in known_identities:
        for ext in identity.external_ids:
            if ext.provider == candidate.external_provider and ext.provider_player_id == candidate.external_player_id:
                return identity
    return None


class IdentityResolver:
    """Stateless, deterministic resolver: a pure function of its inputs.

    Holds no identity store itself — callers supply `known_identities`
    on every call and are responsible for persisting newly created or
    matched identities. This keeps resolution side-effect-free and fully
    testable in isolation.
    """

    def __init__(self, settings: HistoricalIntelligenceSettings | None = None) -> None:
        self._settings = settings or get_historical_intelligence_settings()

    def resolve(
        self,
        candidate: NormalizedPlayerIdentity,
        known_identities: list[PlayerIdentityRecord],
    ) -> IdentityResolution:
        settings = self._settings

        exact_match = _find_exact_external_match(candidate, known_identities)
        if exact_match is not None:
            name_similarity = _best_name_similarity(candidate.normalized_name, exact_match)
            if name_similarity < _EXTERNAL_ID_CONFLICT_SANITY_FLOOR:
                return IdentityResolution(
                    outcome=IdentityOutcome.REJECTED,
                    reasons=[
                        f"External identifier ({candidate.external_provider}, "
                        f"{candidate.external_player_id}) maps to identity {exact_match.id!r}, "
                        f"but name similarity is only {name_similarity:.2f} — likely a data conflict"
                    ],
                )
            return IdentityResolution(
                outcome=IdentityOutcome.MATCHED,
                identity_id=exact_match.id,
                candidates=[
                    IdentityCandidate(
                        identity_id=exact_match.id, score=1.0, reasons=["exact external identifier match"]
                    )
                ],
                reasons=["exact external identifier match outranks fuzzy name comparison"],
            )

        if not known_identities:
            return IdentityResolution(
                outcome=IdentityOutcome.CREATED,
                reasons=["no known identities to compare against"],
            )

        candidates: list[IdentityCandidate] = []
        for identity in known_identities:
            reasons: list[str] = []
            score = _best_name_similarity(candidate.normalized_name, identity)
            reasons.append(f"name similarity {score:.3f}")

            if candidate.country is not None and identity.country is not None:
                if candidate.country != identity.country:
                    score *= settings.country_mismatch_penalty
                    reasons.append(f"country mismatch ({candidate.country} vs {identity.country}) reduced score")
                else:
                    reasons.append("country matches")

            if candidate.birth_date is not None and identity.birth_date is not None:
                if candidate.birth_date != identity.birth_date:
                    score *= settings.birth_date_mismatch_penalty
                    reasons.append("birth date mismatch reduced score")
                else:
                    reasons.append("birth date matches")

            candidates.append(IdentityCandidate(identity_id=identity.id, score=min(score, 1.0), reasons=reasons))

        candidates.sort(key=lambda c: (-c.score, c.identity_id))
        top = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else 0.0

        required_threshold = settings.identity_match_threshold
        is_short_name = len(candidate.normalized_name) <= settings.short_name_length_threshold
        if is_short_name:
            required_threshold += settings.short_name_extra_margin

        if top.score < required_threshold:
            return IdentityResolution(
                outcome=IdentityOutcome.CREATED,
                candidates=candidates,
                reasons=[
                    f"best candidate score {top.score:.3f} is below the required threshold "
                    f"{required_threshold:.3f}; treating as a new identity"
                ],
            )

        margin = top.score - second_score
        if margin < settings.identity_ambiguity_margin:
            return IdentityResolution(
                outcome=IdentityOutcome.AMBIGUOUS,
                candidates=candidates,
                reasons=[
                    f"top two candidates are within the ambiguity margin "
                    f"({top.score:.3f} vs {second_score:.3f}); refusing to auto-merge"
                ],
            )

        return IdentityResolution(
            outcome=IdentityOutcome.MATCHED,
            identity_id=top.identity_id,
            candidates=candidates,
            reasons=[f"best candidate score {top.score:.3f} clears threshold and ambiguity margin"],
        )
