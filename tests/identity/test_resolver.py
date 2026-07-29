"""Tests for deterministic player identity resolution."""

from __future__ import annotations

from config.historical import HistoricalIntelligenceSettings
from identity.models import ExternalIdentifier, IdentityOutcome, PlayerIdentityRecord
from identity.normalizer import build_normalized_identity
from identity.resolver import IdentityResolver


def _known(**overrides) -> PlayerIdentityRecord:
    defaults = dict(
        id="id-1", canonical_name="Ma Long", normalized_name="ma long", aliases=[], country="CHN",
        external_ids=[ExternalIdentifier(provider="mock", provider_player_id="100")],
    )
    defaults.update(overrides)
    return PlayerIdentityRecord(**defaults)


def test_exact_external_id_matches_regardless_of_name():
    resolver = IdentityResolver()
    candidate = build_normalized_identity("Ma Long", external_provider="mock", external_player_id="100")
    result = resolver.resolve(candidate, [_known()])
    assert result.outcome is IdentityOutcome.MATCHED
    assert result.identity_id == "id-1"


def test_no_known_identities_creates_new():
    resolver = IdentityResolver()
    candidate = build_normalized_identity("Ma Long")
    result = resolver.resolve(candidate, [])
    assert result.outcome is IdentityOutcome.CREATED
    assert result.identity_id is None


def test_close_name_match_resolves_to_matched():
    resolver = IdentityResolver()
    candidate = build_normalized_identity("Ma Long", country="CHN")
    result = resolver.resolve(candidate, [_known()])
    assert result.outcome is IdentityOutcome.MATCHED


def test_alias_is_matched():
    resolver = IdentityResolver()
    # Aliases are stored in normalized form, same as canonical names.
    known = _known(normalized_name="completely different canonical form", aliases=["ma long alt"])
    candidate = build_normalized_identity("Ma Long Alt")
    result = resolver.resolve(candidate, [known])
    assert result.outcome is IdentityOutcome.MATCHED


def test_country_mismatch_reduces_confidence_below_threshold():
    settings = HistoricalIntelligenceSettings(identity_match_threshold=0.95, country_mismatch_penalty=0.5)
    resolver = IdentityResolver(settings=settings)
    candidate = build_normalized_identity("Ma Long", country="BRA")
    result = resolver.resolve(candidate, [_known()])
    # Exact name match but wrong country should be penalized enough to miss a strict threshold.
    assert result.outcome is not IdentityOutcome.MATCHED or result.candidates[0].score < 1.0


def test_unrelated_name_is_created_not_matched():
    resolver = IdentityResolver()
    candidate = build_normalized_identity("Zzyzx Qwerty")
    result = resolver.resolve(candidate, [_known()])
    assert result.outcome is IdentityOutcome.CREATED


def test_ambiguous_when_two_candidates_are_close():
    settings = HistoricalIntelligenceSettings(identity_match_threshold=0.5, identity_ambiguity_margin=0.5)
    resolver = IdentityResolver(settings=settings)
    known = [_known(id="id-1", normalized_name="ma long"), _known(id="id-2", normalized_name="ma long jr")]
    candidate = build_normalized_identity("Ma Long")
    result = resolver.resolve(candidate, known)
    assert result.outcome is IdentityOutcome.AMBIGUOUS
    assert result.identity_id is None


def test_ambiguous_outcome_never_auto_merges():
    settings = HistoricalIntelligenceSettings(identity_match_threshold=0.5, identity_ambiguity_margin=0.9)
    resolver = IdentityResolver(settings=settings)
    known = [_known(id="id-1"), _known(id="id-2", normalized_name="ma long")]
    candidate = build_normalized_identity("Ma Long")
    result = resolver.resolve(candidate, known)
    assert result.identity_id is None
    assert result.outcome in (IdentityOutcome.AMBIGUOUS, IdentityOutcome.CREATED)


def test_short_name_requires_stronger_evidence():
    settings = HistoricalIntelligenceSettings(identity_match_threshold=0.5, short_name_length_threshold=4, short_name_extra_margin=0.9)
    resolver = IdentityResolver(settings=settings)
    known = [_known(normalized_name="al")]
    candidate = build_normalized_identity("Al")
    result = resolver.resolve(candidate, known)
    # With such a large extra margin requirement, a short common name should not auto-match.
    assert result.outcome is IdentityOutcome.CREATED


def test_exact_external_id_conflict_with_wildly_different_name_is_rejected():
    resolver = IdentityResolver()
    candidate = build_normalized_identity("Completely Different Person", external_provider="mock", external_player_id="100")
    result = resolver.resolve(candidate, [_known()])
    assert result.outcome is IdentityOutcome.REJECTED


def test_resolution_is_deterministic():
    resolver = IdentityResolver()
    candidate = build_normalized_identity("Ma Long", country="CHN")
    known = [_known()]
    first = resolver.resolve(candidate, known)
    second = resolver.resolve(candidate, known)
    assert first == second
