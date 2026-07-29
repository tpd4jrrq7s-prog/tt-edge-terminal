"""MatchAnalyticsEngine: the single orchestrator tying every engine stage together.

Pipeline: data quality -> form -> momentum -> match features -> probability
-> confidence -> risk -> value -> patterns -> explanations -> MatchAnalysis.

No database, no external API calls, no web scraping, no background
threads, and no global mutable state — every dependency (settings,
clock) is injected, and the engine instance itself holds no per-call state.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timezone

from config.analytics import AnalyticsSettings, get_analytics_settings
from engine.confidence import calculate_confidence
from engine.explanations import build_explanations
from engine.form import calculate_form
from engine.momentum import calculate_momentum
from engine.models import MatchAnalysis, MatchAnalysisRequest, MatchFeatures, MomentumResult, PlayerMetrics
from engine.patterns import detect_patterns
from engine.probability import calculate_probability
from engine.quality import assess_data_quality
from engine.risk import calculate_risk
from engine.value import assess_value


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ranking_advantage(rank_one: int | None, rank_two: int | None, scale: float) -> float | None:
    if rank_one is None or rank_two is None:
        return None
    diff = rank_two - rank_one  # positive favors player_one (lower ranking number is better)
    return math.tanh(diff / scale)


def _head_to_head_signal(head_to_head) -> float:  # type: ignore[no-untyped-def]
    if head_to_head is None or head_to_head.total_matches == 0:
        return 0.0
    return (head_to_head.player_one_wins - head_to_head.player_two_wins) / head_to_head.total_matches


def _match_state_signal(match) -> float:  # type: ignore[no-untyped-def]
    if not match.sets:
        return 0.0
    diff = match.sets_won_player_one - match.sets_won_player_two
    return math.tanh(diff * 0.6)


def build_match_features(
    request: MatchAnalysisRequest,
    form_one,  # type: ignore[no-untyped-def]
    form_two,  # type: ignore[no-untyped-def]
    momentum: MomentumResult,
    settings: AnalyticsSettings,
) -> MatchFeatures:
    """Assemble comparable, derived features for both players in this match."""
    match = request.match

    player_one_metrics = PlayerMetrics(
        player_id=match.player_one.id,
        player_name=match.player_one.name,
        ranking=match.player_one.ranking,
        form_score=form_one.score,
        form_confidence=form_one.confidence,
        matches_considered=form_one.matches_considered,
        momentum_score=momentum.player_one_score,
        momentum_state=momentum.state,
    )
    player_two_metrics = PlayerMetrics(
        player_id=match.player_two.id,
        player_name=match.player_two.name,
        ranking=match.player_two.ranking,
        form_score=form_two.score,
        form_confidence=form_two.confidence,
        matches_considered=form_two.matches_considered,
        momentum_score=momentum.player_two_score,
        momentum_state=momentum.state,
    )

    ranking_differential = _ranking_advantage(
        match.player_one.ranking, match.player_two.ranking, settings.ranking_scale
    )
    form_differential = _clamp((form_one.score - form_two.score) / 100.0, -1.0, 1.0)
    momentum_differential = _clamp(
        (momentum.player_one_score - momentum.player_two_score) / 100.0, -1.0, 1.0
    )
    head_to_head_signal = _clamp(_head_to_head_signal(request.head_to_head), -1.0, 1.0)
    match_state_signal = _clamp(_match_state_signal(match), -1.0, 1.0)

    return MatchFeatures(
        player_one=player_one_metrics,
        player_two=player_two_metrics,
        ranking_differential=ranking_differential,
        form_differential=form_differential,
        momentum_differential=momentum_differential,
        head_to_head=request.head_to_head,
        head_to_head_signal=head_to_head_signal,
        match_state_signal=match_state_signal,
        context=request.context,
    )


class MatchAnalyticsEngine:
    """Deterministic, rules-based analytics orchestrator for a single match."""

    def __init__(
        self,
        settings: AnalyticsSettings | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings or get_analytics_settings()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def analyze(self, request: MatchAnalysisRequest) -> MatchAnalysis:
        """Run the full analytics pipeline for one match and return a typed result."""
        settings = self._settings
        as_of = self._clock()
        match = request.match

        quality = assess_data_quality(
            match=match,
            odds=request.odds,
            player_one_history=request.player_one_history,
            player_two_history=request.player_two_history,
            head_to_head=request.head_to_head,
            as_of=as_of,
            settings=settings,
        )

        form_one = calculate_form(request.player_one_history, as_of, settings)
        form_two = calculate_form(request.player_two_history, as_of, settings)
        momentum = calculate_momentum(
            request.point_progression,
            request.player_one_history,
            request.player_two_history,
            as_of,
            settings,
        )

        features = build_match_features(request, form_one, form_two, momentum, settings)

        probability = calculate_probability(features, quality, settings)
        confidence = calculate_confidence(features, quality, probability, momentum.confidence, settings)
        risk = calculate_risk(features, quality, probability, request.odds, momentum.confidence, settings)
        value = assess_value(
            match.player_one.id,
            match.player_two.id,
            request.odds,
            probability.player_one_probability,
            probability.player_two_probability,
            confidence,
            risk,
            settings,
        )

        patterns = [
            *detect_patterns(
                match.player_one.id, "player_one", request.player_one_history, request.point_progression, settings
            ),
            *detect_patterns(
                match.player_two.id, "player_two", request.player_two_history, request.point_progression, settings
            ),
        ]

        explanations = build_explanations(features, probability, confidence, risk, value, quality)

        return MatchAnalysis(
            match_id=match.id,
            generated_at=as_of,
            match_features=features,
            probability=probability,
            confidence=confidence,
            risk=risk,
            value=value,
            patterns=patterns,
            data_quality=quality,
            explanations=explanations,
        )
