"""Transparent, rules-based win-probability model (not yet machine learning).

Combines standardized feature differentials (each in [-1, 1]) into a
single logit via configurable weights, applies a data-quality shrinkage
toward the neutral 50/50 point, and converts to probabilities with a
numerically stable logistic function. Player probabilities always sum
to 1 and are clamped strictly inside (0, 1).
"""

from __future__ import annotations

import math

from config.analytics import AnalyticsSettings, get_analytics_settings
from engine.models import DataQualityAssessment, MatchFeatures, ProbabilityFactor, ProbabilityResult

_PROBABILITY_EPSILON = 1e-4


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def stable_sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0:
        exp_neg_z = math.exp(-z)
        return 1.0 / (1.0 + exp_neg_z)
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def calculate_probability(
    features: MatchFeatures,
    data_quality: DataQualityAssessment,
    settings: AnalyticsSettings | None = None,
) -> ProbabilityResult:
    """Compute player_one/player_two win probabilities from match features."""
    settings = settings or get_analytics_settings()
    weights = settings.probability_weights

    factor_inputs: list[tuple[str, float, float, str]] = [
        (
            "form",
            weights.form,
            features.form_differential,
            "Recent, recency-weighted match performance for each player",
        ),
        (
            "ranking",
            weights.ranking,
            features.ranking_differential if features.ranking_differential is not None else 0.0,
            "World ranking differential (0 if either ranking is unknown)",
        ),
        (
            "momentum",
            weights.momentum,
            features.momentum_differential,
            "Pre-match trend or in-play point/set momentum",
        ),
        (
            "head_to_head",
            weights.head_to_head,
            features.head_to_head_signal,
            "Historical head-to-head win ratio (0 if no record provided)",
        ),
        (
            "match_state",
            weights.match_state,
            features.match_state_signal,
            "Current match set score, if the match is already underway",
        ),
        (
            "context",
            weights.context,
            0.0,
            "Surface/competition context is not yet modeled numerically; always neutral",
        ),
    ]

    factors: list[ProbabilityFactor] = []
    logit = 0.0
    for name, weight, raw_signal, description in factor_inputs:
        raw_signal = _clamp(raw_signal, -1.0, 1.0)
        contribution = weight * raw_signal
        logit += contribution
        factors.append(
            ProbabilityFactor(
                name=name,
                weight=weight,
                raw_signal=raw_signal,
                weighted_contribution=contribution,
                description=description,
            )
        )

    quality_factor = _clamp(data_quality.score / 100.0, 0.0, 1.0)
    data_quality_penalty = 1.0 - quality_factor
    shrunk_logit = logit * quality_factor

    player_one_probability = _clamp(
        stable_sigmoid(shrunk_logit), _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON
    )
    player_two_probability = 1.0 - player_one_probability

    calibration_ready = (
        features.player_one.matches_considered >= settings.min_history_sample_size
        and features.player_two.matches_considered >= settings.min_history_sample_size
        and features.head_to_head is not None
        and features.head_to_head.total_matches > 0
    )

    return ProbabilityResult(
        player_one_probability=player_one_probability,
        player_two_probability=player_two_probability,
        factors=factors,
        data_quality_penalty=data_quality_penalty,
        calibration_ready=calibration_ready,
    )
