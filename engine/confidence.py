"""Confidence assessment — separate from probability.

A high win probability can still carry low confidence if it rests on
thin data. Confidence blends sample size, data completeness, agreement
between the probability engine's factors, calibration readiness, and
momentum sample confidence into a single 0-1 score with named reasons.
"""

from __future__ import annotations

from config.analytics import AnalyticsSettings, get_analytics_settings
from engine.models import ConfidenceAssessment, ConfidenceLabel, ConfidenceReason, DataQualityAssessment, MatchFeatures, ProbabilityResult

_COMPONENT_WEIGHTS = {
    "sample_size": 0.25,
    "data_completeness": 0.20,
    "signal_agreement": 0.20,
    "calibration_readiness": 0.15,
    "momentum_sample": 0.20,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _signal_agreement(probability: ProbabilityResult) -> float:
    """Fraction of non-neutral probability factors agreeing with the overall favorite."""
    overall_favors_one = probability.player_one_probability >= 0.5
    considered = [f for f in probability.factors if abs(f.weighted_contribution) > 1e-9]
    if not considered:
        return 0.5
    agreeing = sum(
        1
        for f in considered
        if (f.weighted_contribution > 0) == overall_favors_one
    )
    return agreeing / len(considered)


def _label_for(score: float, settings: AnalyticsSettings) -> ConfidenceLabel:
    if score < settings.confidence_low_threshold:
        return ConfidenceLabel.LOW
    if score < settings.confidence_medium_threshold:
        return ConfidenceLabel.MEDIUM
    if score < settings.confidence_high_threshold:
        return ConfidenceLabel.HIGH
    return ConfidenceLabel.VERY_HIGH


def calculate_confidence(
    features: MatchFeatures,
    data_quality: DataQualityAssessment,
    probability: ProbabilityResult,
    momentum_confidence: float,
    settings: AnalyticsSettings | None = None,
) -> ConfidenceAssessment:
    """Compute an overall confidence score, independent of the probability value."""
    settings = settings or get_analytics_settings()

    sample_size_component = (features.player_one.form_confidence + features.player_two.form_confidence) / 2.0
    data_completeness_component = data_quality.score / 100.0
    agreement_component = _signal_agreement(probability)
    calibration_component = 1.0 if probability.calibration_ready else 0.3
    momentum_component = _clamp(momentum_confidence, 0.0, 1.0)

    components = {
        "sample_size": sample_size_component,
        "data_completeness": data_completeness_component,
        "signal_agreement": agreement_component,
        "calibration_readiness": calibration_component,
        "momentum_sample": momentum_component,
    }

    score = _clamp(
        sum(components[name] * weight for name, weight in _COMPONENT_WEIGHTS.items()),
        0.0,
        1.0,
    )

    descriptions = {
        "sample_size": "historical sample size behind each player's form score",
        "data_completeness": "overall completeness of the supplied match data",
        "signal_agreement": "agreement between the probability engine's contributing factors",
        "calibration_readiness": "whether enough history/head-to-head data exists to trust calibration",
        "momentum_sample": "sample size behind the momentum signal",
    }
    reasons = [
        ConfidenceReason(
            factor=name,
            direction="increases" if value >= 0.5 else "decreases",
            detail=f"{descriptions[name]} ({value:.2f})",
        )
        for name, value in components.items()
    ]

    return ConfidenceAssessment(score=score, label=_label_for(score, settings), reasons=reasons)
