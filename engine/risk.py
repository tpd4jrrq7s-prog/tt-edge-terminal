"""Risk assessment (0-100) with a low/medium/high/extreme classification.

Combines eight independently computed, configurably weighted risk
components. Every component is derived only from data actually present
in the request — an absent signal contributes to `missing_data`, never
to a fabricated low-risk reading.
"""

from __future__ import annotations

import statistics

from config.analytics import AnalyticsSettings, get_analytics_settings
from domain.odds import Odds
from engine.models import DataQualityAssessment, MatchFeatures, ProbabilityResult, RiskAssessment, RiskFactor, RiskLabel


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def market_disagreement(odds: list[Odds]) -> float | None:
    """Population standard deviation of decimal odds across bookmakers, or None if <2 quotes."""
    if len(odds) < 2:
        return None
    one_stdev = statistics.pstdev(o.player_one_odds for o in odds)
    two_stdev = statistics.pstdev(o.player_two_odds for o in odds)
    return max(one_stdev, two_stdev)


def market_movement(odds: list[Odds]) -> float | None:
    """Relative change in player_one decimal odds between the earliest and latest quote."""
    if len(odds) < 2:
        return None
    ordered = sorted(odds, key=lambda o: o.captured_at)
    earliest, latest = ordered[0], ordered[-1]
    if earliest.player_one_odds == 0:
        return None
    return abs(latest.player_one_odds - earliest.player_one_odds) / earliest.player_one_odds


def _conflicting_signals_severity(probability: ProbabilityResult) -> float:
    considered = [f for f in probability.factors if abs(f.weighted_contribution) > 1e-9]
    if not considered:
        return 0.0
    overall_favors_one = probability.player_one_probability >= 0.5
    disagreeing = sum(1 for f in considered if (f.weighted_contribution > 0) != overall_favors_one)
    return 100.0 * disagreeing / len(considered)


def calculate_risk(
    features: MatchFeatures,
    data_quality: DataQualityAssessment,
    probability: ProbabilityResult,
    odds: list[Odds],
    momentum_confidence: float,
    settings: AnalyticsSettings | None = None,
) -> RiskAssessment:
    """Compute the overall risk score, label, and contributing factors."""
    settings = settings or get_analytics_settings()
    weights = settings.risk_weights

    data_quality_severity = 100.0 - data_quality.score
    conflicting_signals_severity = _conflicting_signals_severity(probability)
    volatility_severity = 100.0 * (1.0 - _clamp(momentum_confidence, 0.0, 1.0))
    momentum_reversal_severity = _clamp(
        100.0 * abs(features.momentum_differential - features.form_differential) / 2.0, 0.0, 100.0
    )

    disagreement = market_disagreement(odds)
    market_disagreement_severity = _clamp((disagreement or 0.0) / 0.5 * 100.0, 0.0, 100.0) if disagreement is not None else 0.0

    movement = market_movement(odds)
    market_movement_severity = _clamp((movement or 0.0) * 200.0, 0.0, 100.0) if movement is not None else 0.0

    missing_signals = [
        features.player_one.ranking is None,
        features.player_two.ranking is None,
        features.head_to_head is None,
        features.player_one.matches_considered == 0,
        features.player_two.matches_considered == 0,
    ]
    missing_data_severity = 100.0 * sum(missing_signals) / len(missing_signals)

    best_of = features.context.best_of_sets if features.context else None
    short_format_severity = _clamp((7 - best_of) / 6.0 * 100.0, 0.0, 100.0) if best_of else 0.0

    components: list[tuple[str, float, str]] = [
        ("data_quality", data_quality_severity, f"Data quality score is {data_quality.score:.1f}/100"),
        (
            "conflicting_signals",
            conflicting_signals_severity,
            "Fraction of probability factors disagreeing with the overall favorite",
        ),
        ("volatility", volatility_severity, f"Momentum sample confidence is {momentum_confidence:.2f}"),
        (
            "momentum_reversal",
            momentum_reversal_severity,
            "Gap between short-term momentum and longer-term form differentials",
        ),
        (
            "market_disagreement",
            market_disagreement_severity,
            f"Bookmaker odds standard deviation is {disagreement:.3f}" if disagreement is not None else "No multi-bookmaker odds available",
        ),
        (
            "market_movement",
            market_movement_severity,
            f"Odds moved {movement * 100:.1f}% between earliest and latest quote" if movement is not None else "Not enough odds history to detect movement",
        ),
        (
            "missing_data",
            missing_data_severity,
            f"{sum(missing_signals)}/{len(missing_signals)} optional data point(s) are missing",
        ),
        (
            "short_format",
            short_format_severity,
            f"Best-of-{best_of} format" if best_of else "Match format/best-of not provided",
        ),
    ]

    weight_map = weights.model_dump()
    score = _clamp(sum(weight_map[name] * severity for name, severity, _ in components), 0.0, 100.0)
    factors = [RiskFactor(name=name, severity=_clamp(severity, 0.0, 100.0), detail=detail) for name, severity, detail in components]

    if score < settings.risk_low_threshold:
        label = RiskLabel.LOW
    elif score < settings.risk_medium_threshold:
        label = RiskLabel.MEDIUM
    elif score < settings.risk_high_threshold:
        label = RiskLabel.HIGH
    else:
        label = RiskLabel.EXTREME

    return RiskAssessment(score=score, label=label, factors=factors)
