"""Deterministic, human-readable explanations generated only from computed signals.

No LLM and no unsupported natural-language claims: every sentence here
is built directly from a field on one of the typed result models.
"""

from __future__ import annotations

from engine.models import (
    ConfidenceAssessment,
    DataQualityAssessment,
    MatchFeatures,
    ProbabilityResult,
    RiskAssessment,
    ValueAssessment,
)


def explain_probability(features: MatchFeatures, probability: ProbabilityResult) -> list[str]:
    lines: list[str] = []
    one, two = probability.player_one_probability, probability.player_two_probability
    favored, favored_prob, other, other_prob = (
        (features.player_one.player_name, one, features.player_two.player_name, two)
        if one >= two
        else (features.player_two.player_name, two, features.player_one.player_name, one)
    )
    lines.append(
        f"{favored} is favored at {favored_prob * 100:.1f}% versus {other_prob * 100:.1f}% for {other}."
    )

    ranked_factors = sorted(probability.factors, key=lambda f: abs(f.weighted_contribution), reverse=True)
    top_factors = [f for f in ranked_factors if abs(f.weighted_contribution) > 1e-9][:2]
    if top_factors:
        parts = [f"{f.name} ({f.description.lower()})" for f in top_factors]
        lines.append("The largest contributing factor(s): " + "; ".join(parts) + ".")
    else:
        lines.append("No individual factor contributed meaningfully; the probability is close to neutral.")
    return lines


def explain_confidence(confidence: ConfidenceAssessment) -> list[str]:
    lines = [f"Confidence is {confidence.label.value} ({confidence.score:.2f} on a 0-1 scale)."]
    increasing = [r.detail for r in confidence.reasons if r.direction == "increases"]
    decreasing = [r.detail for r in confidence.reasons if r.direction == "decreases"]
    if increasing:
        lines.append("Increasing confidence: " + "; ".join(increasing) + ".")
    if decreasing:
        lines.append("Decreasing confidence: " + "; ".join(decreasing) + ".")
    return lines


def explain_risk(risk: RiskAssessment) -> list[str]:
    lines = [f"Risk is classified as {risk.label.value} ({risk.score:.1f} on a 0-100 scale)."]
    top_factors = sorted(risk.factors, key=lambda f: f.severity, reverse=True)[:3]
    if top_factors:
        parts = [f"{f.name} ({f.severity:.0f}/100: {f.detail})" for f in top_factors]
        lines.append("Largest risk contributors: " + "; ".join(parts) + ".")
    lines.append("This is a risk classification for decision support, not a certainty.")
    return lines


def explain_value(value: ValueAssessment) -> list[str]:
    if value.odds_considered == 0:
        return ["No bookmaker odds were provided, so no market value assessment could be made."]

    lines: list[str] = []
    for assessment in (value.player_one, value.player_two):
        if assessment is None:
            continue
        direction = "positive" if assessment.probability_edge > 0 else "negative"
        lines.append(
            f"{assessment.player_id}: model probability {assessment.model_probability * 100:.1f}% vs "
            f"implied {assessment.implied_probability * 100:.1f}% at {assessment.decimal_odds:.2f} "
            f"decimal odds from {assessment.bookmaker} — {direction} edge of "
            f"{assessment.probability_edge * 100:.1f} points, expected value {assessment.expected_value:+.3f}, "
            f"decision: {assessment.decision.value}."
        )
    if value.market_disagreement is not None:
        lines.append(f"Bookmakers disagree by a standard deviation of {value.market_disagreement:.3f} in decimal odds.")
    return lines


def explain_data_quality(quality: DataQualityAssessment) -> list[str]:
    lines = [f"Data quality score is {quality.score:.1f}/100."]
    if quality.warnings:
        details = [f"{w.severity.value}: {w.detail}" for w in quality.warnings]
        lines.append("Data quality issues: " + "; ".join(details) + ".")
    else:
        lines.append("No data quality issues were detected.")
    return lines


def build_explanations(
    features: MatchFeatures,
    probability: ProbabilityResult,
    confidence: ConfidenceAssessment,
    risk: RiskAssessment,
    value: ValueAssessment,
    quality: DataQualityAssessment,
) -> list[str]:
    """Combine every section's explanation lines into one ordered, deterministic list."""
    return [
        *explain_probability(features, probability),
        *explain_confidence(confidence),
        *explain_risk(risk),
        *explain_value(value),
        *explain_data_quality(quality),
    ]
