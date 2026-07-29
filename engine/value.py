"""Market value assessment: model probability vs. bookmaker decimal odds.

Pure, independently testable math functions (`fair_odds`,
`implied_probability`, `expected_value`) feed a gated decision
classification. A positive edge alone is never enough to call something
actionable — confidence and risk must also clear configured thresholds.
No stake sizing, no automated wagering.
"""

from __future__ import annotations

from config.analytics import AnalyticsSettings, get_analytics_settings
from domain.odds import Odds
from engine.models import ConfidenceAssessment, PlayerValueAssessment, RiskAssessment, ValueAssessment, ValueDecision
from engine.risk import market_disagreement


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fair_odds(model_probability: float) -> float:
    """The decimal odds that would make the model's probability exactly fair."""
    if not (0.0 < model_probability < 1.0):
        raise ValueError(f"model_probability must be in (0, 1), got {model_probability}")
    return 1.0 / model_probability


def implied_probability(decimal_odds: float) -> float:
    """The bookmaker's implied win probability from decimal odds."""
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal_odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def expected_value(model_probability: float, decimal_odds: float) -> float:
    """Expected value of a unit stake: model_probability * decimal_odds - 1."""
    return model_probability * decimal_odds - 1.0


def _value_score(edge: float, ev: float, confidence: float) -> float:
    edge_component = _clamp(edge / 0.15, 0.0, 1.0)
    ev_component = _clamp(ev / 0.30, 0.0, 1.0)
    combined = 0.5 * edge_component + 0.5 * ev_component
    return _clamp(combined * confidence * 100.0, 0.0, 100.0)


def classify_value(
    edge: float,
    ev: float,
    confidence_score: float,
    risk_score: float,
    settings: AnalyticsSettings,
) -> ValueDecision:
    """Gate a positive edge/EV through confidence and risk thresholds before calling it a signal."""
    meets_edge = edge >= settings.min_probability_edge
    meets_ev = ev >= settings.min_expected_value
    if not (meets_edge and meets_ev):
        return ValueDecision.NO_SIGNAL

    meets_confidence = confidence_score >= settings.min_confidence_for_value
    meets_risk = risk_score <= settings.max_risk_for_value
    if not (meets_confidence and meets_risk):
        return ValueDecision.OBSERVE

    is_strong = edge >= settings.min_probability_edge * 2 and ev >= settings.min_expected_value * 2
    return ValueDecision.STRONG_VALUE if is_strong else ValueDecision.POSSIBLE_VALUE


def _pick_primary_odds(odds: list[Odds]) -> Odds:
    return sorted(odds, key=lambda o: (-o.captured_at.timestamp(), o.bookmaker))[0]


def _assess_player_value(
    player_id: str,
    decimal_odds: float,
    bookmaker: str,
    model_probability: float,
    confidence_score: float,
    risk_score: float,
    settings: AnalyticsSettings,
) -> PlayerValueAssessment:
    implied = implied_probability(decimal_odds)
    edge = model_probability - implied
    ev = expected_value(model_probability, decimal_odds)
    score = _value_score(edge, ev, confidence_score)
    decision = classify_value(edge, ev, confidence_score, risk_score, settings)

    return PlayerValueAssessment(
        player_id=player_id,
        bookmaker=bookmaker,
        decimal_odds=decimal_odds,
        model_probability=model_probability,
        implied_probability=implied,
        fair_odds=fair_odds(model_probability),
        probability_edge=edge,
        expected_value=ev,
        value_score=score,
        decision=decision,
    )


def assess_value(
    player_one_id: str,
    player_two_id: str,
    odds: list[Odds],
    player_one_probability: float,
    player_two_probability: float,
    confidence: ConfidenceAssessment,
    risk: RiskAssessment,
    settings: AnalyticsSettings | None = None,
) -> ValueAssessment:
    """Assess market value for both players' sides of the match, if odds are available."""
    settings = settings or get_analytics_settings()

    if not odds:
        return ValueAssessment(player_one=None, player_two=None, odds_considered=0, market_disagreement=None)

    primary = _pick_primary_odds(odds)

    player_one_assessment = _assess_player_value(
        player_one_id, primary.player_one_odds, primary.bookmaker,
        player_one_probability, confidence.score, risk.score, settings,
    )
    player_two_assessment = _assess_player_value(
        player_two_id, primary.player_two_odds, primary.bookmaker,
        player_two_probability, confidence.score, risk.score, settings,
    )

    return ValueAssessment(
        player_one=player_one_assessment,
        player_two=player_two_assessment,
        odds_considered=len(odds),
        market_disagreement=market_disagreement(odds),
    )
