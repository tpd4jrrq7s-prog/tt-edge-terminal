"""Tests for the value engine: fair odds, implied probability, EV, and gated decisions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from config.analytics import AnalyticsSettings
from domain.odds import Odds
from engine.models import ConfidenceAssessment, ConfidenceLabel, RiskAssessment, RiskLabel, ValueDecision
from engine.value import assess_value, classify_value, expected_value, fair_odds, implied_probability

AS_OF = datetime(2026, 7, 29, tzinfo=timezone.utc)


def test_fair_odds_is_inverse_of_probability():
    assert fair_odds(0.5) == pytest.approx(2.0)
    assert fair_odds(0.25) == pytest.approx(4.0)


def test_fair_odds_rejects_out_of_range_probability():
    with pytest.raises(ValueError):
        fair_odds(0.0)
    with pytest.raises(ValueError):
        fair_odds(1.0)


def test_implied_probability_is_inverse_of_odds():
    assert implied_probability(2.0) == pytest.approx(0.5)
    assert implied_probability(4.0) == pytest.approx(0.25)


def test_implied_probability_rejects_invalid_odds():
    with pytest.raises(ValueError):
        implied_probability(1.0)


def test_expected_value_formula():
    assert expected_value(0.6, 2.0) == pytest.approx(0.2)
    assert expected_value(0.4, 2.0) == pytest.approx(-0.2)


def _confidence(score=0.8) -> ConfidenceAssessment:
    return ConfidenceAssessment(score=score, label=ConfidenceLabel.HIGH, reasons=[])


def _risk(score=20.0) -> RiskAssessment:
    return RiskAssessment(score=score, label=RiskLabel.LOW, factors=[])


def test_classify_value_no_signal_below_edge_threshold():
    settings = AnalyticsSettings()
    decision = classify_value(edge=0.01, ev=0.01, confidence_score=0.9, risk_score=10.0, settings=settings)
    assert decision is ValueDecision.NO_SIGNAL


def test_classify_value_observe_when_gates_fail():
    settings = AnalyticsSettings()
    decision = classify_value(edge=0.10, ev=0.10, confidence_score=0.1, risk_score=95.0, settings=settings)
    assert decision is ValueDecision.OBSERVE


def test_classify_value_possible_value_when_gates_pass():
    settings = AnalyticsSettings()
    decision = classify_value(edge=0.04, ev=0.03, confidence_score=0.9, risk_score=10.0, settings=settings)
    assert decision is ValueDecision.POSSIBLE_VALUE


def test_classify_value_strong_value_for_large_edge():
    settings = AnalyticsSettings()
    decision = classify_value(edge=0.20, ev=0.20, confidence_score=0.9, risk_score=10.0, settings=settings)
    assert decision is ValueDecision.STRONG_VALUE


def test_assess_value_with_no_odds_returns_empty_assessment():
    result = assess_value("p1", "p2", [], 0.6, 0.4, _confidence(), _risk(), AnalyticsSettings())
    assert result.player_one is None
    assert result.player_two is None
    assert result.odds_considered == 0
    assert result.market_disagreement is None


def test_assess_value_computes_both_sides():
    odds = [Odds(match_id="m1", bookmaker="Pinnacle", player_one_odds=2.2, player_two_odds=1.7, captured_at=AS_OF)]
    result = assess_value("p1", "p2", odds, 0.55, 0.45, _confidence(), _risk(), AnalyticsSettings())
    assert result.player_one is not None
    assert result.player_two is not None
    assert result.player_one.decimal_odds == 2.2
    assert result.player_two.decimal_odds == 1.7
    assert result.odds_considered == 1


def test_assess_value_picks_most_recent_odds():
    from datetime import timedelta

    older = Odds(match_id="m1", bookmaker="A", player_one_odds=1.5, player_two_odds=2.5, captured_at=AS_OF - timedelta(hours=1))
    newer = Odds(match_id="m1", bookmaker="B", player_one_odds=1.9, player_two_odds=1.9, captured_at=AS_OF)
    result = assess_value("p1", "p2", [older, newer], 0.5, 0.5, _confidence(), _risk(), AnalyticsSettings())
    assert result.player_one.bookmaker == "B"
    assert result.player_one.decimal_odds == 1.9


def test_positive_edge_produces_positive_expected_value_sign_alignment():
    odds = [Odds(match_id="m1", bookmaker="Pinnacle", player_one_odds=3.0, player_two_odds=1.4, captured_at=AS_OF)]
    result = assess_value("p1", "p2", odds, 0.5, 0.5, _confidence(), _risk(), AnalyticsSettings())
    assert result.player_one.probability_edge > 0
    assert result.player_one.expected_value > 0


def test_value_is_deterministic():
    odds = [Odds(match_id="m1", bookmaker="Pinnacle", player_one_odds=2.0, player_two_odds=1.9, captured_at=AS_OF)]
    result_one = assess_value("p1", "p2", odds, 0.55, 0.45, _confidence(), _risk(), AnalyticsSettings())
    result_two = assess_value("p1", "p2", odds, 0.55, 0.45, _confidence(), _risk(), AnalyticsSettings())
    assert result_one == result_two
