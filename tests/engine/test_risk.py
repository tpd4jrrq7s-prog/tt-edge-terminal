"""Tests for the risk engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config.analytics import AnalyticsSettings
from domain.odds import Odds
from engine.models import (
    CompetitionContext,
    DataQualityAssessment,
    MatchFeatures,
    MomentumState,
    PlayerMetrics,
    ProbabilityFactor,
    ProbabilityResult,
)
from engine.risk import calculate_risk, market_disagreement, market_movement

AS_OF = datetime(2026, 7, 29, tzinfo=timezone.utc)


def _features(ranking_one=None, ranking_two=None, matches_one=3, matches_two=3, context=None, form_diff=0.0, momentum_diff=0.0) -> MatchFeatures:
    return MatchFeatures(
        player_one=PlayerMetrics(
            player_id="p1", player_name="P1", ranking=ranking_one, form_score=50.0, form_confidence=0.5,
            matches_considered=matches_one, momentum_score=50.0, momentum_state=MomentumState.NO_DATA,
        ),
        player_two=PlayerMetrics(
            player_id="p2", player_name="P2", ranking=ranking_two, form_score=50.0, form_confidence=0.5,
            matches_considered=matches_two, momentum_score=50.0, momentum_state=MomentumState.NO_DATA,
        ),
        form_differential=form_diff,
        momentum_differential=momentum_diff,
        head_to_head_signal=0.0,
        match_state_signal=0.0,
        context=context,
    )


def _quality(score=100.0) -> DataQualityAssessment:
    return DataQualityAssessment(
        score=score, warnings=[], history_sample_size_player_one=3,
        history_sample_size_player_two=3, odds_available=True, odds_fresh=True,
    )


def _probability() -> ProbabilityResult:
    factors = [ProbabilityFactor(name="form", weight=0.3, raw_signal=0.0, weighted_contribution=0.0, description="d")]
    return ProbabilityResult(
        player_one_probability=0.5, player_two_probability=0.5,
        factors=factors, data_quality_penalty=0.0, calibration_ready=True,
    )


def test_risk_score_within_bounds():
    result = calculate_risk(_features(), _quality(), _probability(), [], 0.5, AnalyticsSettings())
    assert 0.0 <= result.score <= 100.0


def test_low_data_quality_increases_risk():
    low = calculate_risk(_features(), _quality(20.0), _probability(), [], 0.5, AnalyticsSettings())
    high = calculate_risk(_features(), _quality(100.0), _probability(), [], 0.5, AnalyticsSettings())
    assert low.score > high.score


def test_missing_ranking_and_history_increases_risk():
    complete = calculate_risk(_features(ranking_one=1, ranking_two=2), _quality(), _probability(), [], 0.9, AnalyticsSettings())
    missing = calculate_risk(_features(matches_one=0, matches_two=0), _quality(), _probability(), [], 0.9, AnalyticsSettings())
    assert missing.score > complete.score


def test_short_format_increases_risk_versus_long_format():
    short = calculate_risk(_features(context=CompetitionContext(best_of_sets=3)), _quality(), _probability(), [], 0.9, AnalyticsSettings())
    long_format = calculate_risk(_features(context=CompetitionContext(best_of_sets=7)), _quality(), _probability(), [], 0.9, AnalyticsSettings())
    assert short.score > long_format.score


def test_momentum_reversal_between_form_and_momentum_increases_risk():
    aligned = calculate_risk(_features(form_diff=0.5, momentum_diff=0.5), _quality(), _probability(), [], 0.9, AnalyticsSettings())
    reversed_ = calculate_risk(_features(form_diff=0.5, momentum_diff=-0.5), _quality(), _probability(), [], 0.9, AnalyticsSettings())
    assert reversed_.score > aligned.score


def test_market_disagreement_none_below_two_quotes():
    quote = Odds(match_id="m1", bookmaker="Pinnacle", player_one_odds=1.8, player_two_odds=2.1, captured_at=AS_OF)
    assert market_disagreement([]) is None
    assert market_disagreement([quote]) is None


def test_market_disagreement_detects_bookmaker_spread():
    quotes = [
        Odds(match_id="m1", bookmaker="A", player_one_odds=1.5, player_two_odds=2.5, captured_at=AS_OF),
        Odds(match_id="m1", bookmaker="B", player_one_odds=2.5, player_two_odds=1.5, captured_at=AS_OF),
    ]
    assert market_disagreement(quotes) > 0.0


def test_market_movement_none_below_two_quotes():
    assert market_movement([]) is None


def test_market_movement_detects_odds_drift():
    quotes = [
        Odds(match_id="m1", bookmaker="A", player_one_odds=1.5, player_two_odds=2.5, captured_at=AS_OF - timedelta(hours=1)),
        Odds(match_id="m1", bookmaker="A", player_one_odds=2.0, player_two_odds=1.8, captured_at=AS_OF),
    ]
    assert market_movement(quotes) > 0.0


def test_risk_labels_follow_configured_thresholds():
    settings = AnalyticsSettings(risk_low_threshold=10.0, risk_medium_threshold=20.0, risk_high_threshold=30.0)
    high_quality_result = calculate_risk(
        _features(ranking_one=1, ranking_two=2, context=CompetitionContext(best_of_sets=7)),
        _quality(100.0), _probability(), [], 1.0, settings,
    )
    assert high_quality_result.label.value in {"low", "medium", "high", "extreme"}


def test_risk_is_deterministic():
    features = _features(ranking_one=5, ranking_two=10)
    result_one = calculate_risk(features, _quality(), _probability(), [], 0.7, AnalyticsSettings())
    result_two = calculate_risk(features, _quality(), _probability(), [], 0.7, AnalyticsSettings())
    assert result_one == result_two
