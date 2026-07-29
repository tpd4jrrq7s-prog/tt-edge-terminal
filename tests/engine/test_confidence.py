"""Tests for the confidence engine, and its distinction from probability."""

from __future__ import annotations

from config.analytics import AnalyticsSettings
from engine.confidence import calculate_confidence
from engine.models import (
    DataQualityAssessment,
    MatchFeatures,
    MomentumState,
    PlayerMetrics,
    ProbabilityFactor,
    ProbabilityResult,
)


def _features(form_confidence_one=0.5, form_confidence_two=0.5) -> MatchFeatures:
    return MatchFeatures(
        player_one=PlayerMetrics(
            player_id="p1", player_name="P1", form_score=50.0, form_confidence=form_confidence_one,
            matches_considered=3, momentum_score=50.0, momentum_state=MomentumState.NO_DATA,
        ),
        player_two=PlayerMetrics(
            player_id="p2", player_name="P2", form_score=50.0, form_confidence=form_confidence_two,
            matches_considered=3, momentum_score=50.0, momentum_state=MomentumState.NO_DATA,
        ),
        form_differential=0.0,
        momentum_differential=0.0,
        head_to_head_signal=0.0,
        match_state_signal=0.0,
    )


def _quality(score=100.0) -> DataQualityAssessment:
    return DataQualityAssessment(
        score=score, warnings=[], history_sample_size_player_one=3,
        history_sample_size_player_two=3, odds_available=True, odds_fresh=True,
    )


def _probability(calibration_ready=True, agree=True) -> ProbabilityResult:
    sign = 1 if agree else -1
    factors = [
        ProbabilityFactor(name="form", weight=0.3, raw_signal=0.5, weighted_contribution=0.15, description="d"),
        ProbabilityFactor(name="ranking", weight=0.25, raw_signal=0.5 * sign, weighted_contribution=0.1 * sign, description="d"),
    ]
    return ProbabilityResult(
        player_one_probability=0.7, player_two_probability=0.3,
        factors=factors, data_quality_penalty=0.0, calibration_ready=calibration_ready,
    )


def test_confidence_score_within_bounds():
    result = calculate_confidence(_features(), _quality(), _probability(), momentum_confidence=0.5, settings=AnalyticsSettings())
    assert 0.0 <= result.score <= 1.0


def test_confidence_is_independent_of_probability_value():
    # A high-probability result and a near-even one can carry identical confidence
    # given identical supporting data quality/sample size/agreement.
    high_prob = ProbabilityResult(
        player_one_probability=0.95, player_two_probability=0.05,
        factors=_probability().factors, data_quality_penalty=0.0, calibration_ready=True,
    )
    even_prob = ProbabilityResult(
        player_one_probability=0.51, player_two_probability=0.49,
        factors=_probability().factors, data_quality_penalty=0.0, calibration_ready=True,
    )
    features = _features()
    quality = _quality()
    result_high = calculate_confidence(features, quality, high_prob, 0.8, AnalyticsSettings())
    result_even = calculate_confidence(features, quality, even_prob, 0.8, AnalyticsSettings())
    assert result_high.score == result_even.score


def test_low_data_quality_reduces_confidence():
    high = calculate_confidence(_features(), _quality(100.0), _probability(), 0.8, AnalyticsSettings())
    low = calculate_confidence(_features(), _quality(20.0), _probability(), 0.8, AnalyticsSettings())
    assert low.score < high.score


def test_conflicting_signals_reduce_confidence():
    agreeing = calculate_confidence(_features(), _quality(), _probability(agree=True), 0.8, AnalyticsSettings())
    conflicting = calculate_confidence(_features(), _quality(), _probability(agree=False), 0.8, AnalyticsSettings())
    assert conflicting.score < agreeing.score


def test_calibration_not_ready_reduces_confidence():
    ready = calculate_confidence(_features(), _quality(), _probability(calibration_ready=True), 0.8, AnalyticsSettings())
    not_ready = calculate_confidence(_features(), _quality(), _probability(calibration_ready=False), 0.8, AnalyticsSettings())
    assert not_ready.score < ready.score


def test_reasons_are_labeled_increases_or_decreases():
    result = calculate_confidence(_features(), _quality(20.0), _probability(calibration_ready=False), 0.1, AnalyticsSettings())
    directions = {r.direction for r in result.reasons}
    assert directions <= {"increases", "decreases"}
    assert len(result.reasons) == 5


def test_confidence_label_thresholds():
    settings = AnalyticsSettings()
    low = calculate_confidence(_features(0.0, 0.0), _quality(0.0), _probability(calibration_ready=False), 0.0, settings)
    assert low.label.value == "low"

    high = calculate_confidence(_features(1.0, 1.0), _quality(100.0), _probability(calibration_ready=True), 1.0, settings)
    assert high.label.value in {"high", "very_high"}
