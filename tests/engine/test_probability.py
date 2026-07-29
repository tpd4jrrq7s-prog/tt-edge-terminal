"""Tests for the probability engine."""

from __future__ import annotations

from config.analytics import AnalyticsSettings
from engine.models import DataQualityAssessment, MomentumState, PlayerMetrics
from engine.probability import calculate_probability, stable_sigmoid


def _features(form_diff=0.0, ranking_diff=None, momentum_diff=0.0, h2h_signal=0.0, match_state_signal=0.0):
    from engine.models import MatchFeatures

    player_one = PlayerMetrics(
        player_id="p1", player_name="Player One", form_score=50.0, form_confidence=0.5,
        matches_considered=3, momentum_score=50.0, momentum_state=MomentumState.NO_DATA,
    )
    player_two = PlayerMetrics(
        player_id="p2", player_name="Player Two", form_score=50.0, form_confidence=0.5,
        matches_considered=3, momentum_score=50.0, momentum_state=MomentumState.NO_DATA,
    )
    return MatchFeatures(
        player_one=player_one,
        player_two=player_two,
        ranking_differential=ranking_diff,
        form_differential=form_diff,
        momentum_differential=momentum_diff,
        head_to_head=None,
        head_to_head_signal=h2h_signal,
        match_state_signal=match_state_signal,
    )


def _quality(score=100.0) -> DataQualityAssessment:
    return DataQualityAssessment(
        score=score, warnings=[], history_sample_size_player_one=3,
        history_sample_size_player_two=3, odds_available=True, odds_fresh=True,
    )


def test_neutral_features_produce_probabilities_near_half():
    result = calculate_probability(_features(), _quality(), AnalyticsSettings())
    assert abs(result.player_one_probability - 0.5) < 0.01


def test_probabilities_sum_to_one():
    result = calculate_probability(_features(form_diff=0.4), _quality(), AnalyticsSettings())
    assert abs((result.player_one_probability + result.player_two_probability) - 1.0) < 1e-9


def test_probabilities_are_strictly_within_open_interval():
    result = calculate_probability(_features(form_diff=1.0, momentum_diff=1.0, h2h_signal=1.0, match_state_signal=1.0), _quality(), AnalyticsSettings())
    assert 0.0 < result.player_one_probability < 1.0
    assert 0.0 < result.player_two_probability < 1.0


def test_positive_form_differential_favors_player_one():
    result = calculate_probability(_features(form_diff=0.5), _quality(), AnalyticsSettings())
    assert result.player_one_probability > 0.5


def test_negative_form_differential_favors_player_two():
    result = calculate_probability(_features(form_diff=-0.5), _quality(), AnalyticsSettings())
    assert result.player_one_probability < 0.5


def test_low_data_quality_shrinks_probability_toward_half():
    high_quality = calculate_probability(_features(form_diff=0.6), _quality(100.0), AnalyticsSettings())
    low_quality = calculate_probability(_features(form_diff=0.6), _quality(10.0), AnalyticsSettings())
    assert abs(low_quality.player_one_probability - 0.5) < abs(high_quality.player_one_probability - 0.5)


def test_data_quality_penalty_reflects_quality_score():
    result = calculate_probability(_features(), _quality(80.0), AnalyticsSettings())
    assert abs(result.data_quality_penalty - 0.2) < 1e-9


def test_factors_include_all_named_weights():
    result = calculate_probability(_features(form_diff=0.3), _quality(), AnalyticsSettings())
    names = {f.name for f in result.factors}
    assert names == {"form", "ranking", "momentum", "head_to_head", "match_state", "context"}


def test_context_factor_is_always_neutral():
    result = calculate_probability(_features(), _quality(), AnalyticsSettings())
    context_factor = next(f for f in result.factors if f.name == "context")
    assert context_factor.raw_signal == 0.0


def test_missing_ranking_gives_zero_raw_signal():
    result = calculate_probability(_features(ranking_diff=None), _quality(), AnalyticsSettings())
    ranking_factor = next(f for f in result.factors if f.name == "ranking")
    assert ranking_factor.raw_signal == 0.0


def test_calculate_probability_is_deterministic():
    features = _features(form_diff=0.3, momentum_diff=-0.2)
    result_one = calculate_probability(features, _quality(), AnalyticsSettings())
    result_two = calculate_probability(features, _quality(), AnalyticsSettings())
    assert result_one == result_two


def test_stable_sigmoid_bounds_and_midpoint():
    assert stable_sigmoid(0.0) == 0.5
    assert 0.0 < stable_sigmoid(-20.0) < 0.01
    assert 0.99 < stable_sigmoid(20.0) < 1.0
