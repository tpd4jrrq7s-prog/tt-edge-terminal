"""Tests for the deterministic explanation engine."""

from __future__ import annotations

from config.analytics import AnalyticsSettings
from engine.explanations import (
    build_explanations,
    explain_confidence,
    explain_data_quality,
    explain_probability,
    explain_risk,
    explain_value,
)
from engine.models import (
    ConfidenceAssessment,
    ConfidenceLabel,
    ConfidenceReason,
    DataQualityAssessment,
    DataQualityIssue,
    IssueSeverity,
    MatchFeatures,
    MomentumState,
    PlayerMetrics,
    ProbabilityFactor,
    ProbabilityResult,
    RiskAssessment,
    RiskFactor,
    RiskLabel,
    ValueAssessment,
)


def _features() -> MatchFeatures:
    return MatchFeatures(
        player_one=PlayerMetrics(
            player_id="p1", player_name="Ma Long", form_score=70.0, form_confidence=0.8,
            matches_considered=5, momentum_score=60.0, momentum_state=MomentumState.PRE_MATCH,
        ),
        player_two=PlayerMetrics(
            player_id="p2", player_name="Fan Zhendong", form_score=50.0, form_confidence=0.5,
            matches_considered=3, momentum_score=50.0, momentum_state=MomentumState.PRE_MATCH,
        ),
        form_differential=0.2,
        momentum_differential=0.1,
        head_to_head_signal=0.0,
        match_state_signal=0.0,
    )


def _probability() -> ProbabilityResult:
    return ProbabilityResult(
        player_one_probability=0.65,
        player_two_probability=0.35,
        factors=[
            ProbabilityFactor(name="form", weight=0.3, raw_signal=0.2, weighted_contribution=0.06, description="Form"),
            ProbabilityFactor(name="ranking", weight=0.25, raw_signal=0.0, weighted_contribution=0.0, description="Ranking"),
        ],
        data_quality_penalty=0.0,
        calibration_ready=True,
    )


def test_explain_probability_names_the_favored_player():
    lines = explain_probability(_features(), _probability())
    assert any("Ma Long is favored" in line for line in lines)


def test_explain_confidence_reports_label():
    confidence = ConfidenceAssessment(
        score=0.75, label=ConfidenceLabel.HIGH,
        reasons=[ConfidenceReason(factor="sample_size", direction="increases", detail="enough data")],
    )
    lines = explain_confidence(confidence)
    assert any("high" in line for line in lines)
    assert any("Increasing confidence" in line for line in lines)


def test_explain_risk_flags_it_as_not_certainty():
    risk = RiskAssessment(score=15.0, label=RiskLabel.LOW, factors=[RiskFactor(name="data_quality", severity=10.0, detail="fine")])
    lines = explain_risk(risk)
    assert any("not a certainty" in line for line in lines)


def test_explain_value_reports_no_odds():
    lines = explain_value(ValueAssessment(player_one=None, player_two=None, odds_considered=0, market_disagreement=None))
    assert any("No bookmaker odds" in line for line in lines)


def test_explain_data_quality_lists_warnings():
    quality = DataQualityAssessment(
        score=80.0,
        warnings=[DataQualityIssue(field="odds", detail="stale odds", severity=IssueSeverity.INFO)],
        history_sample_size_player_one=3, history_sample_size_player_two=3,
        odds_available=True, odds_fresh=False,
    )
    lines = explain_data_quality(quality)
    assert any("stale odds" in line for line in lines)


def test_explanations_are_deterministic():
    quality = DataQualityAssessment(
        score=90.0, warnings=[], history_sample_size_player_one=5,
        history_sample_size_player_two=3, odds_available=False, odds_fresh=False,
    )
    confidence = ConfidenceAssessment(score=0.6, label=ConfidenceLabel.MEDIUM, reasons=[])
    risk = RiskAssessment(score=30.0, label=RiskLabel.MEDIUM, factors=[])
    value = ValueAssessment(player_one=None, player_two=None, odds_considered=0, market_disagreement=None)

    lines_one = build_explanations(_features(), _probability(), confidence, risk, value, quality)
    lines_two = build_explanations(_features(), _probability(), confidence, risk, value, quality)
    assert lines_one == lines_two
    assert all(isinstance(line, str) for line in lines_one)
