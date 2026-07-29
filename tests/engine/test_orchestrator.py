"""Integration tests for the MatchAnalyticsEngine orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone

from config.analytics import AnalyticsSettings
from domain.match import Match, MatchStatus
from domain.odds import Odds
from domain.player import Player
from engine.models import (
    CompetitionContext,
    HeadToHeadRecord,
    HistoricalMatch,
    MatchAnalysis,
    MatchAnalysisRequest,
    PointEvent,
    SetResult,
)
from engine.orchestrator import MatchAnalyticsEngine

AS_OF = datetime(2026, 7, 29, tzinfo=timezone.utc)


def _fixed_clock() -> datetime:
    return AS_OF


def _match() -> Match:
    return Match(
        id="m1",
        player_one=Player(id="p1", name="Ma Long", country="CHN", ranking=1),
        player_two=Player(id="p2", name="Fan Zhendong", country="CHN", ranking=2),
        status=MatchStatus.SCHEDULED,
    )


def _full_request() -> MatchAnalysisRequest:
    return MatchAnalysisRequest(
        match=_match(),
        odds=[Odds(match_id="m1", bookmaker="Pinnacle", player_one_odds=1.9, player_two_odds=1.95, captured_at=AS_OF)],
        player_one_history=[
            HistoricalMatch(
                player_id="p1", opponent_id="x", played_at=AS_OF, won=True,
                sets=[SetResult(player_points=11, opponent_points=6)],
            )
            for _ in range(4)
        ],
        player_two_history=[
            HistoricalMatch(
                player_id="p2", opponent_id="y", played_at=AS_OF, won=False,
                sets=[SetResult(player_points=6, opponent_points=11)],
            )
            for _ in range(4)
        ],
        point_progression=[
            PointEvent(set_number=1, winner="player_one", player_one_score=1, player_two_score=0),
        ],
        head_to_head=HeadToHeadRecord(player_one_wins=3, player_two_wins=1),
        context=CompetitionContext(surface="indoor", competition_name="Demo Cup", best_of_sets=7),
    )


def test_analyze_returns_match_analysis():
    engine = MatchAnalyticsEngine(settings=AnalyticsSettings(), clock=_fixed_clock)
    analysis = engine.analyze(_full_request())
    assert isinstance(analysis, MatchAnalysis)
    assert analysis.match_id == "m1"
    assert analysis.generated_at == AS_OF


def test_analyze_probabilities_sum_to_one():
    engine = MatchAnalyticsEngine(clock=_fixed_clock)
    analysis = engine.analyze(_full_request())
    total = analysis.probability.player_one_probability + analysis.probability.player_two_probability
    assert abs(total - 1.0) < 1e-9


def test_analyze_is_deterministic_across_repeated_runs():
    engine = MatchAnalyticsEngine(clock=_fixed_clock)
    request = _full_request()
    first = engine.analyze(request)
    second = engine.analyze(request)
    assert first == second


def test_analyze_works_with_minimal_request():
    minimal_request = MatchAnalysisRequest(match=_match())
    engine = MatchAnalyticsEngine(clock=_fixed_clock)
    analysis = engine.analyze(minimal_request)

    assert analysis.data_quality.odds_available is False
    assert analysis.match_features.player_one.matches_considered == 0
    assert analysis.confidence.score < 0.5
    assert analysis.value.player_one is None
    assert analysis.patterns == []


def test_analyze_missing_optional_data_does_not_crash_and_lowers_confidence():
    full_analysis = MatchAnalyticsEngine(clock=_fixed_clock).analyze(_full_request())
    minimal_analysis = MatchAnalyticsEngine(clock=_fixed_clock).analyze(
        MatchAnalysisRequest(match=_match())
    )
    assert minimal_analysis.confidence.score < full_analysis.confidence.score


def test_analyze_uses_injected_clock():
    other_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    engine = MatchAnalyticsEngine(clock=lambda: other_time)
    analysis = engine.analyze(MatchAnalysisRequest(match=_match()))
    assert analysis.generated_at == other_time


def test_analyze_produces_explanations_and_no_llm_randomness():
    engine = MatchAnalyticsEngine(clock=_fixed_clock)
    analysis = engine.analyze(_full_request())
    assert len(analysis.explanations) > 0
    assert all(isinstance(line, str) for line in analysis.explanations)


def test_analyze_detects_patterns_when_history_present():
    engine = MatchAnalyticsEngine(clock=_fixed_clock)
    analysis = engine.analyze(_full_request())
    assert any(p.player_id == "p1" for p in analysis.patterns)
