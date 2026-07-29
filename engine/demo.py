"""Read-only demonstration of the Phase 2B analytics engine.

Run with:
    python -m engine.demo

Loads deterministic mock ingestion data, normalizes it into domain
models, adds a small, explicitly-labeled set of demo-only historical
match/point data (illustrative only — never treated as real by the
engine unless supplied), runs the analytics engine, and prints a
human-readable analysis. Makes no network calls and never suggests or
places a wager.
"""

from __future__ import annotations

from datetime import datetime, timezone

from engine.models import (
    CompetitionContext,
    HeadToHeadRecord,
    HistoricalMatch,
    MatchAnalysisRequest,
    PointEvent,
    SetResult,
)
from engine.orchestrator import MatchAnalyticsEngine
from ingestion.service import IngestionService
from ingestion.sources.mock_source import MockTableTennisSource


def _demo_history_for_timo_boll() -> list[HistoricalMatch]:
    """Illustrative, explicitly hand-authored demo data — not derived from any real record."""
    return [
        HistoricalMatch(
            player_id="timo_boll",
            opponent_id="demo_opponent_a",
            opponent_ranking=25,
            played_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            won=True,
            sets=[SetResult(player_points=11, opponent_points=7), SetResult(player_points=11, opponent_points=9)],
        ),
        HistoricalMatch(
            player_id="timo_boll",
            opponent_id="demo_opponent_b",
            opponent_ranking=8,
            played_at=datetime(2026, 6, 20, tzinfo=timezone.utc),
            won=False,
            sets=[
                SetResult(player_points=9, opponent_points=11),
                SetResult(player_points=11, opponent_points=6),
                SetResult(player_points=8, opponent_points=11),
            ],
        ),
        HistoricalMatch(
            player_id="timo_boll",
            opponent_id="demo_opponent_c",
            opponent_ranking=40,
            played_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
            won=True,
            sets=[SetResult(player_points=11, opponent_points=4), SetResult(player_points=11, opponent_points=8)],
        ),
    ]


def _demo_history_for_dimitrij_ovtcharov() -> list[HistoricalMatch]:
    return [
        HistoricalMatch(
            player_id="dimitrij_ovtcharov",
            opponent_id="demo_opponent_d",
            opponent_ranking=15,
            played_at=datetime(2026, 7, 5, tzinfo=timezone.utc),
            won=True,
            sets=[SetResult(player_points=11, opponent_points=9), SetResult(player_points=11, opponent_points=8)],
        ),
        HistoricalMatch(
            player_id="dimitrij_ovtcharov",
            opponent_id="demo_opponent_e",
            opponent_ranking=6,
            played_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
            won=False,
            sets=[SetResult(player_points=7, opponent_points=11), SetResult(player_points=9, opponent_points=11)],
        ),
    ]


def _demo_point_progression() -> list[PointEvent]:
    """A short, illustrative point sequence for the third (currently live) set."""
    return [
        PointEvent(set_number=3, winner="player_two", player_one_score=0, player_two_score=1),
        PointEvent(set_number=3, winner="player_two", player_one_score=0, player_two_score=2),
        PointEvent(set_number=3, winner="player_one", player_one_score=1, player_two_score=2),
        PointEvent(set_number=3, winner="player_one", player_one_score=2, player_two_score=2),
        PointEvent(set_number=3, winner="player_one", player_one_score=3, player_two_score=2),
        PointEvent(set_number=3, winner="player_one", player_one_score=4, player_two_score=2),
    ]


def _print_analysis(analysis) -> None:  # type: ignore[no-untyped-def]
    print("=" * 70)
    print(f"TT Edge Terminal — Analytics Demo (match {analysis.match_id})")
    print("=" * 70)
    features = analysis.match_features
    print(f"\n{features.player_one.player_name} vs {features.player_two.player_name}")
    print(
        f"  Form:      {features.player_one.form_score:.1f} vs {features.player_two.form_score:.1f}"
    )
    print(
        f"  Momentum:  {features.player_one.momentum_score:.1f} vs {features.player_two.momentum_score:.1f} "
        f"({features.player_one.momentum_state.value})"
    )

    prob = analysis.probability
    print(
        f"\nWin probability: {features.player_one.player_name} "
        f"{prob.player_one_probability * 100:.1f}% vs {features.player_two.player_name} "
        f"{prob.player_two_probability * 100:.1f}%"
    )

    print(f"\nConfidence: {analysis.confidence.label.value} ({analysis.confidence.score:.2f})")
    print(f"Risk:       {analysis.risk.label.value} ({analysis.risk.score:.1f}/100)")

    if analysis.value.player_one:
        v1 = analysis.value.player_one
        print(
            f"\nValue ({v1.player_id}): fair odds {v1.fair_odds:.2f}, market {v1.decimal_odds:.2f}, "
            f"edge {v1.probability_edge * 100:+.1f}pp, EV {v1.expected_value:+.3f} -> {v1.decision.value}"
        )
    if analysis.value.player_two:
        v2 = analysis.value.player_two
        print(
            f"Value ({v2.player_id}): fair odds {v2.fair_odds:.2f}, market {v2.decimal_odds:.2f}, "
            f"edge {v2.probability_edge * 100:+.1f}pp, EV {v2.expected_value:+.3f} -> {v2.decision.value}"
        )

    if analysis.patterns:
        print("\nDetected patterns:")
        for pattern in analysis.patterns:
            print(
                f"  - {pattern.player_id}: {pattern.pattern} "
                f"(strength={pattern.strength:.2f}, confidence={pattern.confidence:.2f}, n={pattern.sample_size})"
            )

    print(f"\nData quality: {analysis.data_quality.score:.1f}/100")
    for warning in analysis.data_quality.warnings:
        print(f"  - [{warning.severity.value}] {warning.detail}")

    print("\nExplanations:")
    for line in analysis.explanations:
        print(f"  - {line}")

    print(
        "\nThis is analytical decision support based on deterministic, rules-based "
        "calculations over mock data — not a certainty, and not a wager suggestion."
    )
    print("=" * 70)


def main() -> None:
    """Run the full demo pipeline: mock ingestion -> normalization -> analytics -> print."""
    ingestion_result = IngestionService(source=MockTableTennisSource()).run_once()

    match = next(m for m in ingestion_result.matches if m.id == "match-002")
    odds = [o for o in ingestion_result.odds if o.match_id == match.id]

    request = MatchAnalysisRequest(
        match=match,
        odds=odds,
        player_one_history=_demo_history_for_timo_boll(),
        player_two_history=_demo_history_for_dimitrij_ovtcharov(),
        point_progression=_demo_point_progression(),
        head_to_head=HeadToHeadRecord(
            player_one_wins=3, player_two_wins=2, last_played_at=datetime(2026, 3, 1, tzinfo=timezone.utc)
        ),
        context=CompetitionContext(surface="indoor", competition_name="Demo Open", best_of_sets=7),
    )

    engine = MatchAnalyticsEngine()
    analysis = engine.analyze(request)
    _print_analysis(analysis)


if __name__ == "__main__":
    main()
