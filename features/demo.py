"""Read-only demonstration of the Phase 3 historical intelligence platform.

Run with:
    python -m features.demo

All data below is explicitly illustrative/mock — hand-authored, fixed
historical records, not real match results. Makes no network calls and
never suggests or places a wager.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from datasets.builder import DatasetBuilder
from datasets.leakage import assert_no_leakage, run_dataset_leakage_report
from datasets.splits import chronological_holdout_split
from features.builder import HistoricalFeatureBuilder
from identity.models import ExternalIdentifier, PlayerIdentityRecord
from identity.normalizer import build_normalized_identity
from identity.resolver import IdentityResolver
from persistence.in_memory import InMemoryMatchRepository, InMemoryOddsRepository, InMemoryPlayerRepository
from persistence.models import (
    HistoricalMatchRecord,
    HistoricalOddsRecord,
    HistoricalPlayerRecord,
    HistoricalSetRecord,
    MatchRecordStatus,
)

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
_PLAYER_A = "demo-player-a"
_PLAYER_B = "demo-player-b"


def _make_match(
    index: int, winner: str | None, day_offset: int, sets: list[tuple[int, int]] | None
) -> HistoricalMatchRecord:
    scheduled = _BASE + timedelta(days=day_offset)
    is_scheduled = winner is None
    return HistoricalMatchRecord(
        id=f"demo-match-{index}",
        provider="mock",
        provider_match_id=str(index),
        competition_id="demo-open",
        player_a_id=_PLAYER_A,
        player_b_id=_PLAYER_B,
        scheduled_at=scheduled,
        actual_start_at=None if is_scheduled else scheduled,
        completed_at=None if is_scheduled else scheduled + timedelta(hours=1),
        status=MatchRecordStatus.SCHEDULED if is_scheduled else MatchRecordStatus.FINISHED,
        best_of=5,
        winner_id=winner,
        sets=[]
        if is_scheduled
        else [HistoricalSetRecord(set_number=i + 1, player_a_points=a, player_b_points=b) for i, (a, b) in enumerate(sets or [])],
        provider_timestamp=scheduled,
        ingested_at=scheduled + timedelta(minutes=5),
    )


def _demonstrate_identity_resolution() -> None:
    print("-- Identity resolution --")
    known = [
        PlayerIdentityRecord(
            id=_PLAYER_A,
            canonical_name="Timo Boll",
            normalized_name="timo boll",
            aliases=["t. boll"],
            country="DEU",
            external_ids=[ExternalIdentifier(provider="mock", provider_player_id="1001")],
        )
    ]
    resolver = IdentityResolver()

    exact = build_normalized_identity("Timo Boll", external_provider="mock", external_player_id="1001")
    exact_result = resolver.resolve(exact, known)
    print(f"  Exact external ID match -> {exact_result.outcome.value} (identity_id={exact_result.identity_id})")

    fuzzy = build_normalized_identity("Timo Böll", country="DEU")
    fuzzy_result = resolver.resolve(fuzzy, known)
    print(f"  Accent-variant name match -> {fuzzy_result.outcome.value} (identity_id={fuzzy_result.identity_id})")

    unrelated = build_normalized_identity("Someone Else Entirely")
    unrelated_result = resolver.resolve(unrelated, known)
    print(f"  Unrelated name -> {unrelated_result.outcome.value}")


def main() -> None:
    """Run the full demo: mock repository -> identity -> feature snapshot -> mini dataset."""
    _demonstrate_identity_resolution()

    player_repo = InMemoryPlayerRepository()
    match_repo = InMemoryMatchRepository()
    odds_repo = InMemoryOddsRepository()

    for player_id, name, country in [(_PLAYER_A, "Demo Player A", "DEU"), (_PLAYER_B, "Demo Player B", "CHN")]:
        player_repo.add(
            HistoricalPlayerRecord(
                id=player_id, name=name, country=country, provider="mock",
                provider_player_id=player_id, ingested_at=_BASE,
            )
        )

    # Ten completed historical matches between the same two players, alternating outcomes.
    completed_matches = []
    for i in range(10):
        winner = _PLAYER_A if i % 3 != 0 else _PLAYER_B
        sets = [(11, 7), (9, 11), (11, 6)] if winner == _PLAYER_A else [(7, 11), (11, 9), (6, 11)]
        completed_matches.append(_make_match(i, winner, day_offset=i * 3, sets=sets))
    match_repo.add_many(completed_matches)

    # The target match: scheduled, not yet played — this is what we build a pre-match snapshot for.
    target_day_offset = 40
    target = _make_match(99, winner=None, day_offset=target_day_offset, sets=None)
    match_repo.add(target)

    odds_repo.add(
        HistoricalOddsRecord(
            id="demo-odds-1",
            match_id=target.id,
            bookmaker="MockBook",
            selection_id=_PLAYER_A,
            decimal_odds=1.85,
            captured_at=target.scheduled_at - timedelta(hours=1),
            provider="mock",
        )
    )

    as_of = target.scheduled_at
    builder = HistoricalFeatureBuilder(match_repo)
    snapshot = builder.build(target.id, as_of)

    print("\n-- Feature snapshot (target match, illustrative mock data) --")
    print(f"  as_of: {snapshot.as_of.isoformat()}")
    for label, features in [("player_a", snapshot.player_a_features), ("player_b", snapshot.player_b_features)]:
        all_time = features.window("all_time")
        last_5 = features.window("last_5")
        print(
            f"  {label} ({features.player_id}): observations={features.observation_count} "
            f"all_time_win_rate={all_time.win_rate} last_5_win_rate={last_5.win_rate if last_5 else None} "
            f"recency_weighted_win_rate={features.recency_weighted_win_rate} "
            f"volatility={features.volatility_score} result_streak={features.result_streak}"
        )

    matchup = snapshot.matchup_features
    print(
        f"  matchup: head_to_head_matches={matchup.head_to_head_matches} "
        f"player_a_win_rate={matchup.player_a_win_rate} "
        f"recent_h2h_win_rate_player_a={matchup.recent_head_to_head_win_rate_player_a} "
        f"days_since_last_meeting={matchup.days_since_last_meeting}"
    )

    provenance = snapshot.provenance
    print(f"  observation counts: player_a={provenance.player_a_observation_count} "
          f"player_b={provenance.player_b_observation_count} h2h={provenance.head_to_head_observation_count}")
    print(f"  data quality score: {provenance.data_quality_score:.1f}/100")
    for warning in provenance.warnings:
        print(f"  - [warning] {warning}")
    print(f"  provenance input_fingerprint: {provenance.input_fingerprint}")
    print(f"  provenance repository_fingerprint: {provenance.repository_fingerprint}")

    dataset_builder = DatasetBuilder(match_repo, builder)
    match_ids = [m.id for m in completed_matches]
    examples, manifest = dataset_builder.build(match_ids, dataset_id="demo-dataset")

    plan = chronological_holdout_split(examples)
    print("\n-- Chronological training dataset (illustrative mock data) --")
    print(f"  training examples: {manifest.training_example_count} (skipped: {manifest.skipped_count})")
    print(f"  label distribution: player_a_wins={manifest.label_distribution.player_a_wins} "
          f"player_b_wins={manifest.label_distribution.player_b_wins}")
    for split in plan.splits:
        print(f"  split {split.label}: {split.size} example(s) [{split.start} -> {split.end}]")

    report = run_dataset_leakage_report(examples, match_repo, plan)
    print(f"\n-- Leakage check status: {'PASSED' if report.passed else 'FAILED'} --")
    for check in report.checks[:5]:
        print(f"  - {check.name}: {'ok' if check.passed else 'FAILED'} ({check.detail})")
    assert_no_leakage(report)

    print(
        "\nAll data above is illustrative/mock. No trained ML model exists yet; this is a "
        "deterministic feature-engineering and leakage-safety demonstration only. Output is "
        "analytical decision support, not certainty, and no automated wagering exists."
    )


if __name__ == "__main__":
    main()
