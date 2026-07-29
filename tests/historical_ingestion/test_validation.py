"""Tests for staged (structural/semantic/temporal) validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from config.historical_ingestion import HistoricalIngestionSettings
from historical_ingestion.models import ImportedMatch, ImportedOdds, ImportedSet, ImportProvenance
from historical_ingestion.validation import IssueSeverity, decide_validation_outcome, validate_match, validate_odds

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _provenance(**overrides) -> ImportProvenance:
    defaults = dict(
        provider="mock", provider_record_id="r1", source_batch_id="b1",
        source_timestamp=NOW, ingested_at=NOW, raw_fingerprint="fp", mapping_version="1.0.0",
    )
    defaults.update(overrides)
    return ImportProvenance(**defaults)


def _valid_match(**overrides) -> ImportedMatch:
    defaults = dict(
        provenance=_provenance(),
        player_a_external_id="pa", player_b_external_id="pb",
        scheduled_at=NOW - timedelta(days=1), actual_start_at=NOW - timedelta(days=1),
        completed_at=NOW - timedelta(days=1) + timedelta(hours=1),
        status_raw="COMPLETED", status="finished", best_of=5, winner_external_id="pa",
        sets=[ImportedSet(set_number=1, player_a_points=11, player_b_points=7)],
    )
    defaults.update(overrides)
    return ImportedMatch(**defaults)


def test_valid_match_has_no_issues():
    issues = validate_match(_valid_match(), HistoricalIngestionSettings(), NOW)
    assert issues == []


def test_missing_players_is_structural_fatal():
    match = _valid_match(player_a_external_id=None)
    issues = validate_match(match, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "missing_player_a" and i.severity == IssueSeverity.FATAL and i.stage == "structural" for i in issues)


def test_negative_set_score_is_fatal():
    match = _valid_match(sets=[ImportedSet(set_number=1, player_a_points=-1, player_b_points=7)])
    issues = validate_match(match, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "negative_set_score" for i in issues)


def test_same_player_is_fatal_semantic():
    match = _valid_match(player_a_external_id="same", player_b_external_id="same")
    issues = validate_match(match, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "same_player" and i.stage == "semantic" for i in issues)


def test_winner_not_participant_is_fatal():
    match = _valid_match(winner_external_id="someone-else")
    issues = validate_match(match, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "winner_not_participant" for i in issues)


def test_completed_without_winner_is_error():
    match = _valid_match(winner_external_id=None)
    issues = validate_match(match, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "completed_without_winner" and i.severity == IssueSeverity.ERROR for i in issues)


def test_scheduled_with_winner_is_error():
    match = _valid_match(status="scheduled", sets=[])
    issues = validate_match(match, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "scheduled_with_winner" for i in issues)


def test_duplicate_set_number_is_error():
    match = _valid_match(
        sets=[
            ImportedSet(set_number=1, player_a_points=11, player_b_points=7),
            ImportedSet(set_number=1, player_a_points=9, player_b_points=11),
        ]
    )
    issues = validate_match(match, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "duplicate_set_number" for i in issues)


def test_score_result_mismatch_detected():
    match = _valid_match(
        winner_external_id="pa",
        sets=[
            ImportedSet(set_number=1, player_a_points=5, player_b_points=11),
            ImportedSet(set_number=2, player_a_points=5, player_b_points=11),
        ],
    )
    issues = validate_match(match, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "score_result_mismatch" for i in issues)


def test_completion_before_start_is_fatal_temporal():
    match = _valid_match(
        actual_start_at=NOW - timedelta(days=1),
        completed_at=NOW - timedelta(days=1, hours=1),
    )
    issues = validate_match(match, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "completion_before_start" and i.stage == "temporal" for i in issues)


def test_future_historical_record_rejected_by_default():
    match = _valid_match(
        scheduled_at=NOW + timedelta(days=10), actual_start_at=NOW + timedelta(days=10),
        completed_at=NOW + timedelta(days=10, hours=1),
    )
    issues = validate_match(match, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "future_historical_record" for i in issues)


def test_future_historical_record_allowed_when_configured():
    settings = HistoricalIngestionSettings(allow_future_records=True)
    match = _valid_match(
        scheduled_at=NOW + timedelta(days=10), actual_start_at=NOW + timedelta(days=10),
        completed_at=NOW + timedelta(days=10, hours=1),
    )
    issues = validate_match(match, settings, NOW)
    assert not any(i.code == "future_historical_record" for i in issues)


def test_decide_outcome_fatal_always_rejects():
    settings = HistoricalIngestionSettings(validation_policy="lenient", warning_policy="accept")
    match = _valid_match(player_a_external_id=None)
    issues = validate_match(match, settings, NOW)
    assert decide_validation_outcome(issues, settings) == "reject"


def test_decide_outcome_strict_rejects_on_error():
    settings = HistoricalIngestionSettings(validation_policy="strict")
    match = _valid_match(winner_external_id=None)
    issues = validate_match(match, settings, NOW)
    assert decide_validation_outcome(issues, settings) == "reject"


def test_decide_outcome_lenient_accepts_with_warnings_on_error():
    settings = HistoricalIngestionSettings(validation_policy="lenient")
    match = _valid_match(winner_external_id=None)
    issues = validate_match(match, settings, NOW)
    assert decide_validation_outcome(issues, settings) == "accept_with_warnings"


def _valid_odds(**overrides) -> ImportedOdds:
    defaults = dict(
        provenance=_provenance(),
        provider_match_id="m1", bookmaker="Pinnacle", selection_external_id="pa",
        decimal_odds=1.8, captured_at=NOW - timedelta(hours=1), market_id="winner",
    )
    defaults.update(overrides)
    return ImportedOdds(**defaults)


def test_valid_odds_has_no_issues():
    issues = validate_odds(_valid_odds(), None, HistoricalIngestionSettings(), NOW)
    assert issues == []


def test_odds_below_one_is_fatal():
    issues = validate_odds(_valid_odds(decimal_odds=0.9), None, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "invalid_odds" and i.severity == IssueSeverity.FATAL for i in issues)


def test_odds_missing_bookmaker_is_fatal():
    issues = validate_odds(_valid_odds(bookmaker=None), None, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "missing_bookmaker" for i in issues)


def test_future_odds_timestamp_rejected():
    issues = validate_odds(_valid_odds(captured_at=NOW + timedelta(days=5)), None, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "future_odds_timestamp" for i in issues)


def test_odds_after_match_lifecycle_is_warning():
    match = _valid_match()
    odds = _valid_odds(captured_at=match.completed_at + timedelta(days=1))
    issues = validate_odds(odds, match, HistoricalIngestionSettings(), NOW)
    assert any(i.code == "odds_after_match_lifecycle" and i.severity == IssueSeverity.WARNING for i in issues)
