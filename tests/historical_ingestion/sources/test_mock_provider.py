"""Tests for the deterministic mock provider source."""

from __future__ import annotations

from historical_ingestion.sources.mock_provider import MockTableTennisProviderSource


def test_two_batches_cover_all_records():
    source = MockTableTennisProviderSource(batch_size=8)
    first = source.fetch_batch(None)
    second = source.fetch_batch(first.next_cursor)
    assert second.next_cursor is None
    assert len(first.records) + len(second.records) == len(source._records)


def test_deterministic_across_instances():
    # `fetched_at` reflects real construction time (by design) and is excluded here;
    # every other field — the actual illustrative data — must be identical.
    a = MockTableTennisProviderSource()
    b = MockTableTennisProviderSource()
    batch_a = a.fetch_batch(None)
    batch_b = b.fetch_batch(None)
    dump = lambda records: [r.model_dump(mode="json", exclude={"fetched_at"}) for r in records]  # noqa: E731
    assert dump(batch_a.records) == dump(batch_b.records)


def test_includes_exact_duplicate_match():
    source = MockTableTennisProviderSource(batch_size=100)
    batch = source.fetch_batch(None)
    match_records = [r for r in batch.records if r.record_type == "match" and r.provider_record_id == "mp-1"]
    assert len(match_records) >= 2


def test_includes_ranking_and_odds_history():
    source = MockTableTennisProviderSource(batch_size=100)
    batch = source.fetch_batch(None)
    odds = [r for r in batch.records if r.record_type == "odds"]
    rankings = [r for r in batch.records if r.record_type == "ranking"]
    assert len(odds) >= 2
    assert len(rankings) >= 2


def test_health_is_always_healthy():
    source = MockTableTennisProviderSource()
    assert source.health().healthy is True


def test_no_network_calls_in_source_module():
    import historical_ingestion.sources.mock_provider as module

    source = open(module.__file__, encoding="utf-8").read()
    for forbidden in ("requests.", "urllib.request", "httpx.", "socket."):
        assert forbidden not in source
