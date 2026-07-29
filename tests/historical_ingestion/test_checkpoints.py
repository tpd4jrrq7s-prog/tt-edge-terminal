"""Tests for import checkpoints and the in-memory CheckpointStore."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from historical_ingestion.checkpoints import ImportCheckpoint, InMemoryCheckpointStore
from historical_ingestion.errors import CheckpointVersionError

AWARE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _checkpoint(**overrides) -> ImportCheckpoint:
    defaults = dict(source_name="s1", provider="mock", cursor="10", checkpoint_version=1, updated_at=AWARE)
    defaults.update(overrides)
    return ImportCheckpoint(**defaults)


def test_checkpoint_rejects_naive_updated_at():
    with pytest.raises(ValidationError):
        ImportCheckpoint(source_name="s1", provider="mock", checkpoint_version=1, updated_at=datetime(2026, 1, 1))


def test_get_missing_returns_none():
    store = InMemoryCheckpointStore()
    assert store.get("s1", "mock") is None


def test_save_and_get_round_trips():
    store = InMemoryCheckpointStore()
    store.save(_checkpoint())
    fetched = store.get("s1", "mock")
    assert fetched.cursor == "10"


def test_save_returns_defensive_copy():
    store = InMemoryCheckpointStore()
    store.save(_checkpoint())
    fetched = store.get("s1", "mock")
    fetched.cursor = "MUTATED"
    assert store.get("s1", "mock").cursor == "10"


def test_reset_clears_checkpoint():
    store = InMemoryCheckpointStore()
    store.save(_checkpoint())
    store.reset("s1", "mock")
    assert store.get("s1", "mock") is None


def test_incompatible_version_on_save_fails_clearly():
    store = InMemoryCheckpointStore(checkpoint_version=2)
    with pytest.raises(CheckpointVersionError):
        store.save(_checkpoint(checkpoint_version=1))


def test_incompatible_version_on_get_fails_clearly():
    store = InMemoryCheckpointStore(checkpoint_version=1)
    store.save(_checkpoint(checkpoint_version=1))
    # Simulate the expected version changing after the checkpoint was written.
    store._checkpoint_version = 2
    with pytest.raises(CheckpointVersionError):
        store.get("s1", "mock")


def test_different_source_provider_pairs_are_independent():
    store = InMemoryCheckpointStore()
    store.save(_checkpoint(source_name="s1", provider="mock", cursor="1"))
    store.save(_checkpoint(source_name="s2", provider="mock", cursor="2"))
    assert store.get("s1", "mock").cursor == "1"
    assert store.get("s2", "mock").cursor == "2"
