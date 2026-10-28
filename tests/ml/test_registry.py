"""Tests for the in-memory model registry."""

from __future__ import annotations

from datetime import datetime, timezone

from ml.models import ClassificationMetrics, ConfusionMatrix, ModelRegistryEntry, TargetKind
from ml.registry import InMemoryModelRegistry


def _entry(model_id: str, training_timestamp: datetime) -> ModelRegistryEntry:
    return ModelRegistryEntry(
        model_id=model_id,
        model_version="1.0.0",
        training_timestamp=training_timestamp,
        feature_schema_version="1.0.0",
        target_kind=TargetKind.PLAYER_A_WON,
        metrics=ClassificationMetrics(
            sample_size=1, accuracy=1.0, precision=1.0, recall=1.0, f1=1.0, roc_auc=None,
            log_loss=0.0, brier_score=0.0,
            confusion_matrix=ConfusionMatrix(true_negative=1, false_positive=0, false_negative=0, true_positive=0),
        ),
        model_path=f"/tmp/{model_id}.joblib",
        metadata_path=f"/tmp/{model_id}.json",
    )


def test_register_and_get():
    registry = InMemoryModelRegistry()
    entry = _entry("m1", datetime(2026, 1, 1, tzinfo=timezone.utc))
    registry.register(entry)
    assert registry.get("m1") == entry


def test_get_missing_returns_none():
    registry = InMemoryModelRegistry()
    assert registry.get("does-not-exist") is None


def test_count():
    registry = InMemoryModelRegistry()
    registry.register(_entry("m1", datetime(2026, 1, 1, tzinfo=timezone.utc)))
    registry.register(_entry("m2", datetime(2026, 1, 2, tzinfo=timezone.utc)))
    assert registry.count() == 2


def test_re_registering_same_model_id_overwrites():
    registry = InMemoryModelRegistry()
    registry.register(_entry("m1", datetime(2026, 1, 1, tzinfo=timezone.utc)))
    updated = _entry("m1", datetime(2026, 1, 5, tzinfo=timezone.utc))
    registry.register(updated)
    assert registry.count() == 1
    assert registry.get("m1").training_timestamp == datetime(2026, 1, 5, tzinfo=timezone.utc)


def test_list_entries_sorted_by_training_timestamp():
    registry = InMemoryModelRegistry()
    registry.register(_entry("later", datetime(2026, 3, 1, tzinfo=timezone.utc)))
    registry.register(_entry("earlier", datetime(2026, 1, 1, tzinfo=timezone.utc)))
    registry.register(_entry("middle", datetime(2026, 2, 1, tzinfo=timezone.utc)))
    ordered_ids = [entry.model_id for entry in registry.list_entries()]
    assert ordered_ids == ["earlier", "middle", "later"]


def test_registry_instances_do_not_share_state():
    registry_one = InMemoryModelRegistry()
    registry_two = InMemoryModelRegistry()
    registry_one.register(_entry("m1", datetime(2026, 1, 1, tzinfo=timezone.utc)))
    assert registry_two.count() == 0
