"""Tests for deterministic model + metadata serialization."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np

from ml.baseline import BaselineModelKind, build_baseline_model
from ml.models import ClassificationMetrics, ConfusionMatrix, ModelRegistryEntry, TargetKind
from ml.serialization import load_metadata, load_model, save_metadata, save_model

X = np.array([[0.0], [1.0], [2.0], [3.0]])
Y = np.array([0, 0, 1, 1])


def _entry(model_path: str, metadata_path: str) -> ModelRegistryEntry:
    return ModelRegistryEntry(
        model_id="test-model",
        model_version="1.0.0",
        training_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        feature_schema_version="1.0.0",
        target_kind=TargetKind.PLAYER_A_WON,
        training_period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        training_period_end=datetime(2026, 2, 1, tzinfo=timezone.utc),
        metrics=ClassificationMetrics(
            sample_size=4, accuracy=1.0, precision=1.0, recall=1.0, f1=1.0, roc_auc=1.0,
            log_loss=0.01, brier_score=0.01,
            confusion_matrix=ConfusionMatrix(true_negative=2, false_positive=0, false_negative=0, true_positive=2),
        ),
        hyperparameters={"random_state": 0, "max_iter": 1000},
        serialization_format="joblib",
        model_path=model_path,
        metadata_path=metadata_path,
    )


def test_model_roundtrip_preserves_predictions(tmp_path):
    model = build_baseline_model(BaselineModelKind.LOGISTIC_REGRESSION)
    model.fit(X, Y)
    model_path = str(tmp_path / "model.joblib")

    save_model(model, model_path)
    reloaded = load_model(model_path)

    assert np.allclose(model.predict_proba(X), reloaded.predict_proba(X))


def test_metadata_roundtrip_is_equal(tmp_path):
    entry = _entry(str(tmp_path / "m.joblib"), str(tmp_path / "m.json"))
    metadata_path = str(tmp_path / "m.json")

    save_metadata(entry, metadata_path)
    reloaded = load_metadata(metadata_path)

    assert reloaded == entry


def test_metadata_json_is_deterministic_across_saves(tmp_path):
    entry = _entry(str(tmp_path / "m.joblib"), str(tmp_path / "m.json"))
    path_one = str(tmp_path / "one.json")
    path_two = str(tmp_path / "two.json")

    save_metadata(entry, path_one)
    save_metadata(entry, path_two)

    with open(path_one) as f_one, open(path_two) as f_two:
        assert f_one.read() == f_two.read()


def test_atomic_write_leaves_no_temp_files_behind(tmp_path):
    model = build_baseline_model(BaselineModelKind.LOGISTIC_REGRESSION)
    model.fit(X, Y)
    save_model(model, str(tmp_path / "model.joblib"))

    leftovers = [name for name in os.listdir(tmp_path) if name.startswith(".tmp-ml-")]
    assert leftovers == []


def test_metadata_json_keys_are_sorted(tmp_path):
    entry = _entry(str(tmp_path / "m.joblib"), str(tmp_path / "m.json"))
    metadata_path = str(tmp_path / "m.json")
    save_metadata(entry, metadata_path)

    with open(metadata_path) as f:
        content = f.read()

    import json

    data = json.loads(content)
    assert list(data.keys()) == sorted(data.keys())
