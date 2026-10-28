"""Read-only demonstration of the Phase 5A ML foundation pipeline.

Run with:
    python -m ml.demo

Training Dataset -> Feature Matrix -> Target Vector -> Time Split ->
Baseline Model API -> Evaluation API.

All data below is explicitly illustrative/mock — hand-authored, fixed
historical records, not real match results. This demo trains only a
baseline logistic regression to verify the training infrastructure; it
is not a production prediction model. There is no live inference.
Makes no network calls and never suggests or places a wager.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from datasets.builder import DatasetBuilder
from features.builder import HistoricalFeatureBuilder
from ml.baseline import BaselineModelKind, build_baseline_model
from ml.calibration import CalibrationMethod, calibrate_model
from ml.evaluation import evaluate_model
from ml.feature_matrix import build_feature_matrix
from ml.models import ModelRegistryEntry, TargetKind
from ml.preprocessing import FittedPreprocessor, PreprocessingConfig
from ml.registry import InMemoryModelRegistry
from ml.serialization import load_metadata, load_model, save_metadata, save_model
from ml.splits import chronological_split
from ml.targets import extract_targets
from persistence.in_memory import InMemoryMatchRepository
from persistence.models import HistoricalMatchRecord, HistoricalSetRecord, MatchRecordStatus

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
_PLAYER_A = "ml-demo-player-a"
_PLAYER_B = "ml-demo-player-b"

_WIN_SETS = [(11, 7), (11, 9), (11, 6)]
_LOSS_SETS = [(7, 11), (9, 11), (6, 11)]


def _make_match(index: int, winner: str, day_offset: float) -> HistoricalMatchRecord:
    scheduled = _BASE + timedelta(days=day_offset)
    sets = _WIN_SETS if winner == _PLAYER_A else _LOSS_SETS
    return HistoricalMatchRecord(
        id=f"ml-demo-match-{index}",
        provider="mock",
        provider_match_id=str(index),
        competition_id="ml-demo-open",
        player_a_id=_PLAYER_A,
        player_b_id=_PLAYER_B,
        scheduled_at=scheduled,
        actual_start_at=scheduled,
        completed_at=scheduled + timedelta(hours=1),
        status=MatchRecordStatus.FINISHED,
        best_of=5,
        winner_id=winner,
        sets=[HistoricalSetRecord(set_number=i + 1, player_a_points=a, player_b_points=b) for i, (a, b) in enumerate(sets)],
        provider_timestamp=scheduled,
        ingested_at=scheduled + timedelta(minutes=5),
    )


def _build_matches(count: int = 70) -> list[HistoricalMatchRecord]:
    """A deterministic, repeating-form pattern (3 A wins, then 2 B wins) so rolling-window
    features actually carry predictive signal for the demo, without any randomness."""
    matches = []
    for i in range(count):
        winner = _PLAYER_A if (i % 5) < 3 else _PLAYER_B
        day_offset = i * 2 + (i % 3) * 0.3
        matches.append(_make_match(i, winner, day_offset))
    return matches


def main() -> None:
    print("=" * 70)
    print("TT Edge Terminal — ML Foundation Demo (Phase 5A, illustrative mock data)")
    print("=" * 70)

    print("\n-- Building mock historical repository --")
    match_repo = InMemoryMatchRepository()
    matches = _build_matches()
    match_repo.add_many(matches)
    print(f"  matches inserted: {match_repo.count()}")

    print("\n-- Building training dataset (leakage-safe pre-match snapshots) --")
    feature_builder = HistoricalFeatureBuilder(match_repo)
    dataset_builder = DatasetBuilder(match_repo, feature_builder)
    match_ids = [m.id for m in matches]
    examples, manifest = dataset_builder.build(match_ids, dataset_id="ml-demo-dataset")
    print(f"  training examples: {manifest.training_example_count} (skipped: {manifest.skipped_count})")
    print(
        f"  label distribution: player_a_wins={manifest.label_distribution.player_a_wins} "
        f"player_b_wins={manifest.label_distribution.player_b_wins}"
    )

    print("\n-- Building the feature matrix --")
    pairs = [(example.id, example.features) for example in examples]
    matrix = build_feature_matrix(pairs)
    print(f"  rows={matrix.n_rows} features={matrix.n_features} schema_version={matrix.schema_.version}")

    print("\n-- Building target vectors --")
    player_a_won = extract_targets(examples, TargetKind.PLAYER_A_WON)
    player_b_won = extract_targets(examples, TargetKind.PLAYER_B_WON)
    print(f"  player_a_won: {player_a_won.size} labels (schema {player_a_won.schema_version})")
    print(f"  player_b_won: {player_b_won.size} labels (schema {player_b_won.schema_version})")
    try:
        best_of = extract_targets(examples, TargetKind.BEST_OF)
        print(f"  best_of: {best_of.size} labels")
    except Exception as exc:
        print(f"  best_of: unavailable ({exc})")

    print("\n-- Chronological time split (train / validation / test, no shuffling) --")
    plan = chronological_split(matrix, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2)
    print(f"  train={len(plan.train_indices)} validation={len(plan.validation_indices)} test={len(plan.test_indices)}")

    y = np.array(player_a_won.values)

    print("\n-- Fitting preprocessor on training rows only (no leakage) --")
    preprocessor = FittedPreprocessor(PreprocessingConfig())
    preprocessor.fit(matrix, plan.train_indices)
    X_train = preprocessor.transform(matrix, plan.train_indices)
    X_validation = preprocessor.transform(matrix, plan.validation_indices)
    X_test = preprocessor.transform(matrix, plan.test_indices)
    y_train = y[plan.train_indices]
    y_validation = y[plan.validation_indices]
    y_test = y[plan.test_indices]
    print(f"  preprocessed shapes: train={X_train.shape} validation={X_validation.shape} test={X_test.shape}")

    print("\n-- Training baseline logistic regression --")
    model = build_baseline_model(BaselineModelKind.LOGISTIC_REGRESSION)
    model.fit(X_train, y_train)
    print("  fitted LogisticRegression on the training split")

    print("\n-- Evaluating on the held-out test split --")
    source_ids_test = [matrix.rows[i].source_id for i in plan.test_indices]
    result = evaluate_model(model, X_test, y_test, source_ids_test)
    metrics = result.metrics
    roc_auc_text = "n/a (single class in test split)" if metrics.roc_auc is None else f"{metrics.roc_auc:.3f}"
    print(f"  accuracy={metrics.accuracy:.3f} precision={metrics.precision:.3f} recall={metrics.recall:.3f} f1={metrics.f1:.3f}")
    print(f"  roc_auc={roc_auc_text} log_loss={metrics.log_loss:.3f} brier_score={metrics.brier_score:.3f}")
    cm = metrics.confusion_matrix
    print(f"  confusion matrix: tn={cm.true_negative} fp={cm.false_positive} fn={cm.false_negative} tp={cm.true_positive}")
    print(f"  calibration curve: {len(result.calibration_curve.bins)}/{result.calibration_curve.n_bins} bins populated")

    print("\n-- Calibrating probabilities on the validation split (Platt scaling) --")
    calibrated = calibrate_model(model, X_validation, y_validation, CalibrationMethod.PLATT)
    calibrated_result = evaluate_model(calibrated, X_test, y_test, source_ids_test)
    calibrated_metrics = calibrated_result.metrics
    print(
        f"  calibrated: brier_score={calibrated_metrics.brier_score:.3f} log_loss={calibrated_metrics.log_loss:.3f} "
        f"(uncalibrated: brier_score={metrics.brier_score:.3f} log_loss={metrics.log_loss:.3f})"
    )

    print("\n-- Saving and reloading the trained model --")
    with tempfile.TemporaryDirectory(prefix="tt-edge-ml-demo-") as tmp_dir:
        model_path = str(Path(tmp_dir) / "logreg-player_a_won.joblib")
        metadata_path = str(Path(tmp_dir) / "logreg-player_a_won.json")

        train_as_of = [matrix.rows[i].as_of for i in plan.train_indices]
        entry = ModelRegistryEntry(
            model_id="logreg-player_a_won-demo",
            model_version="0.1.0-demo",
            training_timestamp=datetime.now(timezone.utc),
            feature_schema_version=matrix.schema_.version,
            target_kind=TargetKind.PLAYER_A_WON,
            training_period_start=min(train_as_of),
            training_period_end=max(train_as_of),
            metrics=metrics,
            hyperparameters={"random_state": 0, "max_iter": 1000},
            serialization_format="joblib",
            model_path=model_path,
            metadata_path=metadata_path,
        )

        save_model(model, model_path)
        save_metadata(entry, metadata_path)
        print(f"  saved model -> {model_path}")
        print(f"  saved metadata -> {metadata_path}")

        reloaded_model = load_model(model_path)
        reloaded_entry = load_metadata(metadata_path)
        reloaded_proba = np.asarray(reloaded_model.predict_proba(X_test))[:, 1]
        original_proba = np.asarray(result.predicted_probabilities)
        max_diff = float(np.max(np.abs(reloaded_proba - original_proba)))
        print(f"  reloaded model prediction max abs diff vs original: {max_diff:.2e}")
        print(f"  reloaded metadata: model_id={reloaded_entry.model_id} model_version={reloaded_entry.model_version}")

        registry = InMemoryModelRegistry()
        registry.register(entry)
        print("\n-- Model registry --")
        print(f"  registered entries: {registry.count()}")
        for registered in registry.list_entries():
            print(
                f"  - {registered.model_id} v{registered.model_version} "
                f"trained_at={registered.training_timestamp.isoformat()} "
                f"schema={registered.feature_schema_version} "
                f"period=[{registered.training_period_start}, {registered.training_period_end}]"
            )

    print(
        "\nAll data above is illustrative/mock (fixed, hand-authored match records). This demo "
        "trains only a baseline logistic regression to verify the training infrastructure — it is "
        "not a production prediction model, and there is no live inference. Output is analytical "
        "decision support only; this system never suggests or places a wager."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
