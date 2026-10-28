"""Phase 5A: Machine Learning Foundation.

Infrastructure for training baseline classification models
reproducibly from `datasets.TrainingExample`s: a versioned numeric
feature matrix, target extraction, leakage-safe preprocessing,
chronological-only data splits, baseline model interfaces, evaluation
metrics, post-hoc calibration, and deterministic model/metadata
serialization plus an in-memory registry.

This package builds training infrastructure only — it performs no live
inference, trains no production model, and never suggests or places a
wager.
"""

from __future__ import annotations
