"""Tests for extracting typed TargetVectors from TrainingExamples."""

from __future__ import annotations

import pytest

from ml.errors import MissingTargetValueError
from ml.models import TargetKind
from ml.targets import TARGET_SCHEMA_VERSION, extract_targets
from tests.ml._fixtures import build_examples


def test_player_a_won_matches_examples():
    examples = build_examples(6)
    vector = extract_targets(examples, TargetKind.PLAYER_A_WON)
    assert vector.values == [float(e.player_a_won) for e in examples]
    assert vector.source_ids == [e.id for e in examples]
    assert vector.schema_version == TARGET_SCHEMA_VERSION


def test_player_b_won_is_complement_of_player_a_won():
    examples = build_examples(6)
    a = extract_targets(examples, TargetKind.PLAYER_A_WON)
    b = extract_targets(examples, TargetKind.PLAYER_B_WON)
    assert all(av + bv == 1.0 for av, bv in zip(a.values, b.values))


def test_best_of_extracted_when_present():
    examples = build_examples(4, best_of=5)
    vector = extract_targets(examples, TargetKind.BEST_OF)
    assert vector.values == [5.0] * 4


def test_best_of_raises_when_missing_rather_than_fabricating():
    examples = build_examples(4, best_of=None)
    with pytest.raises(MissingTargetValueError):
        extract_targets(examples, TargetKind.BEST_OF)


def test_target_vector_size_matches_examples():
    examples = build_examples(5)
    vector = extract_targets(examples, TargetKind.PLAYER_A_WON)
    assert vector.size == len(examples)
