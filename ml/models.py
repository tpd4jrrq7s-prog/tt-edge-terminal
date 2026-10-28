"""Typed public contracts for the ML foundation layer.

No public function in `ml/` returns a loose dictionary — numeric data
is always a `FeatureMatrix`/`TargetVector` (or a `numpy.ndarray` derived
from one), with feature names/kinds carried alongside as an explicit,
versioned schema.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class FeatureKind(str, Enum):
    """The original semantic type of a feature, before numeric casting for the model matrix."""

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"


class FeatureSchema(BaseModel):
    """The stable, versioned, ordered description of every column in a FeatureMatrix."""

    version: str = Field(..., min_length=1)
    feature_names: list[str] = Field(..., min_length=1)
    feature_kinds: list[FeatureKind]

    @model_validator(mode="after")
    def _validate_shape(self) -> "FeatureSchema":
        if len(self.feature_names) != len(self.feature_kinds):
            raise ValueError("feature_names and feature_kinds must be the same length")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        return self

    @property
    def size(self) -> int:
        return len(self.feature_names)


class FeatureRow(BaseModel):
    """One row of numeric feature values, positionally aligned with a FeatureSchema."""

    source_id: str = Field(..., min_length=1)
    as_of: datetime
    values: list[float]
    missing_mask: list[bool]

    @model_validator(mode="after")
    def _validate_shape(self) -> "FeatureRow":
        if len(self.values) != len(self.missing_mask):
            raise ValueError("values and missing_mask must be the same length")
        return self


class FeatureMatrix(BaseModel):
    """A leakage-safe, stably-ordered numeric feature matrix built from FeatureSnapshots."""

    schema_: FeatureSchema = Field(..., alias="schema")
    rows: list[FeatureRow] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_features(self) -> int:
        return self.schema_.size

    def source_ids(self) -> list[str]:
        return [row.source_id for row in self.rows]

    def as_of_values(self) -> list[datetime]:
        return [row.as_of for row in self.rows]

    def select_rows(self, indices: list[int]) -> "FeatureMatrix":
        """Return a new FeatureMatrix containing only the given row indices, same schema."""
        return FeatureMatrix(schema=self.schema_, rows=[self.rows[i] for i in indices])


class TargetKind(str, Enum):
    PLAYER_A_WON = "player_a_won"
    PLAYER_B_WON = "player_b_won"
    BEST_OF = "best_of"


class TargetVector(BaseModel):
    """A labeled target column, positionally aligned with a FeatureMatrix's rows."""

    kind: TargetKind
    schema_version: str = "1.0.0"
    values: list[float] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_shape(self) -> "TargetVector":
        if len(self.values) != len(self.source_ids):
            raise ValueError("values and source_ids must be the same length")
        return self

    @property
    def size(self) -> int:
        return len(self.values)


class ConfusionMatrix(BaseModel):
    true_negative: int = Field(..., ge=0)
    false_positive: int = Field(..., ge=0)
    false_negative: int = Field(..., ge=0)
    true_positive: int = Field(..., ge=0)


class ClassificationMetrics(BaseModel):
    """Standard binary-classification evaluation metrics, always paired with sample_size."""

    sample_size: int = Field(..., ge=0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    roc_auc: float | None = Field(default=None, description="None when only one class is present in y_true")
    log_loss: float = Field(..., ge=0.0)
    brier_score: float = Field(..., ge=0.0, le=1.0)
    confusion_matrix: ConfusionMatrix


class CalibrationBin(BaseModel):
    bin_index: int = Field(..., ge=0)
    mean_predicted_probability: float = Field(..., ge=0.0, le=1.0)
    fraction_of_positives: float = Field(..., ge=0.0, le=1.0)
    count: int = Field(..., ge=0)


class CalibrationCurve(BaseModel):
    n_bins: int = Field(..., ge=1)
    bins: list[CalibrationBin] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """The full evaluation output for one fitted model against one (typically held-out) split."""

    metrics: ClassificationMetrics
    calibration_curve: CalibrationCurve
    predicted_probabilities: list[float]
    predicted_labels: list[int]
    source_ids: list[str]


class ModelRegistryEntry(BaseModel):
    """Everything needed to audit, reproduce, or reload one trained model."""

    model_id: str = Field(..., min_length=1)
    model_version: str = Field(..., min_length=1)
    training_timestamp: datetime
    feature_schema_version: str
    target_kind: TargetKind
    training_period_start: datetime | None = None
    training_period_end: datetime | None = None
    metrics: ClassificationMetrics
    hyperparameters: dict[str, float | int | str | bool] = Field(default_factory=dict)
    serialization_format: str = "joblib"
    model_path: str
    metadata_path: str
