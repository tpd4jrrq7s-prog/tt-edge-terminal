"""Typed, validated configuration for the Phase 3 historical intelligence platform.

Read from environment variables (prefixed `HISTORICAL_`) or a local
`.env` file, falling back to the documented defaults below. Invalid
configuration fails fast with a clear `pydantic.ValidationError`.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class HistoricalIntelligenceSettings(BaseSettings):
    """Central, validated configuration for persistence, identity, features, and datasets."""

    model_config = SettingsConfigDict(
        env_prefix="HISTORICAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Rolling feature windows
    rolling_window_sizes: list[int] = Field(default_factory=lambda: [5, 10, 20])
    form_recency_half_life_days: float = Field(default=90.0, gt=0)
    min_observations_for_reliable_rate: int = Field(default=1, ge=1)

    # Identity resolution
    identity_match_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    identity_ambiguity_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    short_name_length_threshold: int = Field(default=4, ge=1)
    short_name_extra_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    country_mismatch_penalty: float = Field(default=0.7, ge=0.0, le=1.0)
    birth_date_mismatch_penalty: float = Field(default=0.6, ge=0.0, le=1.0)

    # Dataset cutoff policy
    dataset_cutoff_policy: Literal["scheduled_at", "actual_start_at"] = "scheduled_at"
    feature_schema_version: str = Field(default="1.0.0")
    builder_version: str = Field(default="1.0.0")

    # Splits
    split_train_ratio: float = Field(default=0.6, gt=0.0, lt=1.0)
    split_validation_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    split_test_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)

    # Leakage / export
    strict_leakage_mode: bool = True
    export_float_precision: int = Field(default=6, ge=0, le=15)

    @field_validator("rolling_window_sizes")
    @classmethod
    def _validate_windows(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("rolling_window_sizes must not be empty")
        if any(v <= 0 for v in value):
            raise ValueError("rolling_window_sizes must all be positive")
        if len(set(value)) != len(value):
            raise ValueError("rolling_window_sizes must not contain duplicates")
        if value != sorted(value):
            raise ValueError("rolling_window_sizes must be sorted ascending")
        return value

    @field_validator("feature_schema_version", "builder_version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        if not _VERSION_PATTERN.match(value):
            raise ValueError(f"version identifiers must look like 'X.Y.Z' (got {value!r})")
        return value

    @model_validator(mode="after")
    def _validate_split_ratios(self) -> "HistoricalIntelligenceSettings":
        total = self.split_train_ratio + self.split_validation_ratio + self.split_test_ratio
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"split ratios must sum to 1.0 (got {total:.6f})")
        return self


@lru_cache
def get_historical_intelligence_settings() -> HistoricalIntelligenceSettings:
    """Return a cached, validated HistoricalIntelligenceSettings instance."""
    return HistoricalIntelligenceSettings()
