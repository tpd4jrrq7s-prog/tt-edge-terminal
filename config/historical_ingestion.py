"""Typed, validated configuration for the Phase 4 historical ingestion platform."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


class HistoricalIngestionSettings(BaseSettings):
    """Central, validated configuration for source/adapter/validation/import behavior."""

    model_config = SettingsConfigDict(
        env_prefix="INGEST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    batch_size: int = Field(default=100, ge=1)
    validation_policy: Literal["strict", "lenient"] = "strict"
    warning_policy: Literal["accept", "reject"] = "accept"
    timestamp_tolerance_seconds: float = Field(default=60.0, ge=0.0)
    duplicate_timestamp_window_seconds: float = Field(default=300.0, ge=0.0)
    identity_ambiguity_policy: Literal["quarantine", "reject"] = "quarantine"
    dry_run_default: bool = False
    quarantine_enabled: bool = True
    checkpoint_version: int = Field(default=1, ge=1)
    mapping_version: str = Field(default="1.0.0")
    source_timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_records_per_run: int = Field(default=100_000, ge=1)
    strict_mode: bool = True
    allow_future_records: bool = False

    @field_validator("mapping_version")
    @classmethod
    def _validate_mapping_version(cls, value: str) -> str:
        if not _VERSION_PATTERN.match(value):
            raise ValueError(f"mapping_version must look like 'X.Y.Z' (got {value!r})")
        return value


@lru_cache
def get_historical_ingestion_settings() -> HistoricalIngestionSettings:
    """Return a cached, validated HistoricalIngestionSettings instance."""
    return HistoricalIngestionSettings()
