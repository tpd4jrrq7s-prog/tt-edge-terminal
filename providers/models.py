"""Typed contracts for declarative, configurable provider field mappings.

`ProviderMappingConfig` is the *only* place provider-specific field
names live — `historical_ingestion`'s core service/pipeline never sees
a raw payload key, only the canonical `Imported*` models a
`HistoricalProviderAdapter` produces from one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ProviderMappingConfig(BaseModel):
    """A declarative mapping from one provider's raw payload shape to canonical fields.

    Each `*_field_map` is `{canonical_field_name: raw_payload_key}`. Set
    fields are nested lists, so their inner keys are configured
    separately (`set_number_key`, `set_a_points_key`, `set_b_points_key`,
    `sets_key`).
    """

    provider: str = Field(..., min_length=1)
    mapping_version: str = Field(..., min_length=1)

    player_field_map: dict[str, str] = Field(default_factory=dict)
    match_field_map: dict[str, str] = Field(default_factory=dict)
    odds_field_map: dict[str, str] = Field(default_factory=dict)
    competition_field_map: dict[str, str] = Field(default_factory=dict)
    ranking_field_map: dict[str, str] = Field(default_factory=dict)

    sets_key: str = "sets"
    set_number_key: str = "set_number"
    set_a_points_key: str = "a"
    set_b_points_key: str = "b"

    status_map: dict[str, str] = Field(default_factory=dict)
    unknown_status_policy: Literal["warn", "reject"] = "warn"

    @model_validator(mode="after")
    def _validate_status_map_targets(self) -> "ProviderMappingConfig":
        allowed = {"scheduled", "live", "finished", "retired", "cancelled"}
        bad = {v for v in self.status_map.values() if v not in allowed}
        if bad:
            raise ValueError(f"status_map values must be one of {sorted(allowed)}, got {sorted(bad)}")
        return self
