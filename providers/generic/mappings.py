"""A working, example `ProviderMappingConfig` for the deterministic mock provider.

This is the reference mapping used by `historical_ingestion.sources.mock_provider`
and the Phase 4 demo — a template for wiring up a real provider's own
field names later without touching any other module.
"""

from __future__ import annotations

from providers.models import ProviderMappingConfig

MOCK_PROVIDER_STATUS_MAP: dict[str, str] = {
    "SCHEDULED": "scheduled",
    "LIVE": "live",
    "COMPLETED": "finished",
    "RETIRED": "retired",
    "CANCELLED": "cancelled",
}


def mock_provider_mapping(mapping_version: str = "1.0.0") -> ProviderMappingConfig:
    """Build the default field mapping for the `mock` provider."""
    return ProviderMappingConfig(
        provider="mock",
        mapping_version=mapping_version,
        player_field_map={
            "name": "name",
            "country": "country",
            "ranking_hint": "ranking_hint",
        },
        match_field_map={
            "competition_id": "competition_id",
            "competition_name": "competition_name",
            "player_a_external_id": "player_a_id",
            "player_b_external_id": "player_b_id",
            "scheduled_at": "scheduled_at",
            "actual_start_at": "actual_start_at",
            "completed_at": "completed_at",
            "status": "status",
            "best_of": "best_of",
            "winner_external_id": "winner_id",
        },
        odds_field_map={
            "provider_match_id": "match_id",
            "bookmaker": "bookmaker",
            "selection_external_id": "selection_id",
            "decimal_odds": "decimal_odds",
            "captured_at": "captured_at",
            "market_id": "market_id",
        },
        competition_field_map={
            "name": "name",
            "country": "country",
            "level": "level",
            "format": "format",
            "season": "season",
            "indoor": "indoor",
            "active_from": "active_from",
            "active_to": "active_to",
        },
        ranking_field_map={
            "player_external_id": "player_id",
            "ranking": "ranking",
            "ranking_points": "ranking_points",
            "effective_at": "effective_at",
        },
        sets_key="sets",
        set_number_key="set_number",
        set_a_points_key="a",
        set_b_points_key="b",
        status_map=dict(MOCK_PROVIDER_STATUS_MAP),
        unknown_status_policy="warn",
    )
