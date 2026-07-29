"""A generic, configurable `HistoricalProviderAdapter` driven entirely by a `ProviderMappingConfig`.

No provider field names are hardcoded here — they live only in the
`ProviderMappingConfig` instance passed in at construction. Unknown
provider statuses are never silently mapped: depending on
`unknown_status_policy`, they either raise `UnknownProviderStatusError`
or are left unmapped (`status=None`) with an explicit warning attached.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from historical_ingestion.models import (
    ImportedCompetition,
    ImportedMatch,
    ImportedOdds,
    ImportedPlayer,
    ImportedRanking,
    ImportedSet,
    ImportProvenance,
    RawRecord,
)
from providers.errors import ProviderMappingError, UnknownProviderStatusError
from providers.models import ProviderMappingConfig


def _get(payload: dict[str, Any], key: str | None) -> Any:
    if key is None:
        return None
    return payload.get(key)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def raw_fingerprint(payload: dict[str, Any]) -> str:
    """A stable, deterministic fingerprint of a raw payload's content."""
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class GenericProviderAdapter:
    """Maps `RawRecord`s into canonical `Imported*` models using a `ProviderMappingConfig`."""

    def __init__(self, config: ProviderMappingConfig) -> None:
        self._config = config

    @property
    def provider(self) -> str:
        return self._config.provider

    @property
    def mapping_version(self) -> str:
        return self._config.mapping_version

    def _provenance(self, raw: RawRecord, warnings: list[str]) -> ImportProvenance:
        return ImportProvenance(
            provider=raw.provider,
            provider_record_id=raw.provider_record_id,
            source_batch_id=raw.source_batch_id,
            source_timestamp=raw.source_timestamp,
            ingested_at=raw.fetched_at,
            raw_fingerprint=raw_fingerprint(raw.payload),
            mapping_version=self.mapping_version,
            warnings=warnings,
        )

    def _resolve_status(self, payload: dict[str, Any]) -> tuple[str | None, str | None, list[str]]:
        fm = self._config.match_field_map
        raw_status = _get(payload, fm.get("status"))
        raw_status_str = str(raw_status) if raw_status is not None else None
        if raw_status_str is None:
            return None, None, []

        mapped = self._config.status_map.get(raw_status_str)
        if mapped is not None:
            return raw_status_str, mapped, []

        if self._config.unknown_status_policy == "reject":
            raise UnknownProviderStatusError(
                f"Unknown status {raw_status_str!r} for provider {self.provider!r} "
                f"(mapping_version={self.mapping_version!r})"
            )
        return raw_status_str, None, [f"unknown status {raw_status_str!r}; left unmapped"]

    def _adapt_match(self, raw: RawRecord) -> ImportedMatch:
        payload = raw.payload
        fm = self._config.match_field_map
        raw_status, status, warnings = self._resolve_status(payload)

        raw_sets = payload.get(self._config.sets_key) or []
        sets: list[ImportedSet] = []
        for entry in raw_sets:
            if not isinstance(entry, dict):
                warnings.append(f"skipped non-dict set entry: {entry!r}")
                continue
            sets.append(
                ImportedSet(
                    set_number=entry.get(self._config.set_number_key),
                    player_a_points=entry.get(self._config.set_a_points_key),
                    player_b_points=entry.get(self._config.set_b_points_key),
                )
            )

        return ImportedMatch(
            provenance=self._provenance(raw, warnings),
            competition_id=_get(payload, fm.get("competition_id")),
            competition_name=_get(payload, fm.get("competition_name")),
            player_a_external_id=_get(payload, fm.get("player_a_external_id")),
            player_b_external_id=_get(payload, fm.get("player_b_external_id")),
            scheduled_at=_parse_datetime(_get(payload, fm.get("scheduled_at"))),
            actual_start_at=_parse_datetime(_get(payload, fm.get("actual_start_at"))),
            completed_at=_parse_datetime(_get(payload, fm.get("completed_at"))),
            status_raw=raw_status,
            status=status,
            best_of=_get(payload, fm.get("best_of")),
            winner_external_id=_get(payload, fm.get("winner_external_id")),
            sets=sets,
        )

    def _adapt_odds(self, raw: RawRecord) -> ImportedOdds:
        payload = raw.payload
        fm = self._config.odds_field_map
        return ImportedOdds(
            provenance=self._provenance(raw, []),
            provider_match_id=_get(payload, fm.get("provider_match_id")),
            bookmaker=_get(payload, fm.get("bookmaker")),
            selection_external_id=_get(payload, fm.get("selection_external_id")),
            decimal_odds=_get(payload, fm.get("decimal_odds")),
            captured_at=_parse_datetime(_get(payload, fm.get("captured_at"))),
            market_id=_get(payload, fm.get("market_id")),
        )

    def _adapt_player(self, raw: RawRecord) -> ImportedPlayer:
        payload = raw.payload
        fm = self._config.player_field_map
        name = _get(payload, fm.get("name")) or ""
        return ImportedPlayer(
            provenance=self._provenance(raw, []),
            name=name,
            country=_get(payload, fm.get("country")),
            external_player_id=raw.provider_record_id,
            ranking_hint=_get(payload, fm.get("ranking_hint")),
        )

    def _adapt_competition(self, raw: RawRecord) -> ImportedCompetition:
        payload = raw.payload
        fm = self._config.competition_field_map
        name = _get(payload, fm.get("name")) or ""
        return ImportedCompetition(
            provenance=self._provenance(raw, []),
            name=name,
            country=_get(payload, fm.get("country")),
            level=_get(payload, fm.get("level")),
            format=_get(payload, fm.get("format")),
            season=_get(payload, fm.get("season")),
            indoor=_get(payload, fm.get("indoor")),
            active_from=_parse_datetime(_get(payload, fm.get("active_from"))),
            active_to=_parse_datetime(_get(payload, fm.get("active_to"))),
        )

    def _adapt_ranking(self, raw: RawRecord) -> ImportedRanking:
        payload = raw.payload
        fm = self._config.ranking_field_map
        return ImportedRanking(
            provenance=self._provenance(raw, []),
            player_external_id=_get(payload, fm.get("player_external_id")),
            ranking=_get(payload, fm.get("ranking")),
            ranking_points=_get(payload, fm.get("ranking_points")),
            effective_at=_parse_datetime(_get(payload, fm.get("effective_at"))),
        )

    def adapt(
        self, raw: RawRecord
    ) -> ImportedPlayer | ImportedMatch | ImportedOdds | ImportedCompetition | ImportedRanking:
        """Map one raw record into its canonical import model, dispatched on `raw.record_type`."""
        if raw.provider != self.provider:
            raise ProviderMappingError(
                f"Adapter for provider {self.provider!r} cannot adapt a record from provider {raw.provider!r}"
            )
        dispatch = {
            "match": self._adapt_match,
            "odds": self._adapt_odds,
            "player": self._adapt_player,
            "competition": self._adapt_competition,
            "ranking": self._adapt_ranking,
        }
        handler = dispatch.get(raw.record_type)
        if handler is None:
            raise ProviderMappingError(f"Unsupported record_type {raw.record_type!r}")
        return handler(raw)
