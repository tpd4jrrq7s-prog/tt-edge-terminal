"""Protocol interfaces for sources, provider adapters, and checkpoint stores.

Designed so an HTTP-backed source/provider can be added later without
changing `historical_ingestion.service` or `historical_ingestion.pipeline`
at all — both depend only on these Protocols.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from historical_ingestion.checkpoints import ImportCheckpoint
from historical_ingestion.models import (
    ImportedCompetition,
    ImportedMatch,
    ImportedOdds,
    ImportedPlayer,
    ImportedRanking,
    RawRecord,
    SourceBatch,
    SourceHealth,
)


@runtime_checkable
class HistoricalDataSource(Protocol):
    """A provider-independent, checkpoint-aware source of raw historical records."""

    name: str
    provider: str
    version: str

    def fetch_batch(self, cursor: str | None) -> SourceBatch:
        """Fetch/read the next batch starting after `cursor` (None means "from the start")."""
        ...

    def health(self) -> SourceHealth:
        """Report this source's current health/status."""
        ...


@runtime_checkable
class HistoricalProviderAdapter(Protocol):
    """Maps one provider's raw records into canonical import models."""

    provider: str
    mapping_version: str

    def adapt(
        self, raw: RawRecord
    ) -> ImportedPlayer | ImportedMatch | ImportedOdds | ImportedCompetition | ImportedRanking:
        """Map one raw record into its canonical import model, based on `raw.record_type`."""
        ...


@runtime_checkable
class CheckpointStore(Protocol):
    """Durable (within-process) storage for import checkpoints."""

    def get(self, source_name: str, provider: str) -> ImportCheckpoint | None: ...

    def save(self, checkpoint: ImportCheckpoint) -> None: ...

    def reset(self, source_name: str, provider: str) -> None: ...
