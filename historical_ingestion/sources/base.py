"""Shared helpers and a common base class for local-file `HistoricalDataSource`s.

`FileHistoricalDataSource` implements the cursor-as-line-offset
resumption logic shared by JSONL and CSV sources; subclasses only need
to implement per-row parsing. No external network calls are possible
through this base class — it only ever reads a local file path.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from historical_ingestion.errors import SourceReadError
from historical_ingestion.models import RawRecord, SourceBatch, SourceHealth


def compute_batch_checksum(records: list[RawRecord]) -> str:
    """A deterministic checksum over a batch's raw record payloads."""
    parts = [json.dumps(r.payload, sort_keys=True, default=str) for r in records]
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


class FileHistoricalDataSource(ABC):
    """Reads raw records from a local file, resuming via a line-offset cursor."""

    def __init__(
        self,
        path: str,
        provider: str,
        batch_size: int = 100,
        source_name: str | None = None,
        id_field: str = "id",
    ) -> None:
        self.name = source_name or f"file:{path}"
        self.provider = provider
        self.version = "1.0.0"
        self._path = path
        self._batch_size = batch_size
        self._id_field = id_field

    @abstractmethod
    def _record_type(self) -> str:
        """The `RawRecord.record_type` every row in this file represents."""

    @abstractmethod
    def _read_rows(self) -> list[dict]:
        """Read and parse every row in the file into raw payload dicts, in stable file order."""

    def fetch_batch(self, cursor: str | None) -> SourceBatch:
        """Read up to `batch_size` rows starting after `cursor` (a stringified row offset)."""
        offset = int(cursor) if cursor else 0
        try:
            rows = self._read_rows()
        except (OSError, ValueError) as exc:
            raise SourceReadError(f"Failed to read {self._path!r}: {exc}") from exc

        window = rows[offset : offset + self._batch_size]
        now = datetime.now(timezone.utc)

        records: list[RawRecord] = []
        for i, row in enumerate(window):
            row_number = offset + i
            provider_record_id = str(row.get(self._id_field, row_number))
            records.append(
                RawRecord(
                    provider=self.provider,
                    provider_record_id=provider_record_id,
                    record_type=self._record_type(),
                    payload=row,
                    source_batch_id=f"{self.name}:{offset}",
                    source_timestamp=now,
                    fetched_at=now,
                )
            )

        next_offset = offset + len(window)
        next_cursor = str(next_offset) if next_offset < len(rows) else None

        return SourceBatch(
            source_name=self.name,
            provider=self.provider,
            source_version=self.version,
            batch_id=f"{self.name}:{offset}",
            records=records,
            next_cursor=next_cursor,
            source_timestamp=now,
            fetched_at=now,
            source_metadata={"path": self._path, "row_offset": str(offset)},
            checksum=compute_batch_checksum(records),
        )

    def health(self) -> SourceHealth:
        import os

        exists = os.path.isfile(self._path)
        return SourceHealth(
            source_name=self.name,
            provider=self.provider,
            healthy=exists,
            detail="file present" if exists else f"file not found: {self._path}",
            checked_at=datetime.now(timezone.utc),
        )
