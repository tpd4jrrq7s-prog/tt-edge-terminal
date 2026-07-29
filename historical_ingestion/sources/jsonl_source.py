"""A `HistoricalDataSource` reading UTF-8 JSON Lines from a local file.

Each line must be a JSON object; malformed lines raise a clear
`SourceReadError` naming the offending line number rather than silently
skipping or fabricating a record.
"""

from __future__ import annotations

import json

from historical_ingestion.errors import SourceReadError
from historical_ingestion.sources.base import FileHistoricalDataSource


class JSONLFileSource(FileHistoricalDataSource):
    """Reads one raw record per line from a `.jsonl` file."""

    def __init__(
        self,
        path: str,
        provider: str,
        record_type: str,
        batch_size: int = 100,
        source_name: str | None = None,
        id_field: str = "id",
    ) -> None:
        super().__init__(path, provider, batch_size=batch_size, source_name=source_name, id_field=id_field)
        self._record_type_value = record_type

    def _record_type(self) -> str:
        return self._record_type_value

    def _read_rows(self) -> list[dict]:
        rows: list[dict] = []
        with open(self._path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise SourceReadError(f"{self._path}:{line_number}: invalid JSON ({exc})") from exc
                if not isinstance(row, dict):
                    raise SourceReadError(f"{self._path}:{line_number}: expected a JSON object, got {type(row).__name__}")
                rows.append(row)
        return rows
