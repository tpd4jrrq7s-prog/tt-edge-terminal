"""A `HistoricalDataSource` reading UTF-8 CSV from a local file.

Uses the standard library `csv` module only (no pandas, no pickle).
Delimiter is configurable. Nested structures (e.g. set scores) are not
supported by flat CSV — a CSV source is intended for record types with
scalar fields only (players, odds, rankings, competitions); use JSONL
for matches with embedded sets.
"""

from __future__ import annotations

import csv

from historical_ingestion.errors import SourceReadError
from historical_ingestion.sources.base import FileHistoricalDataSource


class CSVFileSource(FileHistoricalDataSource):
    """Reads one raw record per row from a `.csv` file (first row is the header)."""

    def __init__(
        self,
        path: str,
        provider: str,
        record_type: str,
        delimiter: str = ",",
        batch_size: int = 100,
        source_name: str | None = None,
        id_field: str = "id",
    ) -> None:
        super().__init__(path, provider, batch_size=batch_size, source_name=source_name, id_field=id_field)
        self._record_type_value = record_type
        self._delimiter = delimiter

    def _record_type(self) -> str:
        return self._record_type_value

    def _read_rows(self) -> list[dict]:
        rows: list[dict] = []
        with open(self._path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=self._delimiter)
            if reader.fieldnames is None:
                raise SourceReadError(f"{self._path}: CSV file has no header row")
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise SourceReadError(f"{self._path}:{row_number}: row has more columns than the header")
                rows.append(dict(row))
        return rows
