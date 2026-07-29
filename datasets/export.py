"""Deterministic dataset export to JSON Lines and CSV.

No pickle, no arbitrary-code-execution formats. Column/key ordering is
always alphabetical (computed from the actual union of keys present),
so output is stable regardless of dict-construction order. Writes are
atomic (write to a temp file in the same directory, then `os.replace`).
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from typing import Any

from config.historical import HistoricalIntelligenceSettings, get_historical_intelligence_settings
from datasets.models import DatasetManifest, TrainingExample


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value.keys()):
            child_prefix = f"{prefix}.{key}" if prefix else key
            result.update(_flatten(value[key], child_prefix))
        return result
    return {prefix: value}


def _round_floats(row: dict[str, Any], precision: int) -> dict[str, Any]:
    return {k: (round(v, precision) if isinstance(v, float) else v) for k, v in row.items()}


def _row_for(example: TrainingExample, precision: int) -> dict[str, Any]:
    base = {
        "id": example.id,
        "target_match_id": example.target_match_id,
        "as_of": example.as_of.isoformat(),
        "player_a_id": example.player_a_id,
        "player_b_id": example.player_b_id,
        "player_a_won": example.player_a_won,
        "feature_schema_version": example.feature_schema_version,
        "builder_version": example.builder_version,
        "provenance_fingerprint": example.provenance_fingerprint,
    }
    feature_fields = _flatten(example.features.model_dump(mode="json"), prefix="features")
    return _round_floats({**base, **feature_fields}, precision)


def _atomic_write(path: str, content: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-export-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def export_jsonl(
    examples: list[TrainingExample], path: str, settings: HistoricalIntelligenceSettings | None = None
) -> None:
    """Export training examples as JSON Lines, one alphabetically-keyed object per line."""
    settings = settings or get_historical_intelligence_settings()
    rows = [_row_for(e, settings.export_float_precision) for e in examples]
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
    content = "\n".join(lines) + ("\n" if lines else "")
    _atomic_write(path, content)


def export_csv(
    examples: list[TrainingExample], path: str, settings: HistoricalIntelligenceSettings | None = None
) -> None:
    """Export training examples as CSV with a stable, alphabetically-ordered column header."""
    settings = settings or get_historical_intelligence_settings()
    rows = [_row_for(e, settings.export_float_precision) for e in examples]
    columns = sorted({key for row in rows for key in row})

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        normalized = {
            col: ("" if row.get(col) is None else _csv_cell(row.get(col))) for col in columns
        }
        writer.writerow(normalized)
    _atomic_write(path, buffer.getvalue())


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def export_manifest(manifest: DatasetManifest, path: str) -> None:
    """Export a DatasetManifest as pretty-printed, alphabetically-keyed JSON."""
    content = json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
    _atomic_write(path, content)
