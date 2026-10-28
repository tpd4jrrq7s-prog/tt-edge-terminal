"""Convert `FeatureSnapshot` objects into a stable, versioned numeric `FeatureMatrix`.

Every feature is defined once, in a fixed order, by a small
`_FeatureDefinition` (name + kind + a pure extractor function). That
list *is* the stable ordering and the schema — bumping
`FEATURE_SCHEMA_VERSION` is required before ever changing it, so a
consumer can always tell whether two matrices are comparable.

Window-based per-player features are pulled from a fixed set of
canonical window labels (`all_time`, `last_5`, `last_10`, `last_20`)
regardless of what `HistoricalIntelligenceSettings.rolling_window_sizes`
produced for a given snapshot — a window that isn't present is simply
missing (`None`), never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from features.models import FeatureSnapshot, PlayerRollingFeatures, WindowFeatures
from ml.models import FeatureKind, FeatureMatrix, FeatureRow, FeatureSchema

FEATURE_SCHEMA_VERSION = "1.0.0"

_CANONICAL_WINDOW_LABELS = ("all_time", "last_5", "last_10", "last_20")

_WINDOW_FLOAT_FIELDS = (
    "win_rate",
    "set_win_rate",
    "point_win_rate",
    "average_set_margin",
    "average_point_margin",
    "straight_sets_win_rate",
    "deciding_set_appearance_rate",
    "deciding_set_win_rate",
    "first_set_win_rate",
    "comeback_win_rate",
    "loss_rate_after_winning_first_set",
    "average_match_duration_minutes",
)
_WINDOW_INT_FIELDS = ("matches_played", "incomplete_match_count")

_PLAYER_FLOAT_FIELDS = (
    "rest_hours_since_previous_match",
    "recency_weighted_win_rate",
    "volatility_score",
    "historical_data_quality_average",
)
_PLAYER_INT_FIELDS = ("matches_in_previous_24h", "matches_in_previous_7d", "result_streak", "observation_count")

_MATCHUP_FLOAT_FIELDS = (
    "player_a_win_rate",
    "player_b_win_rate",
    "recent_head_to_head_win_rate_player_a",
    "average_set_margin_player_a",
    "deciding_set_h2h_win_rate_player_a",
    "days_since_last_meeting",
    "competition_specific_h2h_win_rate_player_a",
    "format_specific_h2h_win_rate_player_a",
    "ranking_differential",
    "rest_differential_hours",
    "workload_differential",
    "form_differential",
    "volatility_differential",
)
_MATCHUP_INT_FIELDS = (
    "head_to_head_matches",
    "recent_head_to_head_n",
    "deciding_set_h2h_matches",
    "competition_specific_h2h_matches",
    "format_specific_h2h_matches",
)


@dataclass(frozen=True)
class _FeatureDefinition:
    name: str
    kind: FeatureKind
    extractor: Callable[[FeatureSnapshot], float | int | bool | None]


def _player_features(side: str) -> Callable[[FeatureSnapshot], PlayerRollingFeatures]:
    return (lambda s: s.player_a_features) if side == "a" else (lambda s: s.player_b_features)


def _window_value(player_getter, label: str, field: str):
    def extract(snapshot: FeatureSnapshot):
        window: WindowFeatures | None = player_getter(snapshot).window(label)
        if window is None:
            return None
        return getattr(window, field)

    return extract


def _player_value(player_getter, field: str):
    def extract(snapshot: FeatureSnapshot):
        return getattr(player_getter(snapshot), field)

    return extract


def _matchup_value(field: str):
    def extract(snapshot: FeatureSnapshot):
        return getattr(snapshot.matchup_features, field)

    return extract


def _build_feature_definitions() -> list[_FeatureDefinition]:
    definitions: list[_FeatureDefinition] = []

    for side in ("a", "b"):
        getter = _player_features(side)
        for label in _CANONICAL_WINDOW_LABELS:
            for field in _WINDOW_FLOAT_FIELDS:
                definitions.append(
                    _FeatureDefinition(f"player_{side}.{label}.{field}", FeatureKind.FLOAT, _window_value(getter, label, field))
                )
            for field in _WINDOW_INT_FIELDS:
                definitions.append(
                    _FeatureDefinition(f"player_{side}.{label}.{field}", FeatureKind.INT, _window_value(getter, label, field))
                )
        for field in _PLAYER_FLOAT_FIELDS:
            definitions.append(_FeatureDefinition(f"player_{side}.{field}", FeatureKind.FLOAT, _player_value(getter, field)))
        for field in _PLAYER_INT_FIELDS:
            definitions.append(_FeatureDefinition(f"player_{side}.{field}", FeatureKind.INT, _player_value(getter, field)))
        definitions.append(
            _FeatureDefinition(f"player_{side}.ranking", FeatureKind.INT, _player_value(getter, "ranking"))
        )
        definitions.append(
            _FeatureDefinition(
                f"player_{side}.has_history", FeatureKind.BOOL, lambda s, g=getter: g(s).observation_count > 0
            )
        )

    for field in _MATCHUP_FLOAT_FIELDS:
        definitions.append(_FeatureDefinition(f"matchup.{field}", FeatureKind.FLOAT, _matchup_value(field)))
    for field in _MATCHUP_INT_FIELDS:
        definitions.append(_FeatureDefinition(f"matchup.{field}", FeatureKind.INT, _matchup_value(field)))

    definitions.append(
        _FeatureDefinition("matchup.has_head_to_head", FeatureKind.BOOL, lambda s: s.matchup_features.head_to_head_matches > 0)
    )
    definitions.append(
        _FeatureDefinition(
            "matchup.has_ranking_data", FeatureKind.BOOL,
            lambda s: s.player_a_features.ranking is not None and s.player_b_features.ranking is not None,
        )
    )
    definitions.append(
        _FeatureDefinition("snapshot.data_quality_score", FeatureKind.FLOAT, lambda s: s.provenance.data_quality_score)
    )

    return definitions


FEATURE_DEFINITIONS: list[_FeatureDefinition] = _build_feature_definitions()


def build_feature_schema() -> FeatureSchema:
    """The stable, versioned schema every FeatureMatrix built by this module conforms to."""
    return FeatureSchema(
        version=FEATURE_SCHEMA_VERSION,
        feature_names=[d.name for d in FEATURE_DEFINITIONS],
        feature_kinds=[d.kind for d in FEATURE_DEFINITIONS],
    )


def _cast(value, kind: FeatureKind) -> tuple[float, bool]:
    if value is None:
        return 0.0, True
    if kind is FeatureKind.BOOL:
        return (1.0 if value else 0.0), False
    return float(value), False


def build_feature_row(snapshot: FeatureSnapshot, source_id: str) -> FeatureRow:
    """Build one FeatureRow from a single FeatureSnapshot, in the stable, canonical order."""
    values: list[float] = []
    missing_mask: list[bool] = []
    for definition in FEATURE_DEFINITIONS:
        raw = definition.extractor(snapshot)
        value, missing = _cast(raw, definition.kind)
        values.append(value)
        missing_mask.append(missing)
    return FeatureRow(source_id=source_id, as_of=snapshot.as_of, values=values, missing_mask=missing_mask)


def build_feature_matrix(snapshots: list[tuple[str, FeatureSnapshot]]) -> FeatureMatrix:
    """Build a full FeatureMatrix from `(source_id, snapshot)` pairs, preserving input order."""
    schema = build_feature_schema()
    rows = [build_feature_row(snapshot, source_id) for source_id, snapshot in snapshots]
    return FeatureMatrix(schema=schema, rows=rows)
