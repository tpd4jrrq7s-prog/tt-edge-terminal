"""Assemble a leakage-safe `FeatureSnapshot` from already-fetched, pre-filtered inputs.

This module performs the final, in-process leakage assertions
(`SnapshotLeakageError`) right before returning a snapshot — a last line
of defense even though callers (see `features.builder`) are expected to
have already fetched only eligible records.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from config.historical import HistoricalIntelligenceSettings, get_historical_intelligence_settings
from features.errors import SnapshotLeakageError
from features.matchup import build_matchup_features
from features.models import FeatureSnapshot, MatchupFeatures, PlayerRollingFeatures, ProvenanceMetadata
from features.player import build_player_rolling_features
from persistence.models import HistoricalMatchRecord


def compute_fingerprint(parts: list[str]) -> str:
    """A stable, deterministic fingerprint over string parts (never runtime/wall-clock values)."""
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _assert_no_leakage(
    target_match_id: str,
    as_of: datetime,
    source_matches: list[HistoricalMatchRecord],
) -> None:
    for match in source_matches:
        if match.id == target_match_id:
            raise SnapshotLeakageError(
                f"Target match {target_match_id!r} appeared in its own feature history"
            )
        if match.effective_timestamp >= as_of:
            raise SnapshotLeakageError(
                f"Source match {match.id!r} has effective_timestamp "
                f"{match.effective_timestamp.isoformat()} >= as_of {as_of.isoformat()}"
            )


def _data_quality(
    player_a_features: PlayerRollingFeatures,
    player_b_features: PlayerRollingFeatures,
    matchup_features: MatchupFeatures,
    source_matches: list[HistoricalMatchRecord],
    settings: HistoricalIntelligenceSettings,
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    quality_scores = [m.data_quality.completeness_score for m in source_matches]
    score = (sum(quality_scores) / len(quality_scores)) if quality_scores else 100.0

    if player_a_features.observation_count < settings.min_observations_for_reliable_rate:
        score -= 15.0
        warnings.append("player_a has fewer observations than the reliable-rate minimum")
    if player_b_features.observation_count < settings.min_observations_for_reliable_rate:
        score -= 15.0
        warnings.append("player_b has fewer observations than the reliable-rate minimum")
    if matchup_features.head_to_head_matches == 0:
        score -= 5.0
        warnings.append("no head-to-head history available")

    return max(0.0, min(100.0, score)), warnings


def _missing_feature_names(
    player_a_features: PlayerRollingFeatures, player_b_features: PlayerRollingFeatures, matchup_features: MatchupFeatures
) -> list[str]:
    missing: list[str] = []
    a_all_time = player_a_features.window("all_time")
    b_all_time = player_b_features.window("all_time")
    if a_all_time is None or a_all_time.win_rate is None:
        missing.append("player_a.win_rate")
    if b_all_time is None or b_all_time.win_rate is None:
        missing.append("player_b.win_rate")
    if player_a_features.recency_weighted_win_rate is None:
        missing.append("player_a.recency_weighted_win_rate")
    if player_b_features.recency_weighted_win_rate is None:
        missing.append("player_b.recency_weighted_win_rate")
    if matchup_features.player_a_win_rate is None:
        missing.append("matchup.player_a_win_rate")
    return missing


def build_feature_snapshot(
    snapshot_id: str,
    target_match_id: str,
    player_a_id: str,
    player_b_id: str,
    as_of: datetime,
    player_a_history: list[HistoricalMatchRecord],
    player_b_history: list[HistoricalMatchRecord],
    head_to_head_history: list[HistoricalMatchRecord],
    target_competition_id: str | None = None,
    target_best_of: int | None = None,
    settings: HistoricalIntelligenceSettings | None = None,
    player_a_ranking: int | None = None,
    player_b_ranking: int | None = None,
) -> FeatureSnapshot:
    """Assemble a leakage-safe FeatureSnapshot from pre-fetched, pre-filtered history.

    Every input list must already exclude the target match and contain
    only records strictly before `as_of` — this is asserted, not assumed.
    `player_a_ranking`/`player_b_ranking`, if supplied, must already be
    each player's most recent ranking strictly before `as_of` — never a
    "current" ranking backfilled into a historical snapshot.
    """
    settings = settings or get_historical_intelligence_settings()

    all_source_matches = {m.id: m for m in [*player_a_history, *player_b_history, *head_to_head_history]}
    _assert_no_leakage(target_match_id, as_of, list(all_source_matches.values()))

    player_a_features = build_player_rolling_features(
        player_a_id, player_a_history, as_of, settings, latest_ranking=player_a_ranking
    )
    player_b_features = build_player_rolling_features(
        player_b_id, player_b_history, as_of, settings, latest_ranking=player_b_ranking
    )
    matchup_features = build_matchup_features(
        player_a_id,
        player_b_id,
        head_to_head_history,
        player_a_features,
        player_b_features,
        as_of,
        target_competition_id=target_competition_id,
        target_best_of=target_best_of,
        settings=settings,
    )

    data_quality_score, quality_warnings = _data_quality(
        player_a_features, player_b_features, matchup_features, list(all_source_matches.values()), settings
    )
    missing = _missing_feature_names(player_a_features, player_b_features, matchup_features)

    player_a_ids = sorted(m.id for m in player_a_history)
    player_b_ids = sorted(m.id for m in player_b_history)
    h2h_ids = sorted(m.id for m in head_to_head_history)

    repository_fingerprint = compute_fingerprint([*player_a_ids, *player_b_ids, *h2h_ids])
    input_fingerprint = compute_fingerprint(
        [
            target_match_id,
            player_a_id,
            player_b_id,
            as_of.isoformat(),
            settings.feature_schema_version,
            settings.builder_version,
            repository_fingerprint,
        ]
    )

    provenance = ProvenanceMetadata(
        player_a_source_match_ids=player_a_ids,
        player_b_source_match_ids=player_b_ids,
        head_to_head_source_match_ids=h2h_ids,
        cutoff=as_of,
        repository_fingerprint=repository_fingerprint,
        input_fingerprint=input_fingerprint,
        feature_schema_version=settings.feature_schema_version,
        builder_version=settings.builder_version,
        player_a_observation_count=player_a_features.observation_count,
        player_b_observation_count=player_b_features.observation_count,
        head_to_head_observation_count=matchup_features.head_to_head_matches,
        warnings=quality_warnings,
        missing_feature_names=missing,
        data_quality_score=data_quality_score,
    )

    return FeatureSnapshot(
        id=snapshot_id,
        target_match_id=target_match_id,
        as_of=as_of,
        player_a_id=player_a_id,
        player_b_id=player_b_id,
        player_a_features=player_a_features,
        player_b_features=player_b_features,
        matchup_features=matchup_features,
        provenance=provenance,
        feature_schema_version=settings.feature_schema_version,
        builder_version=settings.builder_version,
    )
