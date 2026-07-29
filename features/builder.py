"""HistoricalFeatureBuilder: the single orchestrated entrypoint for building
one leakage-safe FeatureSnapshot from repository-backed historical data.

Dependency injection only — the repository is injected, there is no
global state, no database, and no external calls. Snapshot IDs default
to a deterministic function of (target_match_id, as_of) so rebuilding
from identical inputs is trivially reproducible; callers may inject
their own `id_factory` for a different (still-deterministic) ID scheme.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from config.historical import HistoricalIntelligenceSettings, get_historical_intelligence_settings
from features.errors import TargetMatchNotFoundError
from features.models import FeatureSnapshot
from features.snapshots import build_feature_snapshot
from persistence.protocols import MatchRepository, RankingRepository


class HistoricalFeatureBuilder:
    """Builds leakage-safe FeatureSnapshots from a `MatchRepository`.

    `ranking_repository` is optional — when supplied, each player's most
    recent ranking strictly before `as_of` is looked up and threaded into
    the snapshot; without it, ranking fields stay `None` exactly as in
    Phase 3.
    """

    def __init__(
        self,
        match_repository: MatchRepository,
        settings: HistoricalIntelligenceSettings | None = None,
        id_factory: Callable[[str, datetime], str] | None = None,
        ranking_repository: RankingRepository | None = None,
    ) -> None:
        self._matches = match_repository
        self._settings = settings or get_historical_intelligence_settings()
        self._id_factory = id_factory or (lambda match_id, as_of: f"{match_id}@{as_of.isoformat()}")
        self._rankings = ranking_repository

    def build(
        self,
        target_match_id: str,
        as_of: datetime,
        target_competition_id: str | None = None,
        target_best_of: int | None = None,
    ) -> FeatureSnapshot:
        """Build one FeatureSnapshot for `target_match_id` as of `as_of`."""
        target = self._matches.get(target_match_id)
        if target is None:
            raise TargetMatchNotFoundError(f"Target match {target_match_id!r} not found")

        player_a_history = self._matches.list_player_matches_before(target.player_a_id, as_of)
        player_b_history = self._matches.list_player_matches_before(target.player_b_id, as_of)
        head_to_head_history = self._matches.list_head_to_head_before(
            target.player_a_id, target.player_b_id, as_of
        )

        player_a_ranking = None
        player_b_ranking = None
        if self._rankings is not None:
            a_ranking = self._rankings.latest_before(target.player_a_id, as_of)
            b_ranking = self._rankings.latest_before(target.player_b_id, as_of)
            player_a_ranking = a_ranking.ranking if a_ranking is not None else None
            player_b_ranking = b_ranking.ranking if b_ranking is not None else None

        return build_feature_snapshot(
            snapshot_id=self._id_factory(target_match_id, as_of),
            target_match_id=target_match_id,
            player_a_id=target.player_a_id,
            player_b_id=target.player_b_id,
            as_of=as_of,
            player_a_history=player_a_history,
            player_b_history=player_b_history,
            head_to_head_history=head_to_head_history,
            target_competition_id=target_competition_id if target_competition_id is not None else target.competition_id,
            target_best_of=target_best_of if target_best_of is not None else target.best_of,
            settings=self._settings,
            player_a_ranking=player_a_ranking,
            player_b_ranking=player_b_ranking,
        )
