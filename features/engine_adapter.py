"""Adapter: convert a Phase 3 FeatureSnapshot into Phase 2B engine inputs, where clean.

Does not rewrite or wrap the Phase 2B engine, and does not fabricate a
`domain.match.Match` or `ingestion.models.RawMatch` — those must come
from the caller's own real match data, exactly as the engine already
expects. Only the *optional* historical-context inputs that a
`FeatureSnapshot` can cleanly and honestly supply are converted here:

- `engine.models.HeadToHeadRecord` (derived from aggregate head-to-head
  win rate + count — "no data" stays "no data", never a fabricated
  50/50 record)
- `engine.models.CompetitionContext` (pass-through of caller-supplied
  competition/format metadata; a `FeatureSnapshot` does not itself carry
  the *target* match's own competition/format, only aggregate history)

`PlayerRollingFeatures`/`MatchupFeatures` are richer, aggregate,
recency-weighted signals with no per-match analogue in the Phase 2B
engine's simple `HistoricalMatch`/`PointEvent` inputs. Rather than
lossily reconstructing fake per-match records to force-fit them, this
adapter leaves them out and reports them as `unconvertible_features` —
preserved as-is on the `FeatureSnapshot` for a future ML model to
consume directly.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel

from engine.models import CompetitionContext, HeadToHeadRecord
from features.models import FeatureSnapshot

_UNCONVERTIBLE_FEATURES = (
    "player_a_features.windows (rolling per-window stats)",
    "player_b_features.windows (rolling per-window stats)",
    "player_a_features.recency_weighted_win_rate",
    "player_b_features.recency_weighted_win_rate",
    "player_a_features.volatility_score",
    "player_b_features.volatility_score",
    "matchup_features.competition_specific_h2h_win_rate_player_a",
    "matchup_features.format_specific_h2h_win_rate_player_a",
)


class AdaptedEngineInputs(BaseModel):
    """Whatever historical-context inputs could be cleanly derived for the Phase 2B engine."""

    head_to_head: HeadToHeadRecord
    context: CompetitionContext | None = None
    unconvertible_features: list[str]


def _to_head_to_head(snapshot: FeatureSnapshot) -> HeadToHeadRecord:
    matchup = snapshot.matchup_features
    if matchup.head_to_head_matches == 0 or matchup.player_a_win_rate is None:
        return HeadToHeadRecord()

    player_one_wins = round(matchup.head_to_head_matches * matchup.player_a_win_rate)
    player_two_wins = matchup.head_to_head_matches - player_one_wins

    last_played_at = None
    if matchup.days_since_last_meeting is not None:
        last_played_at = snapshot.as_of - timedelta(days=matchup.days_since_last_meeting)

    return HeadToHeadRecord(
        player_one_wins=player_one_wins, player_two_wins=player_two_wins, last_played_at=last_played_at
    )


def adapt_snapshot_to_engine_inputs(
    snapshot: FeatureSnapshot,
    *,
    competition_name: str | None = None,
    best_of_sets: int | None = None,
) -> AdaptedEngineInputs:
    """Convert a FeatureSnapshot into the subset of engine inputs it can honestly supply.

    `competition_name`/`best_of_sets` are pass-through values the caller
    already knows from their own match record — a FeatureSnapshot does
    not carry the *target* match's own competition/format, only
    aggregate prior history, so these are not derived here.
    """
    context = None
    if competition_name is not None or best_of_sets is not None:
        context = CompetitionContext(competition_name=competition_name, best_of_sets=best_of_sets)

    return AdaptedEngineInputs(
        head_to_head=_to_head_to_head(snapshot),
        context=context,
        unconvertible_features=list(_UNCONVERTIBLE_FEATURES),
    )
