"""DatasetBuilder: converts completed historical matches into labeled TrainingExamples.

The builder does not enumerate the repository itself (the
`MatchRepository` protocol intentionally has no "list all" capability)
— callers supply the exact set of match IDs to consider. Only matches
that are actually completed with a recorded winner become training
examples; everything else is skipped with an explicit, logged reason.
"""

from __future__ import annotations

from datetime import datetime, timezone

from config.historical import HistoricalIntelligenceSettings, get_historical_intelligence_settings
from datasets.models import DatasetManifest, LabelDistribution, TrainingExample
from features.builder import HistoricalFeatureBuilder
from features.snapshots import compute_fingerprint
from persistence.models import COMPLETED_STATUSES, HistoricalMatchRecord
from persistence.protocols import MatchRepository


class DatasetBuilder:
    """Builds a chronologically-ordered list of `TrainingExample`s plus a `DatasetManifest`."""

    def __init__(
        self,
        match_repository: MatchRepository,
        feature_builder: HistoricalFeatureBuilder,
        settings: HistoricalIntelligenceSettings | None = None,
    ) -> None:
        self._matches = match_repository
        self._feature_builder = feature_builder
        self._settings = settings or get_historical_intelligence_settings()

    def _cutoff_for(self, match: HistoricalMatchRecord) -> tuple[datetime, str | None]:
        if self._settings.dataset_cutoff_policy == "actual_start_at":
            if match.actual_start_at is not None:
                return match.actual_start_at, None
            return match.scheduled_at, f"match {match.id!r}: actual_start_at missing, used scheduled_at instead"
        return match.scheduled_at, None

    def build(
        self,
        match_ids: list[str],
        dataset_id: str = "dataset",
        created_at: datetime | None = None,
    ) -> tuple[list[TrainingExample], DatasetManifest]:
        """Build training examples for the given match IDs, plus a manifest describing the result."""
        settings = self._settings
        created_at = created_at or datetime.now(timezone.utc)

        examples: list[TrainingExample] = []
        warnings: list[str] = []
        skipped = 0

        for match_id in match_ids:
            match = self._matches.get(match_id)
            if match is None:
                warnings.append(f"match {match_id!r} skipped: not found in repository")
                skipped += 1
                continue
            if match.status not in COMPLETED_STATUSES or match.winner_id is None:
                warnings.append(f"match {match_id!r} skipped: not completed or has no recorded winner")
                skipped += 1
                continue

            cutoff, cutoff_warning = self._cutoff_for(match)
            if cutoff_warning:
                warnings.append(cutoff_warning)

            snapshot = self._feature_builder.build(
                match_id, cutoff, target_competition_id=match.competition_id, target_best_of=match.best_of
            )
            label = 1 if match.winner_id == match.player_a_id else 0

            examples.append(
                TrainingExample(
                    id=f"ex-{match_id}",
                    target_match_id=match_id,
                    as_of=cutoff,
                    player_a_id=match.player_a_id,
                    player_b_id=match.player_b_id,
                    player_a_won=label,
                    features=snapshot,
                    feature_schema_version=settings.feature_schema_version,
                    builder_version=settings.builder_version,
                    provenance_fingerprint=snapshot.provenance.input_fingerprint,
                )
            )

        examples.sort(key=lambda e: (e.as_of, e.id))

        label_distribution = LabelDistribution(
            player_a_wins=sum(1 for e in examples if e.player_a_won == 1),
            player_b_wins=sum(1 for e in examples if e.player_a_won == 0),
        )

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            created_at=created_at,
            feature_schema_version=settings.feature_schema_version,
            builder_version=settings.builder_version,
            source_match_count=len(match_ids),
            training_example_count=len(examples),
            skipped_count=skipped,
            date_range_start=examples[0].as_of if examples else None,
            date_range_end=examples[-1].as_of if examples else None,
            repository_fingerprint=compute_fingerprint(sorted(match_ids)),
            label_distribution=label_distribution,
            warnings=warnings,
        )

        return examples, manifest
