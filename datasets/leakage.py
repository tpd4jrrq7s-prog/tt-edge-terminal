"""Explicit, structured leakage detection for datasets and splits.

Every check returns a `LeakageCheckResult`; `run_dataset_leakage_report`
combines the relevant checks into one `LeakageReport`. Tests
deliberately construct leaking inputs and assert these checks catch them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from datasets.errors import LeakageViolation
from datasets.models import DatasetSplit, SplitPlan, TrainingExample
from features.models import FeatureSnapshot
from persistence.models import HistoricalMatchRecord
from persistence.protocols import MatchRepository

FORBIDDEN_FEATURE_FIELD_NAMES = frozenset({"winner_id", "player_a_won", "result", "final_winner", "winner"})


class LeakageCheckResult(BaseModel):
    name: str
    passed: bool
    detail: str


class LeakageReport(BaseModel):
    checks: list[LeakageCheckResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[LeakageCheckResult]:
        return [c for c in self.checks if not c.passed]


def check_target_excluded_from_history(snapshot: FeatureSnapshot) -> LeakageCheckResult:
    all_ids = {
        *snapshot.provenance.player_a_source_match_ids,
        *snapshot.provenance.player_b_source_match_ids,
        *snapshot.provenance.head_to_head_source_match_ids,
    }
    passed = snapshot.target_match_id not in all_ids
    detail = "target match excluded from its own feature history" if passed else "target match found in its own feature history"
    return LeakageCheckResult(name="target_match_excluded", passed=passed, detail=detail)


def check_source_timestamps_before_cutoff(
    snapshot: FeatureSnapshot, match_repository: MatchRepository
) -> LeakageCheckResult:
    all_ids = [
        *snapshot.provenance.player_a_source_match_ids,
        *snapshot.provenance.player_b_source_match_ids,
        *snapshot.provenance.head_to_head_source_match_ids,
    ]
    bad = []
    for match_id in all_ids:
        record = match_repository.get(match_id)
        if record is not None and record.effective_timestamp >= snapshot.as_of:
            bad.append(match_id)
    passed = not bad
    detail = "all source records precede as_of" if passed else f"source records at/after as_of: {bad}"
    return LeakageCheckResult(name="source_timestamps_before_cutoff", passed=passed, detail=detail)


def check_odds_captured_before_cutoff(odds_ids_and_timestamps, as_of) -> LeakageCheckResult:  # type: ignore[no-untyped-def]
    bad = [oid for oid, captured_at in odds_ids_and_timestamps if captured_at > as_of]
    passed = not bad
    detail = "all odds captured at or before as_of" if passed else f"odds captured after as_of: {bad}"
    return LeakageCheckResult(name="odds_captured_before_cutoff", passed=passed, detail=detail)


def check_snapshot_matches_own_example(example: TrainingExample) -> LeakageCheckResult:
    passed = (
        example.features.target_match_id == example.target_match_id
        and example.features.as_of == example.as_of
    )
    detail = "snapshot matches its own training example" if passed else (
        "snapshot's target_match_id/as_of does not match the training example it is attached to "
        "(a future snapshot used for an earlier target, or vice versa)"
    )
    return LeakageCheckResult(name="snapshot_matches_own_example", passed=passed, detail=detail)


def check_no_forbidden_target_fields(snapshot: FeatureSnapshot) -> LeakageCheckResult:
    field_names = (
        set(snapshot.model_dump(exclude={"player_a_features", "player_b_features", "matchup_features"}).keys())
        | set(snapshot.player_a_features.model_dump().keys())
        | set(snapshot.player_b_features.model_dump().keys())
        | set(snapshot.matchup_features.model_dump().keys())
    )
    found = field_names & FORBIDDEN_FEATURE_FIELD_NAMES
    passed = not found
    detail = "no forbidden target fields present" if passed else f"forbidden fields present: {sorted(found)}"
    return LeakageCheckResult(name="no_forbidden_target_fields", passed=passed, detail=detail)


def check_no_duplicate_examples_across_splits(splits: list[DatasetSplit]) -> LeakageCheckResult:
    """Check a flat (holdout-style) list of splits for any example appearing twice.

    Not applicable to walk-forward/rolling-window `SplitFold`s, where
    training examples are expected to recur across expanding folds.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for split in splits:
        for example_id in split.example_ids:
            if example_id in seen:
                duplicates.add(example_id)
            seen.add(example_id)
    passed = not duplicates
    detail = "no example appears in more than one split" if passed else f"duplicated example ids: {sorted(duplicates)}"
    return LeakageCheckResult(name="no_duplicate_examples_across_splits", passed=passed, detail=detail)


def check_chronological_split_boundaries(splits: list[DatasetSplit]) -> LeakageCheckResult:
    previous_end = None
    for split in splits:
        if previous_end is not None and split.start is not None and split.start < previous_end:
            return LeakageCheckResult(
                name="chronological_split_boundaries",
                passed=False,
                detail=f"split {split.label!r} starts before the previous split ends",
            )
        if split.end is not None:
            previous_end = split.end
    return LeakageCheckResult(name="chronological_split_boundaries", passed=True, detail="splits are chronological")


def check_provider_mapping_consistency(matches: list[HistoricalMatchRecord]) -> LeakageCheckResult:
    mapping: dict[tuple[str, str], str] = {}
    conflicts: list[tuple[str, str]] = []
    for match in matches:
        key = (match.provider, match.provider_match_id)
        existing = mapping.get(key)
        if existing is not None and existing != match.id:
            conflicts.append(key)
        mapping[key] = match.id
    passed = not conflicts
    detail = "no conflicting provider mappings" if passed else f"conflicting provider mappings: {conflicts}"
    return LeakageCheckResult(name="provider_mapping_consistency", passed=passed, detail=detail)


def run_dataset_leakage_report(
    examples: list[TrainingExample],
    match_repository: MatchRepository,
    plan: SplitPlan | None = None,
) -> LeakageReport:
    """Run every applicable leakage check across a built dataset (and its split plan, if any)."""
    checks: list[LeakageCheckResult] = []
    for example in examples:
        checks.append(check_target_excluded_from_history(example.features))
        checks.append(check_source_timestamps_before_cutoff(example.features, match_repository))
        checks.append(check_snapshot_matches_own_example(example))
        checks.append(check_no_forbidden_target_fields(example.features))

    checks.append(
        check_provider_mapping_consistency([match_repository.get(e.target_match_id) for e in examples if match_repository.get(e.target_match_id)])
    )

    if plan is not None and plan.splits:
        checks.append(check_no_duplicate_examples_across_splits(plan.splits))
        checks.append(check_chronological_split_boundaries(plan.splits))

    return LeakageReport(checks=checks)


def assert_no_leakage(report: LeakageReport) -> None:
    """Raise `LeakageViolation` if any check in the report failed."""
    if not report.passed:
        failures = ", ".join(f"{c.name}: {c.detail}" for c in report.failures)
        raise LeakageViolation(f"Leakage detected: {failures}", report)
