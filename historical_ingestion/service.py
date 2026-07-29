"""HistoricalImportService: the single orchestrated entrypoint for Phase 4 ingestion.

## In-memory transaction semantics (exact, documented — not pretending ACID)

Each batch is processed into local staging lists first (accepted
players/matches/odds/rankings/competitions to insert or replace,
quarantine records, conflict records). Only *after* the entire batch's
records have been processed without an unexpected exception are those
staged writes applied to the repositories, and only after that does the
checkpoint advance. If an unexpected exception occurs mid-batch, the
staging lists for that batch are discarded entirely — nothing from that
batch is written, and the checkpoint stays exactly where it was. This is
plain Python list staging, not a database transaction: there is no
isolation from concurrent readers of the repositories, and repository
mutations that already reference not-yet-committed IDs within the same
batch are also part of the same all-or-nothing unit, since nothing is
called until staging is complete.

Dry-run mode runs every step (adapt, validate, resolve, deduplicate,
report) but skips both repository writes and the checkpoint update.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from config.historical_ingestion import HistoricalIngestionSettings, get_historical_ingestion_settings
from historical_ingestion.checkpoints import ImportCheckpoint
from historical_ingestion.deduplication import (
    DuplicateOutcome,
    compute_match_internal_id,
    compute_odds_internal_id,
    evaluate_match_duplicate,
    evaluate_odds_duplicate,
)
from historical_ingestion.errors import BatchProcessingError
from historical_ingestion.models import ImportedCompetition, ImportedMatch, ImportedOdds, ImportedPlayer, ImportedRanking
from historical_ingestion.pipeline import (
    adapt_record,
    build_historical_competition_record,
    build_historical_match_record,
    build_historical_odds_record,
    build_historical_player_record,
    build_historical_ranking_record,
    resolve_player_identity,
)
from historical_ingestion.protocols import CheckpointStore, HistoricalDataSource, HistoricalProviderAdapter
from historical_ingestion.reports import (
    BatchImportReport,
    ConflictRecord,
    ImportMetrics,
    ImportReport,
    QuarantineRecord,
    RecordImportResult,
    compute_run_id,
)
from historical_ingestion.validation import IssueSeverity, ValidationIssue, decide_validation_outcome, validate_match, validate_odds
from identity.models import ExternalIdentifier, IdentityOutcome, PlayerIdentityRecord
from identity.normalizer import normalize_player_name
from identity.resolver import IdentityResolver
from persistence.errors import DuplicateRecordError
from persistence.models import HistoricalMatchRecord
from persistence.protocols import (
    CompetitionRepository,
    MatchRepository,
    OddsRepository,
    PlayerRepository,
    RankingRepository,
)
from providers.errors import ProviderError


class _BatchStaging:
    """Everything accumulated while processing one batch, before it is (maybe) committed."""

    def __init__(self) -> None:
        self.players: list = []
        self.matches: list = []
        self.matches_to_replace: list = []
        self.odds: list = []
        self.rankings: list = []
        self.competitions: list = []
        self.new_identities: list[PlayerIdentityRecord] = []
        self.external_index_keys_added: list[tuple[str, str]] = []
        self.results: list[RecordImportResult] = []
        self.quarantine: list[QuarantineRecord] = []
        self.conflicts: list[ConflictRecord] = []
        self.issues: list[ValidationIssue] = []


class HistoricalImportService:
    """Reads, adapts, validates, resolves, deduplicates, and persists historical import batches."""

    def __init__(
        self,
        source: HistoricalDataSource,
        adapter: HistoricalProviderAdapter,
        identity_resolver: IdentityResolver,
        player_repository: PlayerRepository,
        match_repository: MatchRepository,
        odds_repository: OddsRepository,
        ranking_repository: RankingRepository,
        competition_repository: CompetitionRepository,
        checkpoint_store: CheckpointStore,
        settings: HistoricalIngestionSettings | None = None,
        clock: Callable[[], datetime] | None = None,
        known_identities: list[PlayerIdentityRecord] | None = None,
    ) -> None:
        self._source = source
        self._adapter = adapter
        self._resolver = identity_resolver
        self._players = player_repository
        self._matches = match_repository
        self._odds = odds_repository
        self._rankings = ranking_repository
        self._competitions = competition_repository
        self._checkpoints = checkpoint_store
        self._settings = settings or get_historical_ingestion_settings()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

        self._known_identities: list[PlayerIdentityRecord] = list(known_identities or [])
        self._external_index: dict[tuple[str, str], str] = {}
        for identity in self._known_identities:
            for ext in identity.external_ids:
                self._external_index[(ext.provider, ext.provider_player_id)] = identity.id

        self._quarantine_store: dict[str, QuarantineRecord] = {}
        self._conflict_store: dict[str, ConflictRecord] = {}

    # -- quarantine / conflict access -------------------------------------------------

    def list_quarantine_records(self) -> list[QuarantineRecord]:
        return list(self._quarantine_store.values())

    def get_quarantine_record(self, quarantine_id: str) -> QuarantineRecord | None:
        return self._quarantine_store.get(quarantine_id)

    def list_conflict_records(self) -> list[ConflictRecord]:
        return list(self._conflict_store.values())

    def get_conflict_record(self, conflict_id: str) -> ConflictRecord | None:
        return self._conflict_store.get(conflict_id)

    # -- repository fingerprint ---------------------------------------------------

    def _repository_fingerprint(self) -> str:
        return f"players={self._players.count()},matches={self._matches.count()}," \
            f"odds={self._odds.count()},rankings={self._rankings.count()},competitions={self._competitions.count()}"

    def _rollback_identity_mutations(self, staging: "_BatchStaging") -> None:
        """Undo the eager, within-batch identity/index mutations (fatal error or dry-run)."""
        for identity in staging.new_identities:
            if identity in self._known_identities:
                self._known_identities.remove(identity)
        for key in staging.external_index_keys_added:
            self._external_index.pop(key, None)

    # -- per-record-type handlers (append to `staging`) ---------------------------

    def _handle_player(self, imported: ImportedPlayer, staging: _BatchStaging) -> None:
        provider = imported.provenance.provider
        provider_record_id = imported.provenance.provider_record_id
        resolution = resolve_player_identity(imported, self._known_identities, self._resolver)

        if resolution.outcome is IdentityOutcome.MATCHED:
            internal_id = resolution.identity_id
            self._external_index[(provider, provider_record_id)] = internal_id
            staging.external_index_keys_added.append((provider, provider_record_id))
            existing = self._players.get(internal_id)
            outcome = "merged_safe" if existing is not None else "inserted"
            if existing is None:
                staging.players.append(build_historical_player_record(imported, internal_id))
            staging.results.append(
                RecordImportResult(record_type="player", provider_record_id=provider_record_id, internal_id=internal_id, outcome=outcome, reasons=resolution.reasons)
            )
            return

        if resolution.outcome is IdentityOutcome.CREATED:
            internal_id = f"player:{provider}:{provider_record_id}"
            new_identity = PlayerIdentityRecord(
                id=internal_id,
                canonical_name=imported.name,
                normalized_name=normalize_player_name(imported.name),
                external_ids=[ExternalIdentifier(provider=provider, provider_player_id=provider_record_id)],
                country=imported.country,
            )
            # Made visible immediately (not deferred to batch commit) so later records in the
            # *same* batch can resolve against it too — e.g. two new players in one batch whose
            # names are close enough to be mutually ambiguous. Rolled back on a fatal batch error.
            staging.new_identities.append(new_identity)
            self._known_identities.append(new_identity)
            self._external_index[(provider, provider_record_id)] = internal_id
            staging.external_index_keys_added.append((provider, provider_record_id))
            staging.players.append(build_historical_player_record(imported, internal_id))
            staging.results.append(
                RecordImportResult(record_type="player", provider_record_id=provider_record_id, internal_id=internal_id, outcome="inserted", reasons=resolution.reasons)
            )
            return

        # AMBIGUOUS or REJECTED: never auto-merge, never silently persist.
        reason = "ambiguous_identity"
        if self._settings.identity_ambiguity_policy == "reject" and resolution.outcome is IdentityOutcome.AMBIGUOUS:
            staging.results.append(
                RecordImportResult(record_type="player", provider_record_id=provider_record_id, internal_id=None, outcome="rejected_invalid", reasons=resolution.reasons)
            )
            return

        if self._settings.quarantine_enabled:
            qid = f"quarantine:player:{provider}:{provider_record_id}"
            staging.quarantine.append(
                QuarantineRecord(
                    id=qid, record_type="player", provider=provider, provider_record_id=provider_record_id,
                    raw_fingerprint=imported.provenance.raw_fingerprint, reason=reason,
                    issues=resolution.reasons,
                    candidate_identity_matches=[c.identity_id for c in resolution.candidates],
                    created_at=self._clock(),
                )
            )
            staging.results.append(
                RecordImportResult(record_type="player", provider_record_id=provider_record_id, internal_id=None, outcome="quarantined", reasons=resolution.reasons)
            )
        else:
            staging.results.append(
                RecordImportResult(record_type="player", provider_record_id=provider_record_id, internal_id=None, outcome="rejected_invalid", reasons=resolution.reasons)
            )

    def _handle_match(self, imported: ImportedMatch, staging: _BatchStaging, now: datetime) -> None:
        provider = imported.provenance.provider
        provider_record_id = imported.provenance.provider_record_id

        issues = validate_match(imported, self._settings, now)
        staging.issues.extend(issues)
        outcome_policy = decide_validation_outcome(issues, self._settings)

        if outcome_policy == "reject":
            self._reject_or_quarantine("match", provider, provider_record_id, imported.provenance.raw_fingerprint, issues, staging)
            return

        player_a_id = self._external_index.get((provider, imported.player_a_external_id))
        player_b_id = self._external_index.get((provider, imported.player_b_external_id))
        if player_a_id is None or player_b_id is None:
            self._quarantine_missing_field("match", provider, provider_record_id, imported.provenance.raw_fingerprint, "player identity not yet resolved", staging)
            return

        internal_id = compute_match_internal_id(provider, provider_record_id)
        candidate = build_historical_match_record(imported, internal_id, player_a_id, player_b_id)
        decision = evaluate_match_duplicate(
            self._matches, provider, provider_record_id, player_a_id, player_b_id,
            candidate.scheduled_at, candidate.status, candidate.winner_id, candidate.sets, self._settings,
        )
        self._apply_match_decision(candidate, decision, staging, provider, provider_record_id)

    def _apply_match_decision(
        self, candidate: HistoricalMatchRecord, decision, staging: _BatchStaging, provider: str, provider_record_id: str
    ) -> None:
        if decision.outcome is DuplicateOutcome.INSERTED:
            staging.matches.append(candidate)
            staging.results.append(RecordImportResult(record_type="match", provider_record_id=provider_record_id, internal_id=decision.internal_match_id, outcome="inserted", reasons=decision.reasons))
        elif decision.outcome is DuplicateOutcome.MERGED_SAFE:
            staging.matches_to_replace.append(candidate)
            staging.results.append(RecordImportResult(record_type="match", provider_record_id=provider_record_id, internal_id=decision.internal_match_id, outcome="merged_safe", reasons=decision.reasons))
        elif decision.outcome is DuplicateOutcome.SKIPPED_IDEMPOTENT:
            staging.results.append(RecordImportResult(record_type="match", provider_record_id=provider_record_id, internal_id=decision.internal_match_id, outcome="skipped_idempotent", reasons=decision.reasons))
        elif decision.outcome is DuplicateOutcome.QUARANTINED:
            if self._settings.quarantine_enabled:
                staging.quarantine.append(
                    QuarantineRecord(
                        id=f"quarantine:match:{provider}:{provider_record_id}", record_type="match", provider=provider,
                        provider_record_id=provider_record_id, raw_fingerprint="", reason="duplicate_conflict",
                        issues=decision.reasons, conflict_detail=decision.conflicting_existing_match_id, created_at=self._clock(),
                    )
                )
            staging.results.append(RecordImportResult(record_type="match", provider_record_id=provider_record_id, internal_id=decision.internal_match_id, outcome="quarantined", reasons=decision.reasons))
        else:  # REJECTED_CONFLICT
            staging.conflicts.append(
                ConflictRecord(
                    id=f"conflict:match:{provider}:{provider_record_id}", record_type="match", internal_id=decision.internal_match_id,
                    provider=provider, provider_record_id=provider_record_id, conflicting_field=decision.reasons[0] if decision.reasons else "unknown",
                    incoming_summary=f"winner={candidate.winner_id}", existing_summary="see repository", created_at=self._clock(),
                )
            )
            staging.results.append(RecordImportResult(record_type="match", provider_record_id=provider_record_id, internal_id=decision.internal_match_id, outcome="rejected_conflict", reasons=decision.reasons))

    def _handle_odds(self, imported: ImportedOdds, staging: _BatchStaging, now: datetime) -> None:
        provider = imported.provenance.provider
        provider_record_id = imported.provenance.provider_record_id
        issues = validate_odds(imported, None, self._settings, now)
        staging.issues.extend(issues)
        outcome_policy = decide_validation_outcome(issues, self._settings)
        if outcome_policy == "reject":
            self._reject_or_quarantine("odds", provider, provider_record_id, imported.provenance.raw_fingerprint, issues, staging)
            return

        internal_match_id = compute_match_internal_id(provider, imported.provider_match_id)
        internal_selection_id = self._external_index.get((provider, imported.selection_external_id))
        if internal_selection_id is None:
            self._quarantine_missing_field("odds", provider, provider_record_id, imported.provenance.raw_fingerprint, "selection player identity not yet resolved", staging)
            return

        decision = evaluate_odds_duplicate(
            self._odds, provider, internal_match_id, imported.bookmaker, internal_selection_id, imported.decimal_odds, imported.captured_at,
        )
        if decision.outcome is DuplicateOutcome.INSERTED:
            record = build_historical_odds_record(imported, decision.internal_odds_id, internal_match_id, internal_selection_id)
            staging.odds.append(record)
            staging.results.append(RecordImportResult(record_type="odds", provider_record_id=provider_record_id, internal_id=decision.internal_odds_id, outcome="inserted", reasons=decision.reasons))
        elif decision.outcome is DuplicateOutcome.SKIPPED_IDEMPOTENT:
            staging.results.append(RecordImportResult(record_type="odds", provider_record_id=provider_record_id, internal_id=decision.internal_odds_id, outcome="skipped_idempotent", reasons=decision.reasons))
        else:  # REJECTED_CONFLICT
            staging.conflicts.append(
                ConflictRecord(
                    id=f"conflict:odds:{provider}:{provider_record_id}", record_type="odds", internal_id=decision.internal_odds_id,
                    provider=provider, provider_record_id=provider_record_id, conflicting_field="decimal_odds",
                    incoming_summary=f"decimal_odds={imported.decimal_odds}", existing_summary="see repository", created_at=self._clock(),
                )
            )
            staging.results.append(RecordImportResult(record_type="odds", provider_record_id=provider_record_id, internal_id=decision.internal_odds_id, outcome="rejected_conflict", reasons=decision.reasons))

    def _handle_ranking(self, imported: ImportedRanking, staging: _BatchStaging) -> None:
        provider = imported.provenance.provider
        provider_record_id = imported.provenance.provider_record_id
        if imported.player_external_id is None or imported.ranking is None or imported.effective_at is None:
            self._quarantine_missing_field("ranking", provider, provider_record_id, imported.provenance.raw_fingerprint, "missing player/ranking/effective_at", staging)
            return
        internal_player_id = self._external_index.get((provider, imported.player_external_id))
        if internal_player_id is None:
            self._quarantine_missing_field("ranking", provider, provider_record_id, imported.provenance.raw_fingerprint, "player identity not yet resolved", staging)
            return

        internal_id = f"{provider}:{provider_record_id}"
        if self._rankings.get(internal_id) is not None:
            staging.results.append(RecordImportResult(record_type="ranking", provider_record_id=provider_record_id, internal_id=internal_id, outcome="skipped_idempotent", reasons=["identical re-import"]))
            return
        record = build_historical_ranking_record(imported, internal_id, internal_player_id)
        staging.rankings.append(record)
        staging.results.append(RecordImportResult(record_type="ranking", provider_record_id=provider_record_id, internal_id=internal_id, outcome="inserted", reasons=[]))

    def _handle_competition(self, imported: ImportedCompetition, staging: _BatchStaging) -> None:
        provider = imported.provenance.provider
        provider_record_id = imported.provenance.provider_record_id
        internal_id = f"{provider}:{provider_record_id}"
        if self._competitions.get(internal_id) is not None:
            staging.results.append(RecordImportResult(record_type="competition", provider_record_id=provider_record_id, internal_id=internal_id, outcome="skipped_idempotent", reasons=["identical re-import"]))
            return
        record = build_historical_competition_record(imported, internal_id)
        staging.competitions.append(record)
        staging.results.append(RecordImportResult(record_type="competition", provider_record_id=provider_record_id, internal_id=internal_id, outcome="inserted", reasons=[]))

    def _reject_or_quarantine(
        self, record_type: str, provider: str, provider_record_id: str, raw_fingerprint: str,
        issues: list[ValidationIssue], staging: _BatchStaging,
    ) -> None:
        reason = _reason_from_issues(issues)
        if self._settings.quarantine_enabled:
            staging.quarantine.append(
                QuarantineRecord(
                    id=f"quarantine:{record_type}:{provider}:{provider_record_id}", record_type=record_type, provider=provider,
                    provider_record_id=provider_record_id, raw_fingerprint=raw_fingerprint, reason=reason,
                    issues=[i.message for i in issues], created_at=self._clock(),
                )
            )
            staging.results.append(RecordImportResult(record_type=record_type, provider_record_id=provider_record_id, internal_id=None, outcome="quarantined", reasons=[i.message for i in issues]))
        else:
            staging.results.append(RecordImportResult(record_type=record_type, provider_record_id=provider_record_id, internal_id=None, outcome="rejected_invalid", reasons=[i.message for i in issues]))

    def _quarantine_missing_field(
        self, record_type: str, provider: str, provider_record_id: str, raw_fingerprint: str, detail: str, staging: _BatchStaging
    ) -> None:
        if self._settings.quarantine_enabled:
            staging.quarantine.append(
                QuarantineRecord(
                    id=f"quarantine:{record_type}:{provider}:{provider_record_id}", record_type=record_type, provider=provider,
                    provider_record_id=provider_record_id, raw_fingerprint=raw_fingerprint, reason="missing_critical_field",
                    issues=[detail], created_at=self._clock(),
                )
            )
            staging.results.append(RecordImportResult(record_type=record_type, provider_record_id=provider_record_id, internal_id=None, outcome="quarantined", reasons=[detail]))
        else:
            staging.results.append(RecordImportResult(record_type=record_type, provider_record_id=provider_record_id, internal_id=None, outcome="rejected_invalid", reasons=[detail]))

    # -- batch processing -----------------------------------------------------

    def process_batch(self, cursor: str | None, dry_run: bool | None = None) -> tuple[str | None, BatchImportReport]:
        """Fetch and process exactly one batch. Returns (next_cursor, report)."""
        dry_run = self._settings.dry_run_default if dry_run is None else dry_run
        started = self._clock()
        batch = self._source.fetch_batch(cursor)
        now = self._clock()
        staging = _BatchStaging()

        try:
            for raw in batch.records:
                try:
                    imported = adapt_record(raw, self._adapter)
                except ProviderError as exc:
                    if self._settings.quarantine_enabled:
                        staging.quarantine.append(
                            QuarantineRecord(
                                id=f"quarantine:{raw.record_type}:{raw.provider}:{raw.provider_record_id}",
                                record_type=raw.record_type, provider=raw.provider, provider_record_id=raw.provider_record_id,
                                raw_fingerprint="", reason="provider_mapping_failure", issues=[str(exc)], created_at=self._clock(),
                            )
                        )
                    staging.results.append(RecordImportResult(record_type=raw.record_type, provider_record_id=raw.provider_record_id, outcome="quarantined" if self._settings.quarantine_enabled else "rejected_invalid", reasons=[str(exc)]))
                    continue

                if isinstance(imported, ImportedPlayer):
                    self._handle_player(imported, staging)
                elif isinstance(imported, ImportedMatch):
                    self._handle_match(imported, staging, now)
                elif isinstance(imported, ImportedOdds):
                    self._handle_odds(imported, staging, now)
                elif isinstance(imported, ImportedRanking):
                    self._handle_ranking(imported, staging)
                elif isinstance(imported, ImportedCompetition):
                    self._handle_competition(imported, staging)
        except Exception as exc:  # noqa: BLE001 - deliberate: any unexpected error aborts the whole batch
            self._rollback_identity_mutations(staging)
            elapsed = (self._clock() - started).total_seconds()
            report = BatchImportReport(
                batch_id=batch.batch_id, records_read=len(batch.records), records_accepted=0, records_inserted=0,
                records_skipped=0, records_quarantined=0, records_rejected=len(batch.records),
                validation_issue_counts_by_severity={}, metrics=ImportMetrics(sample_size=len(batch.records)),
                elapsed_seconds=elapsed, checkpoint_cursor_before=cursor, checkpoint_cursor_after=cursor,
                succeeded=False, failure_reason=f"{type(exc).__name__}: {exc}",
            )
            return cursor, report  # checkpoint unchanged; nothing persisted (including identity state)

        if dry_run:
            # Dry-run must leave zero persistent side effects, including identity/index state
            # that was made eagerly visible during this batch's processing.
            self._rollback_identity_mutations(staging)

        if not dry_run:
            for player in staging.players:
                try:
                    self._players.add(player)
                except DuplicateRecordError:
                    pass
            for match in staging.matches:
                self._matches.add(match)
            for match in staging.matches_to_replace:
                self._matches.replace(match)
            for odds in staging.odds:
                self._odds.add(odds)
            for ranking in staging.rankings:
                self._rankings.add(ranking)
            for competition in staging.competitions:
                self._competitions.add(competition)
            for quarantine in staging.quarantine:
                self._quarantine_store[quarantine.id] = quarantine
            for conflict in staging.conflicts:
                self._conflict_store[conflict.id] = conflict

        outcomes = [r.outcome for r in staging.results]
        accepted = sum(1 for o in outcomes if o in ("inserted", "merged_safe", "skipped_idempotent"))
        inserted = sum(1 for o in outcomes if o in ("inserted", "merged_safe"))
        skipped = sum(1 for o in outcomes if o == "skipped_idempotent")
        quarantined = sum(1 for o in outcomes if o == "quarantined")
        rejected = sum(1 for o in outcomes if o in ("rejected_conflict", "rejected_invalid"))

        severity_counts: dict[str, int] = {}
        for issue in staging.issues:
            severity_counts[issue.severity.value] = severity_counts.get(issue.severity.value, 0) + 1

        metrics = _compute_metrics(staging, len(batch.records))
        elapsed = (self._clock() - started).total_seconds()

        checkpoint_after = batch.next_cursor
        if not dry_run:
            self._checkpoints.save(
                ImportCheckpoint(
                    source_name=self._source.name, provider=self._source.provider, cursor=batch.next_cursor,
                    last_successful_batch_id=batch.batch_id, last_successful_source_timestamp=batch.source_timestamp,
                    processed_record_count=len(batch.records), repository_fingerprint=self._repository_fingerprint(),
                    checkpoint_version=self._settings.checkpoint_version, updated_at=self._clock(),
                )
            )

        report = BatchImportReport(
            batch_id=batch.batch_id, records_read=len(batch.records), records_accepted=accepted,
            records_inserted=inserted, records_skipped=skipped, records_quarantined=quarantined, records_rejected=rejected,
            validation_issue_counts_by_severity=severity_counts, metrics=metrics, elapsed_seconds=elapsed,
            checkpoint_cursor_before=cursor, checkpoint_cursor_after=checkpoint_after, succeeded=True,
        )
        return checkpoint_after, report

    def run(self, dry_run: bool | None = None, max_batches: int | None = None) -> ImportReport:
        """Process batches from the last checkpoint until the source is exhausted or `max_batches` is hit."""
        dry_run = self._settings.dry_run_default if dry_run is None else dry_run
        started_at = self._clock()
        run_id = compute_run_id(self._source.name, self._source.provider, started_at)
        fingerprint_before = self._repository_fingerprint()

        checkpoint = self._checkpoints.get(self._source.name, self._source.provider)
        cursor = checkpoint.cursor if checkpoint else None

        batches: list[BatchImportReport] = []
        batch_count = 0
        while True:
            if max_batches is not None and batch_count >= max_batches:
                break
            next_cursor, batch_report = self.process_batch(cursor, dry_run=dry_run)
            batches.append(batch_report)
            batch_count += 1
            if not batch_report.succeeded:
                break
            cursor = next_cursor
            if cursor is None:
                break

        finished_at = self._clock()
        totals = {
            "read": sum(b.records_read for b in batches), "accepted": sum(b.records_accepted for b in batches),
            "inserted": sum(b.records_inserted for b in batches), "skipped": sum(b.records_skipped for b in batches),
            "quarantined": sum(b.records_quarantined for b in batches), "rejected": sum(b.records_rejected for b in batches),
        }
        cumulative = _combine_metrics([b.metrics for b in batches])

        return ImportReport(
            run_id=run_id, source_name=self._source.name, provider=self._source.provider,
            started_at=started_at, finished_at=finished_at, dry_run=dry_run, batches=batches,
            total_records_read=totals["read"], total_accepted=totals["accepted"], total_inserted=totals["inserted"],
            total_skipped=totals["skipped"], total_quarantined=totals["quarantined"], total_rejected=totals["rejected"],
            cumulative_metrics=cumulative, repository_fingerprint_before=fingerprint_before,
            repository_fingerprint_after=self._repository_fingerprint(),
        )


def _reason_from_issues(issues: list[ValidationIssue]):
    codes = {i.code for i in issues}
    if any(c in {"negative_set_score"} for c in codes):
        return "invalid_score"
    if any(c in {"invalid_status"} for c in codes):
        return "unsupported_status"
    if any(c in {"completion_before_start", "future_historical_record", "future_odds_timestamp"} for c in codes):
        return "impossible_timestamps"
    if any(c in {"score_result_mismatch"} for c in codes):
        return "conflicting_result"
    return "missing_critical_field"


def _compute_metrics(staging: _BatchStaging, total_read: int) -> ImportMetrics:
    total = max(total_read, 1)
    fatal_or_error = sum(1 for i in staging.issues if i.severity in (IssueSeverity.FATAL, IssueSeverity.ERROR))
    accepted = sum(1 for r in staging.results if r.outcome in ("inserted", "merged_safe", "skipped_idempotent"))
    quarantined = sum(1 for r in staging.results if r.outcome == "quarantined")
    rejected_conflicts = sum(1 for r in staging.results if r.outcome == "rejected_conflict")
    idempotent = sum(1 for r in staging.results if r.outcome == "skipped_idempotent")

    return ImportMetrics(
        sample_size=total_read,
        structural_validity_rate=1.0 - (fatal_or_error / total),
        semantic_validity_rate=1.0 - (fatal_or_error / total),
        identity_resolution_rate=None,
        ambiguity_rate=(quarantined / total) if total_read else None,
        exact_duplicate_rate=(idempotent / total) if total_read else None,
        conflict_rate=(rejected_conflicts / total) if total_read else None,
        accepted_record_rate=(accepted / total) if total_read else None,
        timestamp_completeness=None,
        score_completeness=None,
        ranking_coverage=None,
        odds_coverage=None,
    )


def _combine_metrics(all_metrics: list[ImportMetrics]) -> ImportMetrics:
    total = sum(m.sample_size for m in all_metrics)
    if total == 0:
        return ImportMetrics(sample_size=0)

    def _weighted(field: str) -> float | None:
        parts = [(getattr(m, field), m.sample_size) for m in all_metrics if getattr(m, field) is not None]
        weight = sum(w for _, w in parts)
        if weight == 0:
            return None
        return sum(v * w for v, w in parts) / weight

    return ImportMetrics(
        sample_size=total,
        structural_validity_rate=_weighted("structural_validity_rate"),
        semantic_validity_rate=_weighted("semantic_validity_rate"),
        identity_resolution_rate=_weighted("identity_resolution_rate"),
        ambiguity_rate=_weighted("ambiguity_rate"),
        exact_duplicate_rate=_weighted("exact_duplicate_rate"),
        conflict_rate=_weighted("conflict_rate"),
        accepted_record_rate=_weighted("accepted_record_rate"),
        timestamp_completeness=_weighted("timestamp_completeness"),
        score_completeness=_weighted("score_completeness"),
        ranking_coverage=_weighted("ranking_coverage"),
        odds_coverage=_weighted("odds_coverage"),
    )
