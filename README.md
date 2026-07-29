# TT Edge Terminal

A modular, cloud-ready analytics and decision-support platform for table
tennis: live scores, live odds movements, historical performance,
momentum, value opportunities, and risk metrics.

**This is an analytics and decision-support platform only.** It does not
place bets, does not automate any betting action, and never will — no
component in this codebase has write access to any bookmaker or betting
system.

## Status: Phase 4 — Real Historical Data Platform

Phase 1 established the project skeleton. Phase 2A added
provider-independent ingestion and normalization. Phase 2B added a
transparent, deterministic, rules-based analytics engine. Phase 3 added
a leakage-safe historical data and feature-engineering foundation.
**Phase 4 adds a production-oriented, provider-agnostic historical data
acquisition and import layer**: sources, provider adapters, staged
validation, identity resolution, deduplication/conflict handling,
checkpointed resumption, quarantine, and quality metrics, feeding
straight into the existing Phase 3 repositories and feature builders.

- Typed configuration via Pydantic Settings
- Structured (JSON) logging
- Core domain models: `Player`, `Match`, `Odds`
- Provider-independent ingestion (`ingestion/`), currently backed by an
  **in-memory mock data source only** — see "Ingestion source" below
- Normalization (`normalization/`) that validates and converts raw
  provider data into domain models, raising clear errors on bad input
- A deterministic analytics engine (`engine/`): form, momentum,
  probability, confidence, risk, value, pattern detection, data quality,
  and human-readable explanations, orchestrated by
  `engine.orchestrator.MatchAnalyticsEngine`
- Temporal historical persistence (`persistence/`): timezone-aware
  match/player/odds/ranking/competition records with strict-before-cutoff
  repository queries
- Deterministic player identity resolution (`identity/`), never
  auto-merging ambiguous identities
- Leakage-safe rolling player and matchup features (`features/`), built
  only from records strictly before a prediction cutoff, now including
  ranking differentials where historical ranking data exists
- Chronological training datasets, time-based splits, explicit leakage
  detection, and deterministic export (`datasets/`)
- A provider-agnostic historical import platform (`historical_ingestion/`,
  `providers/`): deterministic mock/JSONL/CSV sources, a configurable
  generic provider adapter, staged structural/semantic/temporal
  validation, identity resolution integration, deduplication and
  conflict quarantine, checkpointed resumption, and typed import reports
- A comprehensive `tests/` suite (432 tests) covering domain models,
  ingestion, normalization, every engine module, every Phase 3 package,
  and every Phase 4 package

**Important:** the probability/risk/value model in `engine/` is
**deterministic and rules-based, not trained machine learning.** Neither
Phase 3 nor Phase 4 trains a model — Phase 4 builds the real-data
acquisition/import layer that feeds Phase 3's leakage-safe features and
datasets, which a future ML phase will train and validate against.
Every number produced anywhere in this codebase traces back to an
explicit, documented formula over data actually supplied — nothing is
fabricated when data is missing.

Still no database, no web scraping, no live betting providers, no
FastAPI/HTTP API, and no automated betting — this remains a read-only,
analytics-focused platform. **No production live provider is included
yet** — only a deterministic mock provider and local JSONL/CSV file
import.

## Project Structure

```
app/
  main.py               Entrypoint (python -m app.main)
  bootstrap.py          Startup wiring: loads settings, configures logging
config/
  settings.py           Typed application settings (Pydantic Settings)
  logging.py            Structured JSON logging setup
  analytics.py          Typed, validated analytics engine settings (weights/thresholds)
domain/
  player.py             Player entity
  match.py              Match entity (+ MatchStatus, SetScore)
  odds.py               Odds entity
ingestion/
  models.py             Typed, permissive raw provider models (RawPlayer, RawMatch, RawOdds)
  protocols.py          MatchSource protocol — the provider-independent source interface
  service.py            IngestionService: fetches + normalizes into domain models
  scheduler.py          IngestionScheduler: configurable, explicit run-once polling
  sources/
    mock_source.py      MockTableTennisSource: deterministic, in-memory mock data
normalization/
  match_normalizer.py   Converts RawMatch -> domain Match (validates players, status, schedule)
  odds_normalizer.py    Converts RawOdds -> domain Odds (validates odds, timestamp)
engine/
  models.py             Every typed input/output contract (requests, results, enums)
  form.py               Recent-form scoring (0-100) from historical matches
  momentum.py           Pre-match / in-play / no-data momentum scoring (0-100 per player)
  probability.py        Weighted-logistic win probability model
  confidence.py         Confidence scoring, independent of probability
  risk.py               Risk scoring (0-100) and low/medium/high/extreme classification
  value.py              Fair odds, implied probability, expected value, gated decisions
  patterns.py           Transparent behavioral pattern detection (slow starter, comeback, ...)
  quality.py            Data-quality scoring (0-100) with named, traceable issues
  explanations.py       Deterministic, non-LLM human-readable explanations
  orchestrator.py       MatchAnalyticsEngine: the single orchestrated entrypoint
  demo.py               Read-only demonstration entrypoint (python -m engine.demo)
config/
  historical.py         Typed, validated Phase 3 settings (windows, thresholds, splits, versions)
  historical_ingestion.py  Typed, validated Phase 4 settings (batch size, policies, versions)
persistence/
  models.py             Timezone-aware Historical{Player,Match,Odds,Ranking,Competition}Record
  protocols.py          Player/Match/Odds/FeatureSnapshot/Ranking/CompetitionRepository protocols
  in_memory.py          Deterministic in-memory repository implementations (no database)
  errors.py             DuplicateRecordError, ProviderMappingConflictError, RecordNotFoundError
identity/
  models.py             NormalizedPlayerIdentity, PlayerIdentityRecord, IdentityCandidate/Resolution
  normalizer.py         Deterministic name normalization (Unicode fold, case, punctuation)
  resolver.py           IdentityResolver: transparent, dependency-free similarity scoring
  errors.py             InvalidPlayerNameError
features/
  models.py             PlayerRollingFeatures, MatchupFeatures, FeatureSnapshot, ProvenanceMetadata
  rolling.py            Shared per-player-perspective rolling-window primitives
  player.py             Builds PlayerRollingFeatures (form, sets, points, streaks, rest, volatility, ...)
  matchup.py            Builds MatchupFeatures (head-to-head, no-data-safe)
  snapshots.py          Assembles a leakage-safe FeatureSnapshot + provenance fingerprint
  builder.py            HistoricalFeatureBuilder: the orchestrated, repository-backed entrypoint
  engine_adapter.py      Converts a FeatureSnapshot into Phase 2B engine inputs, where clean
  errors.py             TargetMatchNotFoundError, SnapshotLeakageError
  demo.py               Read-only demonstration entrypoint (python -m features.demo)
datasets/
  models.py             TrainingExample, DatasetManifest, DatasetSplit/SplitFold/SplitPlan
  builder.py            DatasetBuilder: completed matches -> labeled TrainingExamples
  splits.py             Chronological holdout, walk-forward, and rolling-window splits
  leakage.py            LeakageCheckResult/LeakageReport + explicit leakage checks
  export.py             Deterministic JSONL/CSV export + manifest export (no pickle)
  errors.py             DatasetError, LeakageViolation, InsufficientDataForSplitError
historical_ingestion/
  models.py             RawRecord, SourceBatch, SourceHealth, ImportProvenance, Imported* canonical models
  protocols.py          HistoricalDataSource, HistoricalProviderAdapter, CheckpointStore
  validation.py         Staged structural/semantic/temporal ValidationIssue checks + policy
  deduplication.py      Deterministic duplicate/conflict classification (match + odds)
  checkpoints.py        ImportCheckpoint + InMemoryCheckpointStore (no database)
  reports.py            ImportReport/BatchImportReport/ImportMetrics/QuarantineRecord/ConflictRecord
  pipeline.py            Pure per-record adapt/resolve/convert functions
  service.py            HistoricalImportService: the orchestrated entrypoint (staging + transactions)
  errors.py             HistoricalIngestionError, SourceReadError, CheckpointVersionError, ...
  demo.py               Read-only demonstration entrypoint (python -m historical_ingestion.demo)
  sources/
    base.py             Shared FileHistoricalDataSource (cursor-as-line-offset resumption)
    jsonl_source.py      JSONLFileSource: UTF-8 JSON Lines, one record per line
    csv_source.py        CSVFileSource: UTF-8 CSV, configurable delimiter
    mock_provider.py     MockTableTennisProviderSource: deterministic illustrative provider data
providers/
  models.py             ProviderMappingConfig: declarative, per-provider field mappings
  registry.py           ProviderRegistry: explicit, non-global provider-name -> adapter lookup
  errors.py             ProviderError, UnknownProviderStatusError, ProviderMappingError
  generic/
    adapter.py           GenericProviderAdapter: config-driven HistoricalProviderAdapter
    mappings.py          mock_provider_mapping(): the reference ProviderMappingConfig
tests/
  domain/               Tests for Player, Match, Odds
  app/                  Startup smoke test
  ingestion/            Tests for raw models, mock source, service, scheduler
  normalization/        Tests for match/odds normalization, including invalid input
  engine/               Tests for every engine module, config validation, and the orchestrator
  persistence/          Repository cutoff semantics, duplicates, defensive copies, config validation
  identity/             Normalization, exact/fuzzy/alias matching, ambiguity, determinism
  features/             Rolling features, matchup features, snapshots, builder, engine adapter
  datasets/             Dataset building, splits, leakage detection, export
  historical_ingestion/ Sources, validation, deduplication, checkpoints, reports, service, integration
  providers/            Adapter field mapping, status mapping, registry
```

## Analytics Data Flow

```
MatchAnalysisRequest                          # engine/models.py
  (match, odds, player histories, points,
   head-to-head, competition context —
   only `match` is required)
          ▼
MatchAnalyticsEngine.analyze()                # engine/orchestrator.py
  1. assess_data_quality()                    # engine/quality.py
  2. calculate_form() x2                      # engine/form.py
  3. calculate_momentum()                     # engine/momentum.py
  4. build_match_features()                   # engine/orchestrator.py
  5. calculate_probability()                  # engine/probability.py
  6. calculate_confidence()                   # engine/confidence.py
  7. calculate_risk()                         # engine/risk.py
  8. assess_value()                           # engine/value.py
  9. detect_patterns() x2                     # engine/patterns.py
 10. build_explanations()                     # engine/explanations.py
          ▼
MatchAnalysis                                 # single typed result
```

## Probability vs. Confidence

These answer different questions and must not be conflated:

- **Probability** ("how likely is player A to win?") is a point estimate
  in (0, 1) from `engine/probability.py`: a weighted combination of
  standardized feature differentials (form, ranking, momentum,
  head-to-head, match state, context) passed through a logistic function,
  then shrunk toward 0.5 in proportion to how poor the data quality is.
- **Confidence** ("how much should you trust that estimate?") is a
  separate 0-1 score from `engine/confidence.py`, built from sample size,
  data completeness, agreement between the probability engine's factors,
  calibration readiness, and momentum sample size. **A 70% probability
  with only one historical match behind it carries low confidence; the
  same 70% backed by dozens of matches and agreeing signals carries high
  confidence.**

## Risk vs. Value

- **Risk** (`engine/risk.py`) is a 0-100 score classified as
  `low`/`medium`/`high`/`extreme`, combining eight weighted components:
  data quality, conflicting signals, volatility, momentum reversal,
  bookmaker disagreement, market movement, missing data, and short match
  format. It is never presented as certainty — every risk explanation
  says so explicitly.
- **Value** (`engine/value.py`) compares the model's probability against
  bookmaker decimal odds. A positive edge alone is **not** labeled
  actionable: the edge, expected value, confidence, and risk must all
  clear configurable thresholds before a decision above `no_signal` is
  returned. Decisions use neutral wording (`no_signal`, `observe`,
  `possible_value`, `strong_value`) — never a stake size, never automation.

## Mathematical Formulas

```
fair_odds            = 1 / model_probability
implied_probability  = 1 / decimal_odds
probability_edge     = model_probability - implied_probability
expected_value       = model_probability * decimal_odds - 1

logistic combination:
  logit = Σ (weight_i * standardized_signal_i)   for i in {form, ranking, momentum,
                                                            head_to_head, match_state, context}
  shrunk_logit = logit * (data_quality_score / 100)
  player_one_probability = sigmoid(shrunk_logit), clamped to (1e-4, 1 - 1e-4)
  player_two_probability = 1 - player_one_probability

recency weight (form/momentum):
  weight = 0.5 ^ (days_since_match / half_life_days)
```

All weights (`ProbabilityWeights`, `RiskWeights`) are validated to sum to
1.0 and to be non-negative; all thresholds are validated to be correctly
ordered — see `config/analytics.py`.

## Example Analysis Output

Abbreviated output of `python -m engine.demo` (mock data, illustrative
demo history — see the "Current Limitations" section):

```
Timo Boll vs Dimitrij Ovtcharov
  Form:      66.0 vs 47.1
  Momentum:  62.0 vs 38.0 (in_play)

Win probability: Timo Boll 54.8% vs Dimitrij Ovtcharov 45.2%

Confidence: high (0.78)
Risk:       low (6.3/100)

Value (timo_boll): fair odds 1.83, market 2.10, edge +7.1pp, EV +0.150 -> strong_value
Value (dimitrij_ovtcharov): fair odds 2.21, market 1.75, edge -11.9pp, EV -0.208 -> no_signal

Detected patterns:
  - timo_boll: high_volatility (strength=0.94, confidence=1.00, n=3)
  - ...

This is analytical decision support based on deterministic, rules-based
calculations over mock data — not a certainty, and not a wager suggestion.
```

## Running the Analytics Demo

```bash
pip install -r requirements.txt
python -m engine.demo
```

This loads deterministic mock ingestion data, normalizes it, adds a
small explicitly-labeled set of demo-only historical/point data, runs
`MatchAnalyticsEngine`, and prints the full analysis. It makes no network
calls and never suggests or places a wager.

## Current Limitations

- The probability/risk/value model is **deterministic and rules-based**,
  not a trained statistical or machine-learning model.
- The only ingestion source is `MockTableTennisSource` — **all data used
  by the engine and its demo is mock data**, not live results or odds.
- Surface/competition context is accepted as input but not yet modeled
  numerically (its probability weight defaults to 0 — wired for future
  use, not fabricated today).
- Historical recency is computed relative to analysis time, not the
  match's own scheduled time, since the domain `Match` model has no
  timestamp field.
- Pattern detection from set-level history (e.g. `late_set_collapse`) is
  a proxy based on final set scores, not true point-by-point historical
  replay — only the *current* match's `point_progression` carries true
  point-by-point detail.
- Output is analytical decision support, not certainty, and this project
  **does not place bets or automate wagering in any form.**
- **Historical demo data in `features/demo.py` and
  `historical_ingestion/demo.py` is illustrative/mock** — hand-authored
  or deterministically generated fixed records, not real match results.
- **No trained ML model exists yet.** Phases 3 and 4 build the
  leakage-safe feature/dataset foundation and the data acquisition/import
  layer respectively; `datasets/` produces labeled examples and
  chronological splits, not predictions.
- `opponent_adjusted_win_rate` is always `None` — it would require a
  recursive notion of "opponent strength" this phase deliberately does
  not model. `ranking_differential` **is now populated** when historical
  ranking data exists (via `persistence.HistoricalRankingRecord` /
  `RankingRepository`, wired through `HistoricalFeatureBuilder`).
- Identity resolution only folds accents within already-Latin text
  (Unicode NFKD + combining-mark removal); it does **not** perform
  cross-script transliteration (e.g. Cyrillic/CJK to Latin), which would
  require a curated mapping to be safe.
- `DatasetBuilder` does not enumerate the repository itself — callers
  supply the exact match IDs to include, since `MatchRepository` has no
  "list all" capability by design.
- **No production live provider is included yet.** `historical_ingestion`
  ships only a deterministic mock provider and local JSONL/CSV file
  sources — no HTTP client, no scraping, no real bookmaker/data-vendor
  integration.
- A "safe merge" (`merged_safe`, a still-`scheduled` record later
  completed by the same provider record) is the only update path;
  `MatchRepository.replace()` is a narrow, explicit primitive, not a
  general upsert — genuinely conflicting re-imports are always
  quarantined/rejected, never silently overwritten.
- `historical_ingestion` quarantine/conflict stores are per-service-instance,
  in-memory, and not persisted — restarting a process loses them (only
  the `CheckpointStore` cursor and the underlying repositories persist
  within the process's lifetime).

## Next Phase Roadmap

- A real, read-only table tennis data provider integrated behind the
  existing `HistoricalDataSource`/`HistoricalProviderAdapter` protocols
  (a sanctioned API/feed, not scraping), registered via `ProviderRegistry`.
- Training an actual statistical/ML model against the Phase 3
  `datasets/` output (chronological splits, leakage-checked), with the
  Phase 2B rules-based engine retained as an explainable baseline/fallback.
- Historical ranking-over-time data from a real provider, populating
  `RankingRepository` at scale (the model/repository/feature wiring
  already exists as of Phase 4).
- Surface- and competition-aware historical splits once real historical
  data is available (`HistoricalCompetitionRecord`/`CompetitionRepository`
  already exist as of Phase 4).
- A persistent (non-in-memory) repository backing `persistence.protocols`
  once real data volumes require it.
- A manual-review workflow (still no UI) for `pending` quarantine records,
  and durable (non-in-memory) checkpoint/quarantine storage.

## Ingestion Data Flow

```
MatchSource (protocol)
   └── MockTableTennisSource            # ingestion/sources/mock_source.py
          │  fetch_matches() -> RawMatch[]
          │  fetch_odds(match_id) -> RawOdds[]
          ▼
IngestionService.run_once()             # ingestion/service.py
          │  normalize_match(RawMatch) -> Match     (normalization/match_normalizer.py)
          │  normalize_odds(RawOdds)   -> Odds      (normalization/odds_normalizer.py)
          ▼
IngestionResult(matches: list[Match], odds: list[Odds])   # in-memory only
```

`IngestionScheduler` wraps an `IngestionService` with a configurable
polling interval and an explicit `run_once()` method. It does not start
any background thread or loop on its own — a later phase decides how
`run_once()` gets called on a schedule.

## Ingestion source: currently mocked

**The only ingestion source implemented today is `MockTableTennisSource`
(`ingestion/sources/mock_source.py`).** It returns a fixed, deterministic
set of realistic table tennis matches and odds, held entirely in memory.
It does not scrape any website, call any external API, or connect to any
betting provider. Any class implementing the `MatchSource` protocol
(`ingestion/protocols.py`) can be substituted via dependency injection
into `IngestionService` once a real provider is integrated.

## Phase 3 Architecture

```
Historical observations (HistoricalPlayerRecord / HistoricalMatchRecord / HistoricalOddsRecord)
    ↓
Identity resolution (identity.resolver.IdentityResolver)
    ↓
Validation and deduplication (persistence.models validators, persistence.in_memory)
    ↓
Temporal repositories (persistence.protocols + persistence.in_memory)
    ↓
Leakage-safe snapshots (features.snapshots.build_feature_snapshot)
    ↓
Rolling player and matchup features (features.player / features.matchup)
    ↓
Training/backtesting datasets (datasets.builder / datasets.splits)
    ↓
Existing Phase 2B analytics engine (via features.engine_adapter, where clean)
```

## Temporal Data Model & Strict Cutoff Semantics

Every timestamp on a `persistence.models` record is **timezone-aware**
— naive datetimes are rejected at construction. Source (provider)
timestamps are always kept separate from ingestion timestamps.
`HistoricalMatchRecord.effective_timestamp` is the single authoritative
"this match happened at" value used for all cutoff comparisons: it
prefers `completed_at`, then `actual_start_at`, then `scheduled_at`.

**Cutoff semantics are strict-before by default**: a query for records
"before T" never returns a record whose `effective_timestamp` is `>= T`.
`OddsRepository.list_for_match_at_or_before` is the one documented
exception (inclusive — "latest known price" semantics). This asymmetry
is intentional and tested.

## Identity Resolution

`identity.resolver.IdentityResolver` deterministically resolves a raw
player observation against known `PlayerIdentityRecord`s:

1. An **exact external identifier match** (same provider + provider
   player ID) always outranks fuzzy name comparison.
2. Otherwise, name similarity is scored with the stdlib
   `difflib.SequenceMatcher` (transparent, dependency-free) against the
   canonical name and every known alias.
3. Country/birth-date agreement adjusts the score; mismatches apply a
   configurable penalty rather than being ignored.
4. Short names (`short_name_length_threshold`) require a higher score
   margin (`short_name_extra_margin`) before being auto-matched.

Outcomes are `matched`, `created`, `ambiguous`, or `rejected` —
**ambiguous candidates are never auto-merged**; the caller decides.
`REJECTED` is reserved for a genuine data conflict: an exact external ID
match whose name similarity falls below a sanity floor.

## Rolling Feature Definitions

`features.player.build_player_rolling_features` computes, per
configured window (`last_5`, `last_10`, `last_20`, and `all_time`):
matches played/won/lost, win/set/point rates, average set/point margin,
straight-sets win rate, deciding-set appearance/win rate, first-set win
rate, comeback-after-losing-first-set rate, loss-rate-after-winning-first-set,
average match duration, and incomplete-match count — plus non-windowed
signals: rest time since the previous match, matches in the last 24h/7d,
result streak, recency-weighted win rate, and a volatility score.

**Missing-value policy:** every rate is `float | None` — `None` means
"not enough data", never a fabricated `0.0`. Every rate with a natural
denominator is paired with an explicit `_n` observation-count field.
`opponent_adjusted_win_rate` and `ranking_differential` are always
`None` in Phase 3 (see "Current Limitations").

`features.matchup.build_matchup_features` computes head-to-head win
rates, recent-meeting win rate, average set margin, deciding-set
head-to-head, days since last meeting, competition-/format-specific
splits, and rest/workload/form/volatility differentials between the two
players. **No head-to-head history is represented as no-data (`None`),
never a fabricated 50/50 split.**

## Feature Provenance & Leakage Prevention

Every `FeatureSnapshot` carries a `ProvenanceMetadata` block: the exact
source match IDs used per player and for head-to-head, the cutoff,
observation counts, warnings, missing-feature names, a data-quality
score, and two deterministic fingerprints (`repository_fingerprint` over
the source match ID set, `input_fingerprint` over the full build
inputs) — computed with `hashlib.sha256` over stable strings, **never**
over a wall-clock or other nondeterministic value.

`features.snapshots.build_feature_snapshot` asserts, before returning:
the target match's own ID never appears in its own feature history, and
every source record's `effective_timestamp` is strictly before `as_of`
— raising `features.errors.SnapshotLeakageError` otherwise.
`datasets.leakage` adds dataset/split-level checks: target-match
inclusion, source timestamps, a snapshot matching its own training
example, forbidden target fields, duplicate examples across splits, and
non-chronological split boundaries — combined into a `LeakageReport`
via `datasets.leakage.run_dataset_leakage_report`, with
`assert_no_leakage` raising `LeakageViolation` on any failure.

## Chronological Dataset Building & Walk-Forward Validation

`datasets.builder.DatasetBuilder` converts completed matches (status
`finished`/`retired` with a recorded `winner_id`) into labeled
`TrainingExample`s. The pre-match cutoff (`as_of`) defaults to the
match's own `scheduled_at` (configurable to `actual_start_at` via
`dataset_cutoff_policy`) — deliberately the safest, earliest-known
boundary. The label (`player_a_won`) comes only from the match's
recorded result; the `FeatureSnapshot` it's paired with was built to
exclude that match entirely.

**Random train/test splitting is not implemented.** `datasets.splits`
provides three chronological strategies instead:

- `chronological_holdout_split` — one train → validation → test split
  by configured ratios (`split_train_ratio`, etc.), always disjoint and
  time-ordered.
- `walk_forward_splits` — expanding-window folds: each fold's training
  set grows to include all prior folds; only *test* segments across
  folds are checked for non-overlap (training segments are expected to
  share examples by design).
- `rolling_window_splits` — fixed-size (non-expanding) training windows
  advancing by a configurable step.

## Export Formats

`datasets.export` writes JSON Lines and CSV with: alphabetically stable
column/key ordering (computed from the actual key union, not
insertion order), UTC ISO-8601 timestamps, atomic writes (temp file +
`os.replace`), and no pickle or other arbitrary-code-execution formats.
Repeated export of identical examples is byte-for-byte deterministic
(the one documented exception is `DatasetManifest.created_at`, which
reflects real build time unless explicitly injected).

## Running the Feature/Dataset Demo (Phase 3)

```bash
pip install -r requirements.txt
python -m features.demo
```

This demo (1) resolves a few player identities, including an
accent-variant name and an exact-external-ID match, (2) builds a
leakage-safe `FeatureSnapshot` for a scheduled target match from ten
illustrative/mock historical matches, printing rolling/matchup features,
observation counts, data-quality warnings, and the provenance
fingerprint, and (3) builds a small chronological training dataset from
those same matches, printing split sizes and a full leakage-check
report. It makes no network calls and never suggests or places a wager.

## Phase 4 Architecture

```
Provider export / HTTP response / local file
    ↓
Provider-specific raw adapter          (providers.generic.adapter.GenericProviderAdapter)
    ↓
Canonical historical import records    (historical_ingestion.models.Imported*)
    ↓
Validation                             (historical_ingestion.validation — structural/semantic/temporal)
    ↓
Identity resolution                    (identity.resolver.IdentityResolver, integrated in the pipeline)
    ↓
Deduplication and conflict detection   (historical_ingestion.deduplication)
    ↓
Historical repositories                (persistence.in_memory — Player/Match/Odds/Ranking/Competition)
    ↓
Import report and provenance           (historical_ingestion.reports.ImportReport)
    ↓
Existing snapshot and dataset builders (features.builder, datasets.builder)
```

`historical_ingestion.service.HistoricalImportService` is the single
orchestrated entrypoint, wiring a `HistoricalDataSource`, a
`HistoricalProviderAdapter`, an `IdentityResolver`, the five Phase 3/4
repositories, and a `CheckpointStore` — all injected, none hardcoded.

## Source and Adapter Separation

A `HistoricalDataSource` (`historical_ingestion.protocols`) only knows
how to fetch/read a batch of opaque `RawRecord`s and report a next
cursor — it never interprets provider-specific field names. Three
sources exist: `MockTableTennisProviderSource` (deterministic, in-memory,
illustrative), `JSONLFileSource`, and `CSVFileSource` (both local-file
only, sharing cursor-as-line-offset resumption via
`sources.base.FileHistoricalDataSource`). None makes a network call.

A `HistoricalProviderAdapter` (`providers`) maps a `RawRecord`'s opaque
payload into canonical `Imported*` models. `providers.generic.adapter.GenericProviderAdapter`
is entirely driven by a `ProviderMappingConfig` (`providers.models`) —
every provider-specific field name lives in that config object
(see `providers.generic.mappings.mock_provider_mapping` for the
reference example), never hardcoded in `historical_ingestion`'s core
service/pipeline. Unknown provider statuses are never silently mapped:
depending on `unknown_status_policy`, they raise
`UnknownProviderStatusError` or are left unmapped (`status=None`) with
an explicit warning attached to the record's provenance.

## Canonical Import Models

`historical_ingestion.models` defines `ImportedPlayer`, `ImportedMatch`
(+ `ImportedSet`), `ImportedOdds`, `ImportedCompetition`, and
`ImportedRanking` — deliberately more permissive than the
`persistence.models` records they may become (mirroring Phase 2A's
"permissive raw layer, strict validation stage" split). Every one
carries an `ImportProvenance`: provider, provider record ID, source
batch ID, source timestamp (kept separate from ingestion timestamp), a
deterministic raw-payload fingerprint, mapping version, and warnings.
Timestamps are timezone-aware or absent — never fabricated.

## Staged Validation

`historical_ingestion.validation` runs three stages producing typed
`ValidationIssue`s (severity `info`/`warning`/`error`/`fatal`, a code,
message, record ID, field path, stage, and provider):

- **Structural** — required fields, timestamp parseability, numeric
  ranges, negative set scores.
- **Semantic** — player A ≠ player B, winner belongs to the match,
  completed-requires-winner / scheduled-forbids-winner, unique set
  numbers, recorded winner vs. set-score consistency, sets not
  exceeding `best_of`, decimal odds > 1.
- **Temporal** — completion not preceding start, provider timestamp not
  after ingestion (beyond `timestamp_tolerance_seconds`), odds aligned
  to the match's lifecycle, no future historical record unless
  `allow_future_records` is set.

`decide_validation_outcome` applies `validation_policy`
(`strict`/`lenient`) and `warning_policy` (`accept`/`reject`) to turn a
record's issues into `accept` / `accept_with_warnings` / `reject`.

## Identity Resolution (Integrated)

Every imported player is normalized and resolved through the existing
Phase 3 `IdentityResolver`: exact external ID first, then alias, then
deterministic name similarity, with country hints reducing confidence
on mismatch. **Ambiguous or rejected resolutions are never auto-merged
or silently persisted** — they are quarantined (or rejected, per
`identity_ambiguity_policy`) with the candidate scores and reasons
preserved on the `QuarantineRecord`.

## Deduplication and Conflicts

`historical_ingestion.deduplication` computes deterministic internal IDs
— `f"{provider}:{provider_match_id}"` for matches, and a composite key
for odds — so re-importing the same provider record always maps to the
same internal ID. Outcomes: `inserted`, `skipped_idempotent` (identical
re-import), `merged_safe` (a still-`scheduled` record safely completed
by a later import of the *same* provider record — the only supported
update path, via `MatchRepository.replace()`), `quarantined` (a likely
cross-provider duplicate, by normalized players + a configurable
scheduled-time window), and `rejected_conflict` (same provider ID but a
different winner/score/schedule — **never auto-merged**; recorded as a
`ConflictRecord` with both versions' provenance).

## Checkpointing, Transactions, and Dry-Run

`ImportCheckpoint` (source, provider, cursor, last successful batch,
processed count, repository fingerprint, version, timestamp) is only
saved **after** a batch's staged records are successfully persisted —
see `historical_ingestion.service` for the exact in-memory staging
semantics (not real ACID transactions, clearly documented): a batch's
inserts/replacements/quarantine/conflict records accumulate in local
lists first, and only that batch's checkpoint advances if nothing
raised an unexpected exception while staging it. A fatal error aborts
the whole batch with zero partial writes and an unchanged checkpoint.
`InMemoryCheckpointStore` rejects a version mismatch
(`CheckpointVersionError`) rather than silently trusting stale data.

**Dry-run mode** (`service.run(dry_run=True)`) runs every step —
adapt, validate, resolve, deduplicate, report — but persists nothing
and never advances the checkpoint, including the identity
resolver's own within-run state.

## Ranking History and Competition Metadata

`persistence.models.HistoricalRankingRecord` is an append-only series
(multiple observations per player preserved, never overwritten);
`RankingRepository.latest_before(player_id, cutoff)` is strict-before,
so a snapshot never leaks a future or "current" ranking. Wired into
`HistoricalFeatureBuilder` via an optional `ranking_repository`
constructor argument — omitting it preserves exact Phase 3 behavior
(`ranking` stays `None`). `HistoricalCompetitionRecord` carries name,
country, level, format, season, indoor/outdoor, and an active date
range — every field optional, nothing invented for what a provider
didn't supply.

## Quality Metrics

Every batch and the cumulative run report an `ImportMetrics` block
(structural/semantic validity rate, identity resolution/ambiguity rate,
exact-duplicate rate, conflict rate, accepted-record rate, and
coverage figures) **always paired with `sample_size`** — a metric is
never presented without the count backing it, so a small sample can't
be mistaken for a well-supported one.

## Local File Import

`JSONLFileSource`/`CSVFileSource` read UTF-8 only, with explicit
per-source-instance record typing and (for CSV) a configurable
delimiter; row numbers are stable and malformed rows raise a clear
`SourceReadError` naming the file and row/line. Both share deterministic
batch checksums (`sources.base.compute_batch_checksum`) and
cursor-as-line-offset resumption. No pandas, no pickle, no arbitrary
code execution — standard library `json`/`csv` only. Report export
(`historical_ingestion.reports.export_report_json`) is atomic
(temp file + `os.replace`).

## Running the Historical Ingestion Demo (Phase 4)

```bash
pip install -r requirements.txt
python -m historical_ingestion.demo
```

Processes two illustrative/mock batches end to end: valid matches, an
exact-duplicate re-import (skipped), an ambiguous player identity
(quarantined), a conflicting match result (rejected as a conflict),
multi-observation odds history, and ranking history — then prints the
import report, quarantine/conflict records, repository counts, a
built `FeatureSnapshot`, and a small chronological dataset. Makes no
network calls and never suggests or places a wager.

## Requirements

- Python 3.12+ (tested against 3.11/3.12; type hints assume 3.12
  compatibility)

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
python -m app.main
```

Expected output (structured JSON logs to stdout) confirms:
1. Settings loaded successfully
2. Logging configured
3. Domain models (`Player`, `Match`) can be constructed without error

## Run tests

```bash
pip install -r requirements.txt
python -m compileall app config domain ingestion normalization engine persistence identity features datasets historical_ingestion providers
pytest -v
python -m app.main
python -m engine.demo
python -m features.demo
python -m historical_ingestion.demo
```

## Configuration

All configuration is read from environment variables, optionally via a
local `.env` file (see `.env.example`). No secrets are required at this
phase since there are no external integrations yet.

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `TT Edge Terminal` | Display name used in logs |
| `ENVIRONMENT` | `development` | Deployment environment label |
| `LOG_LEVEL` | `INFO` | Root log level |

Analytics engine weights and thresholds are configured separately via
`ANALYTICS_*` environment variables — see `config/analytics.py` and the
"Mathematical Formulas" section above.

Historical intelligence settings (rolling windows, identity thresholds,
dataset cutoff policy, split ratios, schema/builder versions, export
precision) are configured via `HISTORICAL_*` environment variables —
see `config/historical.py`. All weights, thresholds, ratios, and version
identifiers are validated at construction; invalid configuration fails
immediately with a clear `pydantic.ValidationError`.

Historical ingestion settings (batch size, validation/warning policy,
timestamp tolerance, duplicate timestamp window, identity ambiguity
policy, dry-run default, quarantine enabled, checkpoint/mapping
version, max records per run, strict mode) are configured via
`INGEST_*` environment variables — see `config/historical_ingestion.py`.
Same validation guarantee: invalid configuration fails immediately and
clearly.
