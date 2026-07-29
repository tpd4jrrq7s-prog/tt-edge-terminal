# TT Edge Terminal

A modular, cloud-ready analytics and decision-support platform for table
tennis: live scores, live odds movements, historical performance,
momentum, value opportunities, and risk metrics.

**This is an analytics and decision-support platform only.** It does not
place bets, does not automate any betting action, and never will — no
component in this codebase has write access to any bookmaker or betting
system.

## Status: Phase 2B — Deterministic Analytics Engine

Phase 1 established the project skeleton. Phase 2A added
provider-independent ingestion and normalization. Phase 2B adds a
transparent, deterministic, rules-based analytics and feature-engineering
layer on top:

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
- A comprehensive `tests/` suite (169 tests) covering domain models,
  ingestion, normalization, and every engine module

**Important:** the probability/risk/value model in `engine/` is
**deterministic and rules-based, not trained machine learning.** Every
number it produces traces back to an explicit, documented formula over
the data actually supplied — nothing is fabricated when data is missing.
It is the feature-engineering and transparency layer intended to
eventually feed a statistical/ML model, not that model itself.

Still no database, no web scraping, no live betting providers, no
FastAPI/HTTP API, and no automated betting — this remains a read-only,
analytics-focused platform.

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
tests/
  domain/               Tests for Player, Match, Odds
  app/                  Startup smoke test
  ingestion/            Tests for raw models, mock source, service, scheduler
  normalization/        Tests for match/odds normalization, including invalid input
  engine/               Tests for every engine module, config validation, and the orchestrator
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

## Next Phase Roadmap

- A real, read-only table tennis data provider integrated behind the
  existing `MatchSource` protocol (a sanctioned API/feed, not scraping).
- Calibrating the probability model against real outcomes and
  introducing an actual statistical/ML model behind the same
  `engine.models` contracts, with the rules-based engine retained as an
  explainable baseline/fallback.
- Surface- and competition-aware historical splits once real historical
  data is available.

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
python -m compileall app config domain ingestion normalization engine
pytest -v
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
