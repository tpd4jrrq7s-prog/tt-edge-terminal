# TT Edge Terminal

A modular, cloud-ready analytics and decision-support platform for table
tennis: live scores, live odds movements, historical performance,
momentum, value opportunities, and risk metrics.

**This is an analytics and decision-support platform only.** It does not
place bets, does not automate any betting action, and never will — no
component in this codebase has write access to any bookmaker or betting
system.

## Status: Phase 2A — Provider-Independent Ingestion

Phase 1 established the project skeleton (typed config, structured
logging, core domain models). Phase 2A adds a provider-independent
ingestion and normalization pipeline on top of that foundation:

- Typed configuration via Pydantic Settings
- Structured (JSON) logging
- Core domain models: `Player`, `Match`, `Odds`
- Provider-independent ingestion (`ingestion/`), currently backed by an
  **in-memory mock data source only** — see "Ingestion source" below
- Normalization (`normalization/`) that validates and converts raw
  provider data into domain models, raising clear errors on bad input
- A comprehensive `tests/` suite covering domain models, ingestion, and
  normalization

Still no database, no web scraping, no live betting providers, no
FastAPI/HTTP API, and no automated betting — this remains a read-only,
analytics-focused platform.

## Project Structure

```
app/
  main.py             Entrypoint (python -m app.main)
  bootstrap.py        Startup wiring: loads settings, configures logging
config/
  settings.py         Typed application settings (Pydantic Settings)
  logging.py          Structured JSON logging setup
domain/
  player.py           Player entity
  match.py            Match entity (+ MatchStatus, SetScore)
  odds.py             Odds entity
engine/
  momentum.py         Reserved for momentum analytics (future phase)
  risk.py             Reserved for risk analytics (future phase)
  value.py            Reserved for value-opportunity analytics (future phase)
ingestion/
  models.py           Typed, permissive raw provider models (RawPlayer, RawMatch, RawOdds)
  protocols.py        MatchSource protocol — the provider-independent source interface
  service.py          IngestionService: fetches + normalizes into domain models
  scheduler.py        IngestionScheduler: configurable, explicit run-once polling
  sources/
    mock_source.py    MockTableTennisSource: deterministic, in-memory mock data
normalization/
  match_normalizer.py Converts RawMatch -> domain Match (validates players, status, schedule)
  odds_normalizer.py  Converts RawOdds -> domain Odds (validates odds, timestamp)
tests/
  domain/             Tests for Player, Match, Odds
  app/                Startup smoke test
  ingestion/          Tests for raw models, mock source, service, scheduler
  normalization/      Tests for match/odds normalization, including invalid input
```

## Data Flow

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

## Roadmap

See project planning notes for the full v1.0/v2.0 roadmap.

**Next planned phase (2B):** a real, read-only table tennis data
provider integration behind the existing `MatchSource` protocol (no
scraping of arbitrary sites — a sanctioned API/feed), plus wiring
`engine/` (momentum, risk, value) analytics on top of normalized data.
Still no database and no betting automation at that stage.
