# TT Edge Terminal

A modular, cloud-ready analytics and decision-support platform for table
tennis: live scores, live odds movements, historical performance,
momentum, value opportunities, and risk metrics.

**This is an analytics and decision-support platform only.** It does not
place bets, does not automate any betting action, and never will — no
component in this codebase has write access to any bookmaker or betting
system.

## Status: Phase 1 — Foundation

This phase establishes the project skeleton only:

- Typed configuration via Pydantic Settings
- Structured (JSON) logging
- Core domain models: `Player`, `Match`, `Odds`

No external data sources, no database, and no dashboard are implemented
yet — those arrive in later phases.

## Project Structure

```
app/
  main.py        Entrypoint (python -m app.main)
  bootstrap.py    Startup wiring: loads settings, configures logging
config/
  settings.py    Typed application settings (Pydantic Settings)
  logging.py     Structured JSON logging setup
domain/
  player.py      Player entity
  match.py       Match entity (+ MatchStatus, SetScore)
  odds.py        Odds entity
```

## Requirements

- Python 3.12+

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

See project planning notes for the full v1.0/v2.0 roadmap. Immediate
next phase: data ingestion layer (`ingestion/`) and normalization
(`normalization/`), still with no database — in-memory only.
