"""Startup smoke test: bootstrap and main entrypoint wire up without error."""

from __future__ import annotations

import logging

from app.bootstrap import AppContext, bootstrap
from app.main import main


def test_bootstrap_returns_ready_app_context():
    ctx = bootstrap()
    assert isinstance(ctx, AppContext)
    assert isinstance(ctx.logger, logging.Logger)
    assert ctx.settings.app_name


def test_main_runs_without_raising(capsys):
    main()
    captured = capsys.readouterr()
    assert "ready" in captured.out
