"""Phase 2B analytics engine: a transparent, deterministic, rules-based
feature-engineering and analytics layer for table tennis match analysis.

This is not yet machine learning — every score is computed from an
explicit, documented formula over real input data, never fabricated.
See `engine.orchestrator.MatchAnalyticsEngine` for the single entrypoint,
and `engine.models` for every typed input/output contract.

Read-only and analytics-focused: no database, no external API calls, no
web scraping, no background threads, and no automated wagering.
"""

from __future__ import annotations

from engine.models import MatchAnalysis, MatchAnalysisRequest
from engine.orchestrator import MatchAnalyticsEngine

__all__ = ["MatchAnalyticsEngine", "MatchAnalysisRequest", "MatchAnalysis"]
