"""Normalization layer: converts raw ingestion models into validated domain models.

Raises clear, domain-specific errors (`MatchNormalizationError`,
`OddsNormalizationError`) on invalid input rather than silently
discarding or coercing bad data.
"""
