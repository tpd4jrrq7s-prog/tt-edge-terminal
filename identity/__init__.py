"""Deterministic player identity normalization and resolution.

Never auto-merges ambiguous identities; exact external identifiers
always outrank fuzzy name matches; all scoring is transparent and
dependency-free (stdlib `difflib`).
"""
