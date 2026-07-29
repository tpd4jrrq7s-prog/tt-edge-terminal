"""Persistence layer: temporal historical models, repository protocols, and an
in-memory implementation for development and tests.

No database in this phase. Cutoff semantics are strict-before by
default; see `persistence.protocols` for the one documented exception
(odds "at or before" queries).
"""
