"""An explicit, non-global provider adapter registry.

Not a singleton — each `ProviderRegistry` instance is constructed and
owned by whoever wires up the import service, keeping this free of
global mutable state.
"""

from __future__ import annotations

from historical_ingestion.protocols import HistoricalProviderAdapter
from providers.errors import ProviderNotRegisteredError


class ProviderRegistry:
    """Maps provider name -> `HistoricalProviderAdapter` instance."""

    def __init__(self) -> None:
        self._adapters: dict[str, HistoricalProviderAdapter] = {}

    def register(self, adapter: HistoricalProviderAdapter) -> None:
        self._adapters[adapter.provider] = adapter

    def get(self, provider: str) -> HistoricalProviderAdapter:
        adapter = self._adapters.get(provider)
        if adapter is None:
            raise ProviderNotRegisteredError(f"No adapter registered for provider {provider!r}")
        return adapter

    def providers(self) -> list[str]:
        return sorted(self._adapters.keys())
