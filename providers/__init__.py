"""Provider adapters: map provider-specific raw records into canonical import models.

No provider field names are hardcoded outside `providers.generic.mappings`
(and any future provider-specific mapping module) — the core
`historical_ingestion` service depends only on the
`historical_ingestion.protocols.HistoricalProviderAdapter` Protocol.
"""
