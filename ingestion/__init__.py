"""Provider-independent ingestion layer.

Fetches raw table tennis match/odds data from any source implementing
`ingestion.protocols.MatchSource` and hands it to `normalization` for
conversion into domain models. Currently backed only by an in-memory
mock source (`ingestion.sources.mock_source`) — no live providers, web
scraping, or database involved.
"""
