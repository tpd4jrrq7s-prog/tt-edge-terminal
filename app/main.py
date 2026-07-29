"""Entrypoint for TT Edge Terminal.

Run with:
    python -m app.main

Phase 1 scope: verify the foundation (config, logging, domain models)
is wired correctly. No external data sources, database, or dashboard
are involved yet.
"""

from __future__ import annotations

from app.bootstrap import bootstrap
from domain.match import Match, MatchStatus
from domain.player import Player


def main() -> None:
    """Start the application and verify the foundation is wired correctly."""
    ctx = bootstrap()
    logger = ctx.logger

    logger.info(f"{ctx.settings.app_name} starting")
    logger.info(f"Environment: {ctx.settings.environment}")

    # Sanity check only: construct domain objects in memory to confirm
    # the domain layer is wired correctly. No data source is queried.
    player_one = Player(id="p1", name="Ma Long", country="CHN")
    player_two = Player(id="p2", name="Fan Zhendong", country="CHN")
    match = Match(
        id="m1",
        player_one=player_one,
        player_two=player_two,
        status=MatchStatus.SCHEDULED,
    )

    logger.info(
        f"Domain layer OK — sample match created: "
        f"{match.player_one.name} vs {match.player_two.name} "
        f"(status={match.status.value})"
    )
    logger.info(f"{ctx.settings.app_name} ready")


if __name__ == "__main__":
    main()
