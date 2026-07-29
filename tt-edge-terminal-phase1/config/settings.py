"""Application configuration, loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings.

    Values are read from environment variables first, then from a local
    `.env` file if present (see `.env.example` for the supported keys).
    No secrets are hardcoded here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="TT Edge Terminal")
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance loaded from environment/.env."""
    return Settings()
