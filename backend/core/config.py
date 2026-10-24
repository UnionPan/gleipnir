"""Backend configuration using environment variables."""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Gleipnir API"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Market data, model calibration and the Counterfactual Hedging Lab, built on mjollnir"

    # CORS
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # Backend dev server
        "http://localhost:8080",  # Alternative frontend port
    ]

    # Data Configuration
    DEFAULT_RISK_FREE_RATE: float = 0.05
    DEFAULT_DATA_SOURCE: str = "yfinance"

    # When the lab compiles its simulations. "startup" finishes before the server
    # accepts connections: Cloud Run gives an instance full CPU only while it starts
    # or serves a request, so a background compile there crawls for minutes.
    # "background" serves at once and compiles in a thread (dev, so reloads stay
    # instant); "off" compiles on the first request of each kind.
    LAB_WARM_UP: Literal["startup", "background", "off"] = "startup"

    # Cache Configuration
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 300  # 5 minutes

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
