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

    # When the compiled programs (lab simulations, Heston chain fit) compile.
    # "startup" finishes before the server accepts connections: Cloud Run gives
    # an instance full CPU only while it starts or serves a request, so a
    # background compile there crawls for minutes. "background" serves at once
    # and compiles in a thread (dev, so reloads stay instant); "off" compiles on
    # the first request of each kind.
    WARM_UP: Literal["startup", "background", "off"] = "startup"

    # Compute guard (backend/core/compute_guard.py). Proxies that append to
    # X-Forwarded-For in front of the app: Cloud Run's front end is one.
    TRUSTED_PROXY_HOPS: int = 1
    # Every request, per client: a page load makes a few dozen.
    RATE_REQUESTS_BURST: int = 120
    RATE_REQUESTS_PER_MINUTE: float = 240.0
    # Calibrations, simulations and lab runs, per client.
    RATE_HEAVY_BURST: int = 20
    RATE_HEAVY_PER_MINUTE: float = 12.0
    # Heavy jobs running at once on an instance (its vCPUs), and how long a
    # request waits for a free slot before the API answers 503.
    HEAVY_JOB_SLOTS: int = 2
    HEAVY_JOB_QUEUE_SECONDS: float = 15.0
    # Wall-clock budget of one calibration; optimizers stop with their best point.
    CALIBRATION_TIME_BUDGET_SECONDS: float = 20.0

    # Cache Configuration
    ENABLE_CACHE: bool = True
    CACHE_TTL_SECONDS: int = 300  # 5 minutes

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
