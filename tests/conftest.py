"""Shared fixtures."""

import pytest

from backend.core import compute_guard
from backend.core.config import settings


@pytest.fixture(autouse=True)
def fresh_rate_limits(monkeypatch):
    """Every test starts with full token buckets: TestClient is one client to the
    guard, and the suite as a whole makes more heavy requests than one burst."""
    monkeypatch.setattr(compute_guard, "_REQUESTS", compute_guard.TokenBuckets(
        compute_guard.Rate(settings.RATE_REQUESTS_BURST, settings.RATE_REQUESTS_PER_MINUTE)))
    monkeypatch.setattr(compute_guard, "_HEAVY", compute_guard.TokenBuckets(
        compute_guard.Rate(settings.RATE_HEAVY_BURST, settings.RATE_HEAVY_PER_MINUTE)))
