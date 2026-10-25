"""Compute guard: caps, per-client budgets, per-instance slots and time budgets."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy import optimize
from starlette.requests import Request

from backend.core import compute_guard
from backend.core.compute_guard import Deadline, JobSlots, Overloaded, Rate, TokenBuckets
from backend.main import app

client = TestClient(app)
ORIGIN = {"Origin": "http://localhost:3000"}
SIMULATE = {"model": "gbm", "parameters": {"sigma": 0.2}, "s0": 100.0, "n_steps": 21, "n_paths": 10}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_token_bucket_allows_a_burst_then_refills():
    clock = FakeClock()
    buckets = TokenBuckets(Rate(burst=2, per_minute=6), clock=clock)
    assert buckets.take("a") == buckets.take("a") == 0.0
    assert buckets.take("a") == pytest.approx(10.0)  # one token per 10 s
    assert buckets.take("b") == 0.0  # clients do not share a bucket
    clock.now = 10.0
    assert buckets.take("a") == 0.0


def test_token_buckets_forget_the_oldest_clients():
    buckets = TokenBuckets(Rate(burst=1, per_minute=1), max_clients=2, clock=FakeClock())
    for name in ("a", "b", "c"):
        buckets.take(name)
    assert buckets.take("a") == 0.0  # evicted, so it starts over with a full bucket


def _request(forwarded: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({"type": "http", "headers": headers, "client": ("10.0.0.7", 5000)})


def test_client_key_trusts_only_the_proxy_appended_address(monkeypatch):
    monkeypatch.setattr(compute_guard.settings, "TRUSTED_PROXY_HOPS", 1)
    # A client can put anything on the left; the front end appends what it saw.
    assert compute_guard.client_key(_request("6.6.6.6, 203.0.113.9")) == "203.0.113.9"
    assert compute_guard.client_key(_request()) == "10.0.0.7"
    monkeypatch.setattr(compute_guard.settings, "TRUSTED_PROXY_HOPS", 0)
    assert compute_guard.client_key(_request("203.0.113.9")) == "10.0.0.7"


def test_heavy_endpoint_answers_429_with_retry_after_and_cors(monkeypatch):
    monkeypatch.setattr(compute_guard, "_HEAVY", TokenBuckets(Rate(burst=1, per_minute=2)))
    assert client.post("/api/v1/calibration/simulate", json=SIMULATE).status_code == 200
    refused = client.post("/api/v1/calibration/simulate", json=SIMULATE, headers=ORIGIN)
    assert refused.status_code == 429
    assert refused.headers["retry-after"] == "30"
    assert refused.headers["access-control-allow-origin"] == ORIGIN["Origin"]
    assert "retry-after" in refused.headers["access-control-expose-headers"].lower()


def test_flood_of_any_request_answers_429_with_cors(monkeypatch):
    monkeypatch.setattr(compute_guard, "_REQUESTS", TokenBuckets(Rate(burst=1, per_minute=60)))
    assert client.get("/health").status_code == 200
    refused = client.get("/health", headers=ORIGIN)
    assert refused.status_code == 429
    assert refused.headers["access-control-allow-origin"] == ORIGIN["Origin"]


def test_busy_instance_answers_503(monkeypatch):
    monkeypatch.setattr(compute_guard, "_SLOTS", JobSlots(slots=1, queue_seconds=0.05))
    with compute_guard._SLOTS.hold():  # another job holds the only slot
        refused = client.post("/api/v1/calibration/simulate", json=SIMULATE)
    assert refused.status_code == 503 and int(refused.headers["retry-after"]) >= 1
    assert client.post("/api/v1/calibration/simulate", json=SIMULATE).status_code == 200


def test_slots_are_released_when_the_job_fails():
    slots = JobSlots(slots=1, queue_seconds=0.01)
    with pytest.raises(ValueError), slots.hold():
        raise ValueError("calibration blew up")
    with slots.hold():
        pass
    with slots.hold(), pytest.raises(Overloaded), slots.hold():
        pass


def test_deadline_stops_an_optimizer_with_its_best_point():
    def slow(x):
        time.sleep(0.002)
        return float(((x - 0.3) ** 2).sum())

    deadline = Deadline(0.2)
    started = time.monotonic()
    result = optimize.differential_evolution(slow, [(-1, 1)] * 4, maxiter=100_000, tol=0, seed=1,
                                             polish=False, callback=deadline.stop_when_expired())
    assert time.monotonic() - started < 1.0
    assert deadline.expired() and result.fun < slow(np.full(4, 0.9))


@pytest.mark.parametrize(("path", "body"), [
    ("/api/v1/calibration/simulate", {**SIMULATE, "n_paths": 5000}),
    ("/api/v1/calibration/simulate", {**SIMULATE, "n_steps": 504, "dt": 1 / 52}),  # 9.7 years
    ("/api/v1/calibration/calibrate", {"tickers": ["SPY"], "model": "gbm", "start_date": "2010-01-01",
                                       "end_date": "2025-01-01"}),
    ("/api/v1/calibration/calibrate", {"tickers": list("ABCDEF"), "model": "gbm",
                                       "start_date": "2024-01-01", "end_date": "2025-01-01"}),
    ("/api/v1/calibration/calibrate-qmeasure", {"ticker": "SPY", "model": "heston", "maxiter": 1000}),
    ("/api/v1/calibration/vol-surface", {"ticker": "SPY", "model": "heston", "parameters": {},
                                         "spot_price": 100, "risk_free_rate": 0.0, "n_strikes": 500}),
])
def test_oversized_requests_are_rejected_before_any_work(path, body):
    response = client.post(path, json=body)
    assert response.status_code == 422, response.text
