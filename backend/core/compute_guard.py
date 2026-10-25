"""Compute guard: what one visitor can make one instance do.

Cloud Run saturates and bills by CPU time, and a calibration or a simulation
can be made arbitrarily expensive by its inputs. Three layers bound what a
visitor can make an instance spend:

- Input caps live in the request schemas: every size a request controls is
  bounded there.
- Admission: every client has token buckets (429 when empty), and an instance
  runs at most ``HEAVY_JOB_SLOTS`` heavy jobs at once (503 when they stay full),
  so cheap endpoints keep answering while compute is saturated.
- Time budgets: long computations poll a ``Deadline`` and stop with their best
  result so far. A thread cannot be killed, so a computation that ignores its
  deadline keeps its slot; every loop a visitor can lengthen must poll one.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from fastapi import Request

from backend.core.config import settings


class Overloaded(RuntimeError):
    """The request was refused to protect the instance; retry after ``retry_after`` seconds."""

    def __init__(self, status_code: int, detail: str, retry_after: float):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.retry_after = max(1, math.ceil(retry_after))


class BudgetExceeded(RuntimeError):
    """A computation ran out of time before producing any usable result."""


@dataclass(frozen=True)
class Rate:
    """A token bucket: ``burst`` requests at once, refilled at ``per_minute``."""

    burst: int
    per_minute: float


class TokenBuckets:
    """Per-client token buckets, oldest clients evicted past ``max_clients``."""

    def __init__(self, rate: Rate, max_clients: int = 10_000, clock: Callable[[], float] = time.monotonic):
        self._rate = rate
        self._max_clients = max_clients
        self._clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = threading.Lock()

    def take(self, client: str) -> float:
        """Spend one token; return 0 if allowed, else the seconds until one is available."""
        refill_per_s = self._rate.per_minute / 60.0
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.pop(client, (float(self._rate.burst), now))
            tokens = min(float(self._rate.burst), tokens + (now - last) * refill_per_s)
            if tokens >= 1.0:
                self._buckets[client] = (tokens - 1.0, now)
                wait = 0.0
            else:
                self._buckets[client] = (tokens, now)
                wait = (1.0 - tokens) / refill_per_s
            while len(self._buckets) > self._max_clients:
                self._buckets.popitem(last=False)
        return wait


class JobSlots:
    """At most ``slots`` heavy jobs at once on this instance."""

    def __init__(self, slots: int, queue_seconds: float):
        self._semaphore = threading.BoundedSemaphore(slots)
        self._queue_seconds = queue_seconds

    @contextmanager
    def hold(self) -> Iterator[None]:
        if not self._semaphore.acquire(timeout=self._queue_seconds):
            raise Overloaded(503, "The server is busy with other computations; try again shortly.",
                             retry_after=self._queue_seconds)
        try:
            yield
        finally:
            self._semaphore.release()


class Deadline:
    """A wall-clock budget that long computations poll."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._end = clock() + seconds
        self.seconds = seconds

    def remaining(self) -> float:
        return max(0.0, self._end - self._clock())

    def expired(self) -> bool:
        return self._clock() >= self._end

    def stop_when_expired(self) -> Callable[..., bool]:
        """A scipy optimizer callback (minimize or differential_evolution) that
        stops the optimizer, keeping its best point, once the budget is spent."""

        def callback(*_args: object, **_kwargs: object) -> bool:
            if self.expired():
                raise StopIteration
            return False

        return callback


def client_key(request: Request) -> str:
    """The client's address, as seen by the last trusted proxy.

    Behind Cloud Run the front end appends the address it saw to
    X-Forwarded-For; entries further left are client-supplied and spoofable.
    """
    hops = settings.TRUSTED_PROXY_HOPS
    forwarded = request.headers.get("x-forwarded-for")
    if hops > 0 and forwarded:
        entries = [e.strip() for e in forwarded.split(",") if e.strip()]
        if len(entries) >= hops:
            return entries[-hops]
    return request.client.host if request.client else "unknown"


_REQUESTS = TokenBuckets(Rate(settings.RATE_REQUESTS_BURST, settings.RATE_REQUESTS_PER_MINUTE))
_HEAVY = TokenBuckets(Rate(settings.RATE_HEAVY_BURST, settings.RATE_HEAVY_PER_MINUTE))
_SLOTS = JobSlots(settings.HEAVY_JOB_SLOTS, settings.HEAVY_JOB_QUEUE_SECONDS)


def admit_request(request: Request) -> None:
    """Every request: a generous per-client limit against floods."""
    wait = _REQUESTS.take(client_key(request))
    if wait > 0:
        raise Overloaded(429, "Too many requests; slow down.", retry_after=wait)


def heavy_job(request: Request) -> Iterator[None]:
    """FastAPI dependency for endpoints that burn CPU: a per-client budget, then a slot."""
    wait = _HEAVY.take(client_key(request))
    if wait > 0:
        raise Overloaded(429, "Too many calibrations or simulations in a row; "
                              "wait a moment before the next one.", retry_after=wait)
    with _SLOTS.hold():
        yield
