"""Lab warm-up timing: production compiles before the server takes traffic."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from backend.core.config import settings
from backend.main import app
from backend.services import lab_service


@pytest.fixture
def warm_up_calls(monkeypatch):
    """Replace the compile with a slow recorder, so a compile left in the
    background is still running when the lifespan hands over."""
    called = threading.Event()

    def compile_slowly() -> None:
        time.sleep(0.3)
        called.set()

    monkeypatch.setattr(lab_service, "warm_up", compile_slowly)
    return called


def test_startup_mode_compiles_before_serving(monkeypatch, warm_up_calls):
    monkeypatch.setattr(settings, "LAB_WARM_UP", "startup")
    with TestClient(app):
        # The lifespan has finished, so the compile must already be done.
        assert warm_up_calls.is_set()


def test_background_mode_compiles_in_a_thread(monkeypatch, warm_up_calls):
    monkeypatch.setattr(settings, "LAB_WARM_UP", "background")
    with TestClient(app):
        assert warm_up_calls.wait(timeout=10)


def test_off_mode_skips_the_compile(monkeypatch, warm_up_calls):
    monkeypatch.setattr(settings, "LAB_WARM_UP", "off")
    with TestClient(app):
        pass
    assert not warm_up_calls.is_set()
