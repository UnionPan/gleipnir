"""The Calibration-to-Lab handoff: a fit's bundle runs in the lab, hashed and bounded."""

import copy
import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import lab_bundle
from backend.services.calibration_service import CalibrationService
from tests.test_physical_fits import _history

client = TestClient(app)


@pytest.fixture
def offline(monkeypatch):
    # rate and dividend yield are stubbed for every test (conftest)
    monkeypatch.setattr(CalibrationService, "fetch_ohlcv_data", staticmethod(lambda **_: [_history()]))


def _bundle(model, method):
    response = client.post("/api/v1/calibration/calibrate", json={
        "tickers": ["TEST"], "model": model, "method": method,
        "start_date": "2021-01-04", "end_date": "2026-01-02", "include_drift": True})
    assert response.status_code == 200, response.text
    return response.json()[0]["lab_bundle"]


def _run(bundle, **overrides):
    body = {"calibration": bundle, "option": {"maturity_days": 21}, "seed": 2, **overrides}
    return client.post("/api/v1/lab/hedge", json=body)


@pytest.mark.parametrize(("model", "method"), [("heston", "moments"), ("garch", "mle"), ("regime_switching_gbm", "threshold"),
                                             ("garch", "bayesian"), ("regime_switching_gbm", "bayesian")])
def test_a_fit_runs_in_the_lab_as_the_same_book_every_time(offline, model, method):
    bundle = _bundle(model, method)
    assert bundle["rate"] == 0.041 and bundle["dividend_yield"] == 0.012 and "mu" not in bundle["parameters"]
    stored = json.loads(json.dumps(bundle))  # what the browser keeps and sends back
    first, second = _run(stored), _run(stored)
    assert first.status_code == 200, first.text
    book = first.json()["book"]
    assert book["asset"] == "TEST" and book["model"] == model and book["rate"] == 0.041
    assert book["origin"]["kind"] == "visitor" and book["origin"]["calibration_hash"] == bundle["hash"]
    assert book["params_hash"] == second.json()["book"]["params_hash"]


def test_models_the_lab_does_not_run_get_no_bundle(offline):
    assert _bundle("ou", "mle") is None


def test_an_edited_bundle_shows_its_true_hash(offline):
    bundle = _bundle("gbm", "mle")
    edited = copy.deepcopy(bundle)
    edited["parameters"]["sigma"] *= 1.1
    book = _run(edited).json()["book"]
    assert book["origin"]["calibration_hash"] != bundle["hash"]
    assert book["origin"]["calibration_hash"] == lab_bundle.bundle_hash(edited)


@pytest.mark.parametrize(("change", "message"), [
    (lambda b: b["parameters"].update(sigma=9.0), "outside the lab's range"),
    (lambda b: b["parameters"].update(sigma=float("nan")), ""),
])
def test_out_of_range_parameters_are_refused(offline, change, message):
    bundle = _bundle("gbm", "mle")
    change(bundle)
    response = client.post("/api/v1/lab/hedge", content=json.dumps(
        {"calibration": bundle, "option": {"maturity_days": 21}}, allow_nan=True),
        headers={"content-type": "application/json"})
    assert response.status_code in (400, 422), response.text
    assert message in response.text


def test_a_request_names_exactly_one_source(offline):
    bundle = _bundle("gbm", "mle")
    assert _run(bundle, asset="AAPL").status_code == 422
    assert client.post("/api/v1/lab/hedge", json={"option": {"maturity_days": 21}}).status_code == 422


def test_shipped_books_report_their_origin():
    book = client.post("/api/v1/lab/hedge", json={"asset": "KO", "option": {"maturity_days": 21}}).json()["book"]
    assert book["origin"]["kind"] == "shipped" and len(book["origin"]["calibration_hash"]) == 64
    assert book["origin"]["historical_drift"]["ci_low"] < book["origin"]["historical_drift"]["ci_high"]
