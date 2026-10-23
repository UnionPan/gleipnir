"""HTTP error contract: bad input 400, upstream data failure 502."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.calibration_service import CalibrationService
from backend.services.market_data import MarketDataError

client = TestClient(app)
BODY = {"tickers": ["SPY"], "start_date": "2025-01-01", "end_date": "2025-06-01",
        "model": "gbm", "method": "mle", "include_drift": True}


def test_upstream_data_failure_is_a_bad_gateway(monkeypatch):
    def unavailable(**_):
        raise MarketDataError("Failed to fetch data for SPY: rate limited")

    monkeypatch.setattr(CalibrationService, "fetch_ohlcv_data", staticmethod(unavailable))
    response = client.post("/api/v1/calibration/calibrate", json=BODY)
    assert response.status_code == 502
    assert "rate limited" in response.json()["detail"]


def test_unsupported_model_is_a_bad_request(monkeypatch):
    monkeypatch.setattr(CalibrationService, "fetch_ohlcv_data",
                        staticmethod(lambda **_: [{"ticker": "SPY", "close": [100.0, 101.0, 99.0],
                                                    "dates": ["a", "b", "c"]}]))
    response = client.post("/api/v1/calibration/calibrate", json={**BODY, "model": "nope"})
    assert response.status_code == 400
    assert "Unsupported model" in response.json()["detail"]


@pytest.mark.parametrize("path", ["/api/v1/pricing/price", "/api/v1/lab/hedge"])
def test_malformed_body_is_unprocessable(path):
    assert client.post(path, json={"unexpected": True}).status_code == 422
