"""Real-world fits through the API: every model, with uncertainty and an educational drift."""

import json
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import physical_fits
from backend.services.calibration_service import CalibrationService

client = TestClient(app)
MODELS = [("gbm", "mle"), ("ou", "mle"), ("heston", "moments"), ("rough_bergomi", "variogram"),
          ("regime_switching_gbm", "threshold"), ("merton_jump", "mle"), ("garch", "mle"),
          *((m, "bayesian") for m in ("gbm", "heston", "rough_bergomi", "regime_switching_gbm", "merton_jump",
                                      "garch"))]


def _history(n=1260, seed=0):
    """Five years of a GBM with a 30% crash and rebound, so every regime occurs."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.015, n)
    returns[400:440] -= 0.009  # a drawdown deep enough to label a bear regime
    returns[700:740] += 0.009  # and a rally for a bull one
    closes = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    start = date(2021, 1, 4)
    dates = [(start + timedelta(days=int(d * 1.45))).isoformat() for d in range(n + 1)]
    return {"ticker": "TEST", "dates": dates, "open": closes.tolist(), "high": closes.tolist(),
            "low": closes.tolist(), "close": closes.tolist(), "volume": [1e6] * (n + 1)}


@pytest.fixture
def offline_history(monkeypatch):
    monkeypatch.setattr(CalibrationService, "fetch_ohlcv_data", staticmethod(lambda **_: [_history()]))


@pytest.mark.parametrize(("model", "method"), MODELS)
def test_every_model_reports_estimates_errors_and_drift(offline_history, model, method):
    response = client.post("/api/v1/calibration/calibrate", json={
        "tickers": ["TEST"], "model": model, "method": method,
        "start_date": "2021-01-04", "end_date": "2026-01-02", "include_drift": True})
    assert response.status_code == 200, response.text
    json.loads(response.text)  # strict JSON: no NaN or Infinity
    result = response.json()[0]
    diagnostics = result["diagnostics"]
    assert all(np.isfinite(v) for v in result["parameters"].values())
    assert diagnostics["method_label"] and diagnostics["converged"] is not False
    errors = diagnostics["std_errors"]
    assert errors and set(errors) <= set(result["parameters"])
    drift = diagnostics["drift"]
    assert drift["ci_low"] < drift["estimate"] < drift["ci_high"]
    assert drift["std_error"] == pytest.approx(0.015 * np.sqrt(252) / np.sqrt(5.0), rel=0.2)


def test_drift_can_be_left_out_of_the_parameters(offline_history):
    response = client.post("/api/v1/calibration/calibrate", json={
        "tickers": ["TEST"], "model": "gbm", "method": "mle",
        "start_date": "2021-01-04", "end_date": "2026-01-02", "include_drift": False})
    result = response.json()[0]
    assert "mu" not in result["parameters"] and result["diagnostics"]["drift"] is not None


def test_older_method_names_map_to_the_real_estimator():
    assert physical_fits.canonical_method("heston", "particle_filter") == "moments"
    assert physical_fits.canonical_method("rough_bergomi", "moment_matching") == "variogram"
    with pytest.raises(ValueError, match="fitted by 'bayesian' or 'mle'"):
        physical_fits.canonical_method("garch", "particle_filter")
    assert physical_fits.canonical_method("garch", None) == "bayesian"
    assert physical_fits.canonical_method("ou", None) == "mle"
    with pytest.raises(ValueError, match="Unsupported model"):
        physical_fits.canonical_method("sabr", "mle")


def test_ou_simulation_reverts_to_the_fitted_log_level():
    result = CalibrationService.simulate_model(
        model="ou", parameters={"kappa": 20.0, "theta": float(np.log(150.0)), "sigma": 0.2},
        s0=100.0, n_steps=252, n_paths=400, dt=1 / 252, max_paths_return=5, random_seed=1)
    assert result["stats"]["p50"] == pytest.approx(150.0, rel=0.05)
