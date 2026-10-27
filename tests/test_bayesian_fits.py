"""Bayesian fits: posteriors from the shipped networks, sector priors, and predictive checks."""

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy import stats

from backend.main import app
from backend.services import bayesian_fits, physical_fits
from backend.services.calibration_service import CalibrationService
from tests.test_physical_fits import _history

client = TestClient(app)
NETWORK_MODELS = list(bayesian_fits.NPE_NAME)


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(CalibrationService, "fetch_ohlcv_data",
                        staticmethod(lambda tickers, **_: [{**_history(), "ticker": tickers[0]}]))


def _calibrate(model, **extra):
    response = client.post("/api/v1/calibration/calibrate", json={
        "tickers": ["XOM"], "model": model, "start_date": "2021-01-04", "end_date": "2026-01-02", **extra})
    assert response.status_code == 200, response.text
    return response.json()[0]


@pytest.mark.parametrize("model", ["gbm", *NETWORK_MODELS])
def test_the_default_fit_is_a_posterior_with_intervals_and_a_predictive_check(offline, model):
    result = _calibrate(model)
    assert result["method"] == "bayesian"
    posterior = result["diagnostics"]["posterior"]
    for name, summary in posterior["parameters"].items():
        assert summary["q05"] <= summary["mean"] <= summary["q95"], name
        assert summary["sd"] > 0, name
    for name, row in posterior["identification"].items():
        assert row["contraction"] <= 1 and 0.85 < row["coverage_90"] < 0.95, name
    check = result["diagnostics"]["predictive_check"]
    assert check["n_replicates"] == bayesian_fits.N_REPLICATES
    assert [a["aspect"] for a in check["aspects"]] == list(bayesian_fits.ASPECTS)
    assert all(a["verdict"] in ("fits", "more", "less") for a in check["aspects"])


def test_a_sector_prior_applies_to_its_stocks_and_a_broad_one_otherwise(offline):
    assert "Energy sector prior" in _calibrate("heston")["diagnostics"]["posterior"]["prior"]
    broad = _calibrate("heston", prior="broad")["diagnostics"]["posterior"]
    assert broad["prior_kind"] == "broad" and broad["prior"] == "broad training prior"
    prior, kind, label = bayesian_fits.prior_for("heston", "NOTATICKER", "sector")
    assert prior is None and kind == "broad" and "not in the S&P 500" in label


def test_gbm_posterior_is_the_exact_inverse_chi_one():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 0.02, 500)
    prices = 100 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    _, diagnostics = bayesian_fits.fit("gbm", "SYN", prices, returns, "sector")
    sigma = diagnostics["posterior"]["parameters"]["sigma"]
    nu = returns.size - 1
    scale = math.sqrt(nu * returns.var(ddof=1) * 252)
    draws = scale / np.sqrt(stats.chi2.rvs(nu, size=400_000, random_state=1))
    assert sigma["mean"] == pytest.approx(draws.mean(), rel=1e-3)
    assert sigma["sd"] == pytest.approx(draws.std(), rel=1e-2)
    assert sigma["q05"] == pytest.approx(np.quantile(draws, 0.05), rel=1e-3)


def test_gbm_cannot_reproduce_fat_tails_and_says_so():
    rng = np.random.default_rng(5)
    returns = 0.012 * rng.standard_t(3, 1000) / math.sqrt(3)
    prices = 100 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    _, diagnostics = bayesian_fits.fit("gbm", "SYN", prices, returns, "broad")
    tails = next(a for a in diagnostics["predictive_check"]["aspects"] if a["aspect"] == "Fat tails")
    assert tails["verdict"] == "more" and "fat tails" in diagnostics["note"]


def test_a_short_window_is_refused_with_its_length(offline):
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 200)))
    with pytest.raises(ValueError, match="at least 240 daily returns"):
        physical_fits.fit("heston", "bayesian", prices, [str(i) for i in range(200)], True)


def test_student_t_garch_runs_in_the_lab_and_the_simulator(offline):
    result = _calibrate("garch")
    assert result["parameters"]["nu"] > 2.5 and result["lab_bundle"]["parameters"]["nu"] > 2.5
    simulated = CalibrationService.simulate_model(
        model="garch", parameters=result["parameters"], s0=100.0, n_steps=252, n_paths=2000, dt=1 / 252,
        random_seed=3, risk_free_rate=0.04, dividend_yield=0.0)
    terminal_mean = simulated["stats"]["mean"]
    assert terminal_mean == pytest.approx(100 * math.exp(0.04), rel=0.03)


def test_two_regime_posterior_simulates(offline):
    result = _calibrate("regime_switching_gbm")
    params = result["parameters"]
    assert params["sigma_regime0"] < params["sigma_regime1"]
    simulated = CalibrationService.simulate_model(
        model="regime_switching_gbm", parameters=params, s0=100.0, n_steps=126, n_paths=500, dt=1 / 252,
        random_seed=1)
    assert np.all(np.isfinite(simulated["quantiles"]["p50"]))
