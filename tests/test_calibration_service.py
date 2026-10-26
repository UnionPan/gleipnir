"""Tests for the calibration service behind the Calibration page.

Offline tests run by default. Tests marked ``network`` fetch live market data and
are deselected unless requested with ``-m network``.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from backend.services.calibration_service import CalibrationService
from backend.services.market_data import dividend_yield

DT = 1.0 / 252


class TestDividendYield:
    """yfinance reports `dividendYield` in percent; the service needs a decimal."""

    def test_sub_one_percent_payer_is_not_read_as_tens_of_percent(self):
        aapl = {"dividendYield": 0.31, "dividendRate": 1.08, "currentPrice": 347.7,
                "trailingAnnualDividendYield": 0.0030}
        assert dividend_yield(aapl) == pytest.approx(1.08 / 347.7)
        assert dividend_yield(aapl) < 0.01

    def test_falls_back_to_trailing_yield(self):
        assert dividend_yield({"trailingAnnualDividendYield": 0.0218}) == pytest.approx(0.0218)

    def test_non_payer_is_zero(self):
        assert dividend_yield({"dividendYield": None, "currentPrice": 210.0}) == 0.0


class TestPMeasureOffline:
    def test_gbm_calibration_recovers_volatility(self):
        rng = np.random.default_rng(3)
        sigma = 0.3
        log_returns = (0.05 - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * rng.standard_normal(2000)
        prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
        dates = [str(i) for i in range(len(prices))]
        params, _ = CalibrationService.calibrate_model(
            ticker="SYN", prices=prices, dates=dates, model="gbm", method="mle", include_drift=True)
        assert params["sigma"] == pytest.approx(sigma, rel=0.05)

    def test_ou_calibration_recovers_mean_reversion_of_the_log_price(self):
        rng = np.random.default_rng(5)
        kappa, theta, sigma, n = 4.0, np.log(100.0), 0.15, 5000
        x = np.empty(n)
        x[0] = theta
        a = np.exp(-kappa * DT)
        sd = sigma * np.sqrt((1 - a**2) / (2 * kappa))
        for i in range(1, n):
            x[i] = theta + (x[i - 1] - theta) * a + sd * rng.standard_normal()
        params, diagnostics = CalibrationService.calibrate_model(
            ticker="SYN", prices=np.exp(x), dates=[str(i) for i in range(n)], model="ou", method="mle",
            include_drift=False)
        assert params["kappa"] == pytest.approx(kappa, rel=0.3)
        assert params["theta"] == pytest.approx(theta, abs=0.02)
        assert diagnostics["derived"]["long_run_level"] == pytest.approx(100.0, rel=0.02)

    def test_regime_switching_reports_regimes_in_matrix_order(self):
        rng = np.random.default_rng(1)
        drifts = np.concatenate([np.zeros(200), np.full(40, np.log(0.65) / 40), np.full(60, np.log(1.45) / 60)])
        prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(drifts + 0.004 * rng.standard_normal(300))]))
        dates = [str(d) for d in np.arange("2024-01-01", prices.size, dtype="datetime64[D]")]
        params, _ = CalibrationService.calibrate_model(
            ticker="SYN", prices=prices, dates=dates, model="regime_switching_gbm", method="threshold",
            include_drift=True)
        n = sum(1 for key in params if key.startswith("mu_regime"))
        assert n == 3
        assert params["mu_regime0"] < 0 < params["mu_regime2"]  # bear, then bull
        for i in range(n):
            assert sum(params[f"p_{i}{j}"] for j in range(n)) == pytest.approx(1.0)

    @pytest.mark.parametrize("model, params", [
        ("merton_jump", {"mu": 0.05, "sigma": 0.2, "lambda": 1.0, "mu_j": -0.05, "sigma_j": 0.1}),
        ("regime_switching_gbm", {"mu_regime0": 0.05, "mu_regime1": -0.1, "sigma_regime0": 0.15,
                                  "sigma_regime1": 0.4, "p_00": 0.95, "p_01": 0.05,
                                  "p_10": 0.1, "p_11": 0.9}),
    ])
    def test_seeded_simulation_is_reproducible(self, model, params):
        kw = {"model": model, "parameters": params, "s0": 100.0, "n_steps": 50, "n_paths": 64,
              "dt": DT, "random_seed": 11}
        first = CalibrationService.simulate_model(**kw)
        again = CalibrationService.simulate_model(**kw)
        other = CalibrationService.simulate_model(**{**kw, "random_seed": 12})
        assert first["mean_path"] == again["mean_path"]
        assert first["mean_path"] != other["mean_path"]


class TestVolSurfaceOffline:
    def test_heston_surface_has_no_placeholder_vols(self):
        surface = CalibrationService.generate_vol_surface(
            ticker="SYN", model="heston",
            parameters={"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.5, "rho": -0.7},
            spot_price=100.0, risk_free_rate=0.03, n_strikes=21, n_maturities=6,
            strike_range=(0.5, 1.5), maturity_range=(0.05, 1.0),
        )
        vols = np.array([[np.nan if v is None else v for v in row] for row in surface["vols"]])
        assert vols.shape == (6, 21)
        atm = vols[:, 10]
        assert np.isfinite(atm).all() and (0.1 < atm).all() and (atm < 0.4).all()
        # A failed inversion used to be reported as exactly 20% vol.
        assert not np.any(vols == 0.2)
        # Negative rho: downside strikes carry higher implied vol (equity skew).
        assert np.nanmean(vols[:, 5]) > np.nanmean(vols[:, 15])


@pytest.mark.network
class TestLiveData:
    def test_fetch_ohlcv_returns_recent_prices(self):
        end = datetime.now()
        data = CalibrationService.fetch_ohlcv_data(
            tickers=["SPY"], start_date=(end - timedelta(days=90)).strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"))
        assert data[0]["ticker"] == "SPY"
        assert len(data[0]["close"]) > 40
        assert all(p > 0 for p in data[0]["close"])

    def test_option_chain_uses_a_decimal_dividend_yield(self):
        chain = CalibrationService.fetch_option_chain(ticker="AAPL", risk_free_rate=0.04)
        assert chain["n_options"] > 100
        assert 0.0 <= chain["dividend_yield"] < 0.05
