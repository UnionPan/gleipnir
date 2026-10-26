"""Spot paths under the drift policy: every model drifts at r - q unless it is OU."""

import numpy as np
import pytest

from backend.services import path_simulation

R, Q, S0 = 0.04, 0.01, 100.0
MODELS = {
    "gbm": {"sigma": 0.25, "mu": 0.40},
    "heston": {"kappa": 2.0, "theta": 0.05, "xi": 0.4, "rho": -0.6, "v0": 0.05, "mu": 0.40},
    "rough_bergomi": {"xi0": 0.05, "eta": 1.2, "rho": -0.6, "H": 0.15, "mu": 0.40},
    "merton_jump": {"sigma": 0.15, "lambda": 3.0, "mu_j": -0.08, "sigma_j": 0.1, "mu": 0.40},
    "garch": {"omega": 4e-6, "alpha": 0.08, "beta": 0.9, "mu": 0.0015},
    "regime_switching_gbm": {"mu_regime0": -0.9, "sigma_regime0": 0.45, "mu_regime1": 0.3, "sigma_regime1": 0.15,
                             "p_00": 0.95, "p_01": 0.05, "p_10": 0.02, "p_11": 0.98},
}


@pytest.mark.parametrize("model", MODELS)
def test_spot_drifts_at_r_minus_q_whatever_the_history_says(model):
    out = path_simulation.simulate(model, MODELS[model], S0, n_steps=252, n_paths=2000, dt=1 / 252,
                                   random_seed=3, risk_free_rate=R, dividend_yield=Q)
    final_mean = out["stats"]["mean"]
    standard_error = out["stats"]["std"] / np.sqrt(2000)
    assert abs(final_mean - S0 * np.exp(R - Q)) < 4 * standard_error, (model, final_mean)
    assert out["spot_drift"] == pytest.approx(R - Q)
    bands = [out["quantiles"][k] for k in ("p5", "p25", "p50", "p75", "p95")]
    assert all(len(band) == 253 for band in bands)
    finals = [band[-1] for band in bands]
    assert finals == sorted(finals)


def test_scenarios_appear_only_beyond_three_months():
    long_run = path_simulation.simulate("gbm", MODELS["gbm"], S0, 252, 500, 1 / 252, random_seed=1)
    labels = [s["label"] for s in long_run["scenarios"]]
    assert labels[0].startswith("Positive excess drift (+5%") and labels[1].startswith("Negative excess drift (-5%")
    assert "historical drift continued (40%" in labels[2] and "not a forecast" in labels[2]
    median = np.array(long_run["quantiles"]["p50"])
    assert np.array(long_run["scenarios"][0]["median"])[-1] == pytest.approx(median[-1] * np.exp(0.05))

    short = path_simulation.simulate("gbm", MODELS["gbm"], S0, 21, 500, 1 / 252, random_seed=1)
    assert [s["excess_drift"] for s in short["scenarios"]] == [pytest.approx(0.40 - 0.04)]  # only the historical line


def test_ou_follows_its_fitted_level_and_says_so():
    params = {"kappa": 20.0, "theta": float(np.log(150.0)), "sigma": 0.2}
    out = path_simulation.simulate("ou", params, S0, 252, 400, 1 / 252, random_seed=1)
    assert out["spot_drift"] is None and out["scenarios"] == []
    assert "not as a forecast" in out["drift_note"]
    assert out["stats"]["p50"] == pytest.approx(150.0, rel=0.05)
