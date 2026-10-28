"""Spot paths from a calibrated real-world model, under the drift policy.

The simulated spot drifts at r - q with no excess drift: a stock's historical
drift is far too uncertain to forecast with. Beyond 3 months the policy adds
explicit positive and negative excess-drift scenarios, and the historical
drift is drawn as what it would imply, labelled as such.

Every model here except OU has its drift additive in the log price, so a drift
shifted by d moves every path by the factor exp(d t): scenario and historical
medians are the policy median rescaled, with no extra simulation. OU's fitted
long-run level is itself a statement about the past; it is simulated as fitted
and says so.
"""

from __future__ import annotations

import numpy as np
from mjollnir.processes import Heston, RoughBergomi, SimulationConfig
from mjollnir.processes.gbm import GBM

QUANTILES = {"p5": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95}
SCENARIO_FROM_DAYS = 63  # the policy's scenarios start beyond 3 months
DAYS_PER_YEAR = 252


def _historical_drift(model: str, parameters: dict[str, float], dt: float) -> float | None:
    """The fit's own annual spot drift, in the units the policy drift uses."""
    if model == "garch" and "mu" in parameters:
        long_run_var = parameters["omega"] / max(1.0 - parameters["alpha"] - parameters["beta"], 1e-6)
        return parameters["mu"] / dt + 0.5 * long_run_var / dt
    if model in ("gbm", "heston", "rough_bergomi", "merton_jump") and "mu" in parameters:
        return parameters["mu"]
    return None


def _simulate(model, parameters, s0, n_steps, n_paths, dt, rng, random_seed, drift):
    """(n_steps + 1, n_paths) spot paths; ``drift`` is the annual spot drift
    (None only for OU, which follows its own long-run level)."""
    if model == "gbm":
        process = GBM(mu=drift, sigma=parameters["sigma"])
        config = SimulationConfig(n_paths=n_paths, n_steps=n_steps, random_seed=random_seed)
        _, paths = process.simulate(np.array([s0]), n_steps * dt, config, scheme="exact")
        return paths[:, :, 0]

    if model == "ou":
        # The OU fit describes the log price: x = log S reverts to theta.
        kappa, theta, sigma = parameters["kappa"], parameters["theta"], parameters["sigma"]
        decay = np.exp(-kappa * dt)
        shock_sd = sigma * np.sqrt((1.0 - decay**2) / (2.0 * kappa))  # exact discretization
        log_s = np.empty((n_steps + 1, n_paths))
        log_s[0] = np.log(s0)
        for t in range(n_steps):
            log_s[t + 1] = theta + (log_s[t] - theta) * decay + shock_sd * rng.standard_normal(n_paths)
        return np.exp(log_s)

    if model == "heston":
        v0 = parameters.get("v0", parameters["theta"])
        process = Heston(mu=drift, kappa=parameters["kappa"], theta=parameters["theta"],
                         sigma_v=parameters["xi"], rho=parameters.get("rho", 0.0), v0=v0)
        config = SimulationConfig(n_paths=n_paths, n_steps=n_steps, random_seed=random_seed)
        _, paths = process.simulate(np.array([s0, v0]), n_steps * dt, config, scheme="milstein")
        return paths[:, :, 0]

    if model == "rough_bergomi":
        process = RoughBergomi(mu=drift, xi0=parameters["xi0"], eta=parameters["eta"],
                               rho=parameters["rho"], H=parameters["H"])
        config = SimulationConfig(n_paths=n_paths, n_steps=n_steps, random_seed=random_seed)
        _, paths = process.simulate(np.array([s0, parameters["xi0"]]), n_steps * dt, config)
        return paths[:, :, 0]

    if model == "merton_jump":
        sigma, lam = parameters["sigma"], parameters["lambda"]
        mu_j, sigma_j = parameters["mu_j"], parameters["sigma_j"]
        compensator = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        step_drift = (drift - 0.5 * sigma**2 - lam * compensator) * dt
        log_s = np.zeros((n_steps + 1, n_paths))
        log_s[0] = np.log(s0)
        for t in range(n_steps):
            n_jump = rng.poisson(lam * dt, size=n_paths)
            jump = n_jump * mu_j + np.sqrt(n_jump) * sigma_j * rng.standard_normal(n_paths)
            log_s[t + 1] = log_s[t] + step_drift + sigma * np.sqrt(dt) * rng.standard_normal(n_paths) + jump
        return np.exp(log_s)

    if model == "garch":
        omega, alpha, beta = parameters["omega"], parameters["alpha"], parameters["beta"]
        nu = parameters.get("nu")  # Student-t innovations when fitted (the Bayesian fit's)
        sigma2 = np.full(n_paths, omega / max(1.0 - alpha - beta, 1e-6))
        log_s = np.zeros((n_steps + 1, n_paths))
        log_s[0] = np.log(s0)
        for t in range(n_steps):
            # E[S_{t+1} / S_t] = exp(drift dt), as in mjollnir's GARCH world: the Gaussian log step's
            # mean offsets its variance; Student-t shocks compound as simple returns (floored at -99%)
            if nu is None:
                eps = np.sqrt(sigma2) * rng.standard_normal(n_paths)
                log_s[t + 1] = log_s[t] + drift * dt - 0.5 * sigma2 + eps
            else:
                eps = np.sqrt(sigma2 * (nu - 2.0) / nu) * rng.standard_t(nu, n_paths)
                log_s[t + 1] = log_s[t] + drift * dt + np.log(np.maximum(1.0 + eps, 0.01))
            sigma2 = np.maximum(omega + alpha * eps**2 + beta * sigma2, 1e-12)
        return np.exp(log_s)

    if model == "regime_switching_gbm":
        n_regimes = sum(1 for key in parameters if key.startswith("sigma_regime"))
        sigma = np.array([parameters[f"sigma_regime{k}"] for k in range(n_regimes)])
        transition = np.array([[parameters[f"p_{i}{j}"] for j in range(n_regimes)] for i in range(n_regimes)])
        if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("regime transition probabilities must sum to 1 in every row")
        cumulative = np.cumsum(transition, axis=1)
        log_s = np.zeros((n_steps + 1, n_paths))
        log_s[0] = np.log(s0)
        regime = np.zeros(n_paths, dtype=int)
        for t in range(n_steps):
            u = rng.random(n_paths)[:, None]
            regime = np.minimum((u > cumulative[regime]).sum(axis=1), n_regimes - 1)
            sd = sigma[regime]
            # the regimes keep their volatilities; the policy sets every regime's drift
            log_s[t + 1] = log_s[t] + (drift - 0.5 * sd**2) * dt + sd * np.sqrt(dt) * rng.standard_normal(n_paths)
        return np.exp(log_s)

    raise ValueError(f"Unsupported model for simulation: {model}")


def _price_decimals(s0: float) -> int:
    """Decimals that keep six significant figures of prices near ``s0``: enough to
    draw and read, and a response a third the size of full doubles."""
    return max(0, 6 - int(np.ceil(np.log10(s0))))


def simulate(model: str, parameters: dict[str, float], s0: float, n_steps: int, n_paths: int, dt: float,
             max_paths_return: int = 20, random_seed: int | None = None, risk_free_rate: float = 0.04,
             dividend_yield: float = 0.0, scenario_excess_drift: float = 0.05) -> dict:
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be >= 1")
    model = model.lower()
    rng = np.random.default_rng(random_seed)  # per call: the global generator is shared
    time_grid = np.linspace(0.0, n_steps * dt, n_steps + 1)

    policy_drift = None if model == "ou" else risk_free_rate - dividend_yield
    spot_paths = _simulate(model, parameters, s0, n_steps, n_paths, dt, rng, random_seed, policy_drift)

    quantiles = {name: np.percentile(spot_paths, q, axis=1) for name, q in QUANTILES.items()}
    scenarios = []
    if policy_drift is None:
        drift_note = ("OU follows its fitted long-run level, which is itself a statement about the past: "
                      "read this fan as the fitted model's, not as a forecast.")
    else:
        drift_note = (f"Spot drifts at r - q = {100 * policy_drift:.1f}% a year with no excess drift; "
                      "a stock's historical drift is too uncertain to forecast with.")
        horizon_days = n_steps * dt * DAYS_PER_YEAR
        median = quantiles["p50"]
        if horizon_days > SCENARIO_FROM_DAYS + 1e-9 and scenario_excess_drift > 0:
            for sign, kind in ((1, "positive"), (-1, "negative")):
                d = sign * scenario_excess_drift
                scenarios.append({"kind": kind, "label": f"{kind.capitalize()} excess drift ({100 * d:+.0f}% a year)",
                                  "excess_drift": d, "median": median * np.exp(d * time_grid)})
            drift_note += (f" Beyond 3 months the policy adds explicit scenarios of ±{100 * scenario_excess_drift:.0f}% "
                           "a year of excess drift.")
        historical = _historical_drift(model, parameters, dt)
        if historical is not None:
            d = historical - policy_drift
            scenarios.append({"kind": "historical",
                              "label": f"If the historical drift continued ({100 * historical:.0f}% a year; not a forecast)",
                              "excess_drift": d, "median": median * np.exp(d * time_grid)})

    final = spot_paths[-1]
    n_return = min(max_paths_return, n_paths)
    decimals = _price_decimals(s0)

    def prices(values: np.ndarray) -> list[float]:
        return np.round(values, decimals).tolist()

    for scenario in scenarios:
        scenario["median"] = prices(scenario["median"])
    return {
        "model": model,
        "n_steps": n_steps,
        "n_paths": n_paths,
        "dt": dt,
        "time_grid": time_grid.tolist(),
        "mean_path": prices(np.mean(spot_paths, axis=1)),
        "quantiles": {name: prices(values) for name, values in quantiles.items()},
        "sample_paths": [prices(spot_paths[:, i]) for i in range(n_return)],
        "stats": {
            "mean": float(np.mean(final)), "std": float(np.std(final)),
            "p5": float(np.percentile(final, 5)), "p50": float(np.percentile(final, 50)),
            "p95": float(np.percentile(final, 95)), "min": float(np.min(final)), "max": float(np.max(final)),
        },
        "spot_drift": policy_drift,
        "drift_note": drift_note,
        "scenarios": scenarios,
    }
