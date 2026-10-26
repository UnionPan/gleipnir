"""Real-world (P-measure) model fits from daily closes, each with its uncertainty.

Every fit returns ``(parameters, diagnostics)`` in the API's names. Diagnostics
carry, besides the model's own fit statistics:

- ``std_errors`` per parameter: observed information for likelihood fits, a
  parametric bootstrap for moment estimators (with its ``bias``), the
  closed form where one exists;
- ``drift``: the historical drift as an educational estimate with its standard
  error and 95% interval (the drift policy never simulates it as a forecast);
- ``derived``: readable quantities (long-run volatility, half-lives, the jump
  share of variance);
- ``note``: what daily closes can and cannot identify for this model.

Costs are bounded by the capped history length and fixed bootstrap sizes.
Non-finite numbers become None (JSON has no infinity).
"""

from __future__ import annotations

import math

import numpy as np
from mjollnir.calibration.physical import (
    GARCHCalibrator,
    GBMCalibrator,
    HestonQMLECalibrator,
    MertonJumpCalibrator,
    OUCalibrator,
    RegimeSwitchingCalibrator,
    RoughBergomiVariogramCalibrator,
)

DT = 1.0 / 252
CI_Z = 1.959964
STATE_WINDOW = 21  # trailing days for "variance today", as in the lab
HESTON_BOOTSTRAP = 200
ROUGH_BOOTSTRAP = 50
MIN_REGIME_DAYS = 20  # below this, per-regime drift and vol are mostly noise
ANNUAL_DRIFT_FROM_BOOTSTRAP = {"heston", "rough_bergomi"}

# Canonical method per model; older names the frontend once sent map onto it.
METHODS = {
    "gbm": ("mle", {}),
    "ou": ("mle", {}),
    "heston": ("moments", {"particle_filter": "moments", "qmle": "moments"}),
    "rough_bergomi": ("variogram", {"moment_matching": "variogram", "particle_filter": "variogram"}),
    "regime_switching_gbm": ("threshold", {}),
    "merton_jump": ("mle", {}),
    "garch": ("mle", {}),
}


def canonical_method(model: str, method: str) -> str:
    if model not in METHODS:
        raise ValueError(f"Unsupported model: {model}")
    canonical, aliases = METHODS[model]
    resolved = aliases.get(method, method)
    if resolved != canonical:
        raise ValueError(f"{model} is fitted by '{canonical}', not '{method}'")
    return canonical


def finite(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _finite_dict(d: dict) -> dict:
    return {k: finite(v) for k, v in d.items()}


def historical_drift(returns: np.ndarray) -> dict:
    """Arithmetic drift of the price (annualized) with its standard error: the
    sample mean of daily log returns plus the Ito term, over the window."""
    years = returns.size * DT
    var = float(np.var(returns, ddof=1)) / DT
    estimate = float(np.mean(returns)) / DT + 0.5 * var
    std_error = math.sqrt(var / years)
    return {"estimate": estimate, "std_error": std_error,
            "ci_low": estimate - CI_Z * std_error, "ci_high": estimate + CI_Z * std_error}


def fit(model: str, method: str, prices, dates: list[str], include_drift: bool) -> tuple[dict, dict]:
    method = canonical_method(model, method)
    prices = np.asarray(prices, dtype=float)
    if prices.size < 60 or np.any(~np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("need at least 60 positive daily closes")
    returns = np.diff(np.log(prices))
    parameters, diagnostics = _FITTERS[model](prices, dates, returns)
    if not include_drift:
        parameters.pop("mu", None)
    diagnostics["drift"] = historical_drift(returns)
    if model in ANNUAL_DRIFT_FROM_BOOTSTRAP and "mu" in parameters:
        # A bootstrap simulates from today's variance, which can sit far from
        # the window's average; the drift's error is the window's, analytically.
        diagnostics["std_errors"]["mu"] = diagnostics["drift"]["std_error"]
        diagnostics.get("bias", {}).pop("mu", None)
    diagnostics.setdefault("errorMetrics", {})
    for key in ("std_errors", "bias", "derived"):
        if key in diagnostics:
            diagnostics[key] = _finite_dict(diagnostics[key])
    for key in ("logLikelihood", "aic", "bic"):
        if diagnostics.get(key) is not None:
            diagnostics[key] = finite(diagnostics[key])
    return {k: float(v) for k, v in parameters.items()}, diagnostics


def _gbm(prices, dates, returns):
    r = GBMCalibrator().fit(prices, DT)
    return ({"sigma": r.sigma, "mu": r.mu}, {
        "method_label": "Maximum likelihood (closed form)",
        "logLikelihood": r.log_likelihood, "aic": r.aic, "bic": r.bic,
        "std_errors": {"sigma": r.sigma_stderr, "mu": r.mu_stderr},
        "converged": True,
    })


def _ou(prices, dates, returns):
    """Mean reversion of the log price: a statistical description, not a
    pricing model (under Q the drift is r - q, whatever the history says)."""
    r = OUCalibrator().fit(np.log(prices), DT)
    half_life_days = math.log(2.0) / max(r.kappa, 1e-12) / DT
    return ({"kappa": r.kappa, "theta": r.theta, "sigma": r.sigma}, {
        "method_label": "Maximum likelihood on log prices",
        "logLikelihood": r.log_likelihood,
        "std_errors": {"kappa": r.kappa_stderr, "theta": r.theta_stderr, "sigma": r.sigma_stderr},
        "derived": {"long_run_level": math.exp(r.theta), "half_life_days": half_life_days,
                    "stationarity_pvalue": r.stationarity_test_pvalue},
        "converged": True,
        "note": (f"Fitted to log prices: the price reverts toward {math.exp(r.theta):,.2f} with a "
                 f"half-life of {half_life_days:,.0f} trading days. A stock's mean reversion is rarely "
                 "statistically real; the stationarity p-value and kappa's error say how much this one is."),
    })


def _heston(prices, dates, returns):
    calibrator = HestonQMLECalibrator(smooth_window=10)
    r = calibrator.fit(prices, DT)
    boot = calibrator.bootstrap(r, n_boot=HESTON_BOOTSTRAP, seed=0)
    v0 = float(np.mean(returns[-STATE_WINDOW:] ** 2) / DT)
    names = {"kappa": "kappa", "theta": "theta", "sigma_v": "xi", "rho": "rho", "mu": "mu"}
    return ({"kappa": r.kappa, "theta": r.theta, "xi": r.sigma_v, "rho": r.rho, "v0": v0, "mu": r.mu}, {
        "method_label": "Method of moments on a realized-variance proxy, parametric bootstrap",
        "logLikelihood": r.log_likelihood,
        "std_errors": {names[k]: v for k, v in boot.std_errors.items()},
        "bias": {names[k]: v for k, v in boot.bias.items()},
        "derived": {"long_run_vol": math.sqrt(r.theta), "vol_today": math.sqrt(v0),
                    "half_life_days": math.log(2.0) / max(r.kappa, 1e-12) / DT,
                    "feller_ratio": r.feller_ratio, "variance_proxy_r2": r.variance_proxy_r2},
        "converged": True,
        "note": ("Moments of a 10-day realized-variance proxy, so kappa and xi come out biased: the "
                 f"bootstrap ({boot.n_boot} simulated histories from this fit) measures how much. "
                 "Daily closes barely identify rho; equity option markets typically imply about -0.7. "
                 "v0 is the trailing 21-day realized variance."),
    })


def _rough_bergomi(prices, dates, returns):
    calibrator = RoughBergomiVariogramCalibrator()
    r = calibrator.fit(prices, DT)
    boot = calibrator.bootstrap(r, n_boot=ROUGH_BOOTSTRAP, seed=0)
    return ({"xi0": r.xi0, "eta": r.eta, "rho": r.rho, "H": r.H, "mu": r.mu}, {
        "method_label": "Noise-corrected log-|return| variogram, parametric bootstrap",
        "variogram_r2": r.fit_r2,
        "std_errors": boot.std_errors, "bias": boot.bias,
        "derived": {"vol_today": math.sqrt(r.xi0), "noise_floor": r.noise_floor},
        "converged": True,
        "note": ("H and eta come from how fast log|returns| decorrelate across lags, above a noise floor "
                 "(pi^2/4 for Gaussian shocks; fitted here). Daily closes barely identify roughness: "
                 "the bootstrap spread of H says how little. The rho proxy is strongly attenuated toward 0 "
                 "(in simulations a true -0.7 reads as about -0.1). xi0 is the trailing 21-day realized variance."),
    })


def _regime_switching(prices, dates, returns):
    """Bear, sideways and bull GBM regimes labelled by 20% drawdowns and rallies;
    numbered 0..n-1 in id order over the regimes that occur."""
    result = RegimeSwitchingCalibrator(dt=DT).fit_threshold(prices, np.asarray(dates, dtype="datetime64[D]"))
    ids = sorted(result.regime_params)
    parameters, std_errors = {}, {}
    for k, rid in enumerate(ids):
        regime = result.regime_params[rid]
        parameters[f"mu_regime{k}"] = regime.mu
        parameters[f"sigma_regime{k}"] = regime.sigma
        std_errors[f"mu_regime{k}"] = regime.mu_stderr
        std_errors[f"sigma_regime{k}"] = regime.sigma_stderr
        for j in range(len(ids)):
            parameters[f"p_{k}{j}"] = float(result.transition_matrix[k, j])
    legend = ", ".join(
        f"{k} = {result.regime_params[rid].regime_name} ({result.regime_params[rid].n_observations} days"
        + (", too few for a reliable estimate" if result.regime_params[rid].n_observations < MIN_REGIME_DAYS else "")
        + ")" for k, rid in enumerate(ids))
    return parameters, {
        "method_label": "Drawdown/rally thresholds, per-regime maximum likelihood",
        "std_errors": std_errors, "converged": True,
        "note": (f"Regimes {legend}. Days are labelled bear after a 20% drawdown from the 60-day high and "
                 "bull after a 20% rally off the 60-day low; regimes that never occur are omitted. A regime's "
                 "drift over a few weeks is mostly noise: read it with its error."),
    }


def _merton(prices, dates, returns):
    r = MertonJumpCalibrator().fit(prices, DT)
    std = {("lambda" if k == "lambda_" else k): v for k, v in r.std_errors.items()}
    note = (f"Jumps carry {100 * r.jump_variance_share:.0f}% of the return variance, the share the data "
            "identify best.")
    if r.lambda_ > 50:
        note += (f" {r.lambda_:.0f} jumps a year of about {100 * r.sigma_j:.1f}% each: the likelihood is "
                 "describing fat daily tails with frequent small jumps, not rare crashes.")
    if not r.converged:
        note += f" The optimizer did not converge ({r.message}); treat these values as unreliable."
    return ({"sigma": r.sigma, "lambda": r.lambda_, "mu_j": r.mu_j, "sigma_j": r.sigma_j, "mu": r.mu}, {
        "method_label": "Maximum likelihood (multi-start), observed information",
        "logLikelihood": r.log_likelihood, "aic": r.aic, "bic": r.bic,
        "std_errors": std, "converged": r.converged,
        "derived": {"jump_variance_share": r.jump_variance_share, "total_vol": r.total_vol},
        "note": note,
    })


def _garch(prices, dates, returns):
    r = GARCHCalibrator().fit(prices, DT)
    note = (f"Parameters are per day. Long-run volatility {100 * r.long_run_vol:.1f}% a year; a variance "
            f"shock halves in {r.half_life_days:.0f} trading days (persistence {r.persistence:.3f}).")
    if not r.converged:
        note += f" The optimizer did not converge ({r.message}); treat these values as unreliable."
    return ({"omega": r.omega, "alpha": r.alpha, "beta": r.beta, "mu": r.mu}, {
        "method_label": "Gaussian quasi-maximum likelihood, observed information",
        "logLikelihood": r.log_likelihood, "aic": r.aic, "bic": r.bic,
        "std_errors": r.std_errors, "converged": r.converged,
        "derived": {"long_run_vol": r.long_run_vol, "persistence": r.persistence,
                    "half_life_days": r.half_life_days},
        "note": note,
    })


_FITTERS = {
    "gbm": _gbm,
    "ou": _ou,
    "heston": _heston,
    "rough_bergomi": _rough_bergomi,
    "regime_switching_gbm": _regime_switching,
    "merton_jump": _merton,
    "garch": _garch,
}
