"""Bayesian P-measure fits: a posterior over each model's dynamics, not a point estimate.

For Heston, Merton, regime switching, GARCH and rough Bergomi the posterior
comes from a neural network trained offline (mjollnir.inference): it read
millions of histories simulated from the lab's own worlds and learned, for
any history's summaries, the distribution of the parameters that could have
produced it. A fit here is one forward pass, so it costs the same for every
visitor and never iterates on their data.

Every posterior carries what makes it trustworthy or not:

- its mean, standard deviation and 90% credible interval per parameter;
- per inferred coordinate, how much the history narrowed the prior
  (contraction: near 0, the interval is the prior's) and the offline
  coverage of the network's 50% and 90% intervals on fresh simulations;
- the prior: a sector's, learned from the S&P 500 by empirical Bayes, when
  the ticker belongs to it, otherwise the broad training prior;
- the share of the network's draws inside the range it was trained on;
- a posterior predictive check: histories simulated from posterior draws,
  against which each aspect of the real history (its tails, its volatility
  clustering, ...) either fits or is something the model cannot produce.

GBM needs no network: under the Jeffreys prior the posterior of sigma is
exact (a scaled inverse chi distribution).

Today's state (Heston's v0, rough Bergomi's xi0, GARCH's variance, the
regime) is not inferred: it is read off the end of the history, as in the
classical fits, given the posterior-mean dynamics.
"""

from __future__ import annotations

import json
import math
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from mjollnir.calibration.physical.garch_calibrator import garch_variances
from mjollnir.inference import GaussianPrior, load_artifact
from mjollnir.inference import fit as npe_fit
from mjollnir.inference.check import NOT_CHECKED, predictive_check, predictive_p_values
from mjollnir.inference.summaries import FEATURE_NAMES, pad_returns, summary_features
from scipy import special, stats

DT = 1.0 / 252
NPE_DIR = Path(__file__).resolve().parents[1] / "data" / "npe"
# The networks trained on 250 to 1260 days; a calendar year of closes gives 248 to 252 returns, a
# shortfall the length summary extrapolates over without strain.
MIN_DAYS = 240
MAX_DAYS = 1260  # the networks read the last five years
N_DRAWS = 4096
STATE_WINDOW = 21
WEAKLY_IDENTIFIED = 0.3   # contraction below this: the interval is mostly the prior's
LOW_IN_PRIOR_SHARE = 0.5
N_REPLICATES = 64          # posterior predictive histories per check

# What each summary measures, and how to show it (the summaries are logs where noted).
_LAGS = {"acf": (1, 2, 5, 10, 21, 63), "abs": (1, 5, 21, 63), "variogram": (1, 2, 5, 10, 21)}


def _days(k: int) -> str:
    return f"{k} day" if k == 1 else f"{k} days"

SUMMARY_LABELS: dict[str, tuple[str, str]] = {
    "log_vol": ("Volatility level", "annualized volatility"),
    "skew": ("Skewness", "skewness of daily returns"),
    "log_kurtosis": ("Fat tails", "kurtosis of daily returns"),
    "tail_3sd": ("Fat tails", "share of days beyond 3 standard deviations"),
    "tail_5sd": ("Fat tails", "share of days beyond 5 standard deviations"),
    "max_abs_over_sd": ("Fat tails", "largest daily move, in standard deviations"),
    **{f"acf_sq_{k}": ("Volatility clustering", f"autocorrelation of squared returns at {_days(k)}")
       for k in _LAGS["acf"]},
    **{f"acf_abs_{k}": ("Volatility clustering", f"autocorrelation of absolute returns at {_days(k)}")
       for k in _LAGS["abs"]},
    "leverage_1": ("Leverage", "correlation of a return with the next day's squared return"),
    "leverage_5": ("Leverage", "correlation of a return with the next 5 days' realized variance"),
    "leverage_21": ("Leverage", "correlation of a return with the next 21 days' realized variance"),
    "log_rv21_sd": ("Volatility of volatility", "spread of the log 21-day realized variance"),
    "log_rv5_sd": ("Volatility of volatility", "spread of the log 5-day realized variance"),
    "rv21_q90_over_q10": ("Volatility of volatility", "log ratio of high to low 21-day realized variance"),
    **{f"variogram_{k}": ("Roughness", f"variogram of log absolute returns at {_days(k)}") for k in _LAGS["variogram"]},
}
EXPONENTIATED = {"log_vol", "log_kurtosis", "max_abs_over_sd"}
ASPECTS = tuple(dict.fromkeys(label for label, _ in SUMMARY_LABELS.values()))

NPE_NAME = {"heston": "heston", "merton_jump": "merton", "regime_switching_gbm": "regime_switching_gbm",
            "garch": "garch", "rough_bergomi": "rough_bergomi"}
# the network's coordinates, in the names the receipt shows
IDENTIFICATION_NAMES = {
    "heston": {"kappa": "kappa", "theta": "theta", "sigma_v": "xi", "rho": "rho"},
    "merton_jump": {"sigma": "sigma", "lambda_j": "lambda", "mu_j": "mu_j", "sigma_j": "sigma_j"},
    "regime_switching_gbm": {"sigma_0": "sigma_regime0", "sigma_1": "sigma_regime1", "p_00": "p_00", "p_11": "p_11"},
    "garch": {"long_run_vol": "long_run_vol", "persistence": "persistence", "alpha_share": "alpha_share",
              "nu": "nu"},
    "rough_bergomi": {"H": "H", "eta": "eta", "rho": "rho", "xi0": "window_variance"},
}
METHOD_LABEL = "Amortized neural posterior (mixture density network), {prior}"


@cache
def artifact(model: str):
    return load_artifact(NPE_DIR / NPE_NAME[model])


@lru_cache(maxsize=1)
def _sector_priors() -> dict:
    path = NPE_DIR / "sector_priors.json"
    return json.loads(path.read_text()) if path.exists() else {"tickers": {}, "models": {}}


def prior_for(model: str, ticker: str, choice: str) -> tuple[GaussianPrior | None, str, str]:
    """(prior, kind, label): the ticker's sector prior when asked for and available."""
    if choice == "sector":
        table = _sector_priors()
        sector = table["tickers"].get(ticker.upper())
        entry = table["models"].get(NPE_NAME[model], {}).get(sector) if sector else None
        if entry is not None:
            label = f"{sector} sector prior ({entry['n']} {table['universe']} stocks)"
            return GaussianPrior(np.asarray(entry["mean"]), np.asarray(entry["cov"]), label), "sector", label
        label = (f"broad training prior ({ticker.upper()} is not in the {table.get('universe', 'S&P 500')} "
                 "universe the sector priors come from)")
        return None, "broad", label
    return None, "broad", "broad training prior"


def _summary(draws: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {k: {"mean": float(v.mean()), "sd": float(v.std()), "q05": float(np.quantile(v, 0.05)),
                "q95": float(np.quantile(v, 0.95))} for k, v in draws.items()}


def _api_draws(model: str, x: np.ndarray) -> dict[str, np.ndarray]:
    """Natural posterior draws (n, d) as draws of the API's parameters."""
    if model == "heston":
        return {"kappa": x[:, 0], "theta": x[:, 1], "xi": x[:, 2], "rho": x[:, 3]}
    if model == "merton_jump":
        return {"sigma": x[:, 0], "lambda": x[:, 1], "mu_j": x[:, 2], "sigma_j": x[:, 3]}
    if model == "regime_switching_gbm":
        return {"sigma_regime0": x[:, 0], "sigma_regime1": x[:, 1], "p_00": x[:, 2], "p_01": 1.0 - x[:, 2],
                "p_10": 1.0 - x[:, 3], "p_11": x[:, 3]}
    if model == "garch":
        long_run_var = x[:, 0] ** 2 * DT
        return {"omega": long_run_var * (1.0 - x[:, 1]), "alpha": x[:, 2] * x[:, 1],
                "beta": (1.0 - x[:, 2]) * x[:, 1], "nu": x[:, 3]}
    if model == "rough_bergomi":
        return {"H": x[:, 0], "eta": x[:, 1], "rho": x[:, 2]}
    raise ValueError(f"{model} has no posterior network")


def _turbulent_probability(returns: np.ndarray, sigmas: np.ndarray, transition: np.ndarray) -> float:
    """Filtered probability of the turbulent regime on the last day (Hamilton filter)."""
    r = returns - returns.mean()
    sd = sigmas * math.sqrt(DT)
    stay = np.diag(transition)
    p = (1.0 - stay[::-1]) / np.sum(1.0 - stay)  # the stationary distribution
    for x in r:
        p = (p @ transition) * np.exp(-0.5 * (x / sd) ** 2) / sd
        p /= p.sum()
    return float(p[1])


def _check_by_aspect(names, observed, quantiles, p_values, n_replicates: int) -> dict:
    """The predictive check per aspect of the history: its most extreme summary, and whether the
    history lies beyond every replicate (the model does not produce histories like it)."""
    rows = []
    for aspect in ASPECTS:
        idx = [i for i, n in enumerate(names) if SUMMARY_LABELS[n][0] == aspect]
        tails = np.minimum(p_values[idx], 1.0 - p_values[idx])
        worst = idx[int(np.argmin(tails))]
        p = float(p_values[worst])
        show = np.exp if names[worst] in EXPONENTIATED else float
        rows.append({
            "aspect": aspect,
            "verdict": "more" if p >= 1.0 else "less" if p <= 0.0 else "fits",
            "summary": SUMMARY_LABELS[names[worst]][1],
            "observed": float(show(observed[worst])),
            "q05": float(show(quantiles[0, worst])), "q50": float(show(quantiles[1, worst])),
            "q95": float(show(quantiles[2, worst])), "p_value": p,
        })
    return {"n_replicates": n_replicates, "aspects": rows}


def _misfit_note(check: dict) -> str | None:
    misses = [a["aspect"].lower() for a in check["aspects"] if a["verdict"] != "fits"]
    if not misses:
        return None
    listed = misses[0] if len(misses) == 1 else f"{', '.join(misses[:-1])} and {misses[-1]}"
    return (f"None of {check['n_replicates']} histories simulated from this posterior matches the stock's "
            f"{listed}: the model leaves {'it' if len(misses) == 1 else 'them'} out.")


def _identified_note(identification: dict[str, dict]) -> str:
    weak = [k for k, v in identification.items() if v["contraction"] is not None and v["contraction"] < WEAKLY_IDENTIFIED]
    if not weak:
        return "No parameter is left mostly to the prior."
    return (f"The history says little about {', '.join(weak)}: "
            f"{'its interval is' if len(weak) == 1 else 'their intervals are'} mostly the prior's.")


def fit(model: str, ticker: str, prices: np.ndarray, returns: np.ndarray, prior_choice: str) -> tuple[dict, dict]:
    """(parameters, diagnostics) of a Bayesian fit; parameters are posterior means."""
    if returns.size < MIN_DAYS:
        raise ValueError(f"a Bayesian fit needs at least {MIN_DAYS} daily returns (about a year); "
                         f"this window has {returns.size}")
    if model == "gbm":
        return _gbm(returns)
    trained = artifact(model)
    prior, prior_kind, prior_label = prior_for(model, ticker, prior_choice)
    posterior = npe_fit(trained, prices, n_draws=N_DRAWS, seed=0, prior=prior)
    draws = _api_draws(model, posterior.draws)
    summary = _summary(draws)
    parameters = {k: v["mean"] for k, v in summary.items()}
    validation = trained.validation
    identification = {
        shown: {"contraction": float(posterior.contraction[name]),
                "coverage_50": float(validation["coverage_50"][name]),
                "coverage_90": float(validation["coverage_90"][name])}
        for name, shown in IDENTIFICATION_NAMES[model].items()
    }
    recent = returns[-MAX_DAYS:]
    variance_today = float(np.mean(returns[-STATE_WINDOW:] ** 2) / DT)
    derived: dict[str, float] = {}
    notes: list[str] = []

    if model == "heston":
        parameters["v0"] = variance_today
        derived = {"long_run_vol": math.sqrt(parameters["theta"]), "vol_today": math.sqrt(variance_today),
                   "half_life_days": math.log(2.0) / parameters["kappa"] / DT,
                   "feller_ratio": 2 * parameters["kappa"] * parameters["theta"] / parameters["xi"] ** 2}
        notes.append("v0 is the trailing 21-day realized variance.")
    elif model == "merton_jump":
        jump_var = draws["lambda"] * (draws["mu_j"] ** 2 + draws["sigma_j"] ** 2)
        share = jump_var / (draws["sigma"] ** 2 + jump_var)
        derived = {"jump_variance_share": float(share.mean()),
                   "total_vol": float(np.sqrt(draws["sigma"] ** 2 + jump_var).mean())}
        notes.append(f"Jumps carry {100 * share.mean():.0f}% of the return variance "
                     f"(90% interval {100 * np.quantile(share, 0.05):.0f}-{100 * np.quantile(share, 0.95):.0f}%).")
    elif model == "regime_switching_gbm":
        sigmas = np.array([parameters["sigma_regime0"], parameters["sigma_regime1"]])
        transition = np.array([[parameters["p_00"], parameters["p_01"]], [parameters["p_10"], parameters["p_11"]]])
        turbulent = _turbulent_probability(recent, sigmas, transition)
        derived = {"regime_today": float(turbulent >= 0.5), "turbulent_probability_today": turbulent,
                   "calm_duration_days": 1.0 / max(parameters["p_01"], 1e-12),
                   "turbulent_duration_days": 1.0 / max(parameters["p_10"], 1e-12)}
        notes.append("Regime 0 is calm and regime 1 turbulent, told apart by volatility alone; both drift at "
                     "r - q in simulations. Today's regime is the filtered likelier one.")
    elif model == "garch":
        persistence = parameters["alpha"] + parameters["beta"]
        mu = float(recent.mean())
        h = garch_variances(np.array([mu, parameters["omega"], parameters["alpha"], parameters["beta"]]), recent,
                            float(np.var(recent, ddof=1)))
        h_next = parameters["omega"] + parameters["alpha"] * (recent[-1] - mu) ** 2 + parameters["beta"] * h[-1]
        derived = {"long_run_vol": math.sqrt(parameters["omega"] / (1.0 - persistence) / DT),
                   "vol_today": math.sqrt(h_next / DT), "persistence": persistence,
                   "half_life_days": math.log(0.5) / math.log(persistence)}
        notes.append("Parameters are per day.")
    elif model == "rough_bergomi":
        parameters["xi0"] = variance_today
        window = posterior.draws[:, 3]
        derived = {"vol_today": math.sqrt(variance_today), "window_vol": float(np.sqrt(window).mean())}
        summary["window_variance"] = _summary({"window_variance": window})["window_variance"]
        notes.append("xi0 is the trailing 21-day realized variance; the posterior's variance level is the "
                     "window's average.")

    notes.append(_identified_note(identification))
    replicated = predictive_check(trained, prices, posterior, n_replicates=N_REPLICATES, seed=0)
    check = _check_by_aspect(replicated.names, replicated.observed, replicated.replicate_quantiles,
                             replicated.p_values, replicated.n_replicates)
    if (misfit := _misfit_note(check)) is not None:
        notes.append(misfit)
    if posterior.in_prior_share < LOW_IN_PRIOR_SHARE:
        notes.append(f"Only {100 * posterior.in_prior_share:.0f}% of the network's draws fell inside the range it "
                     "was trained on: this history is unusual for the model, so read the posterior with care.")
    if returns.size > MAX_DAYS:
        notes.append("The network reads the last five years of the window.")
    training = trained.training
    return parameters, {
        "method_label": METHOD_LABEL.format(prior=prior_label),
        "std_errors": {k: v["sd"] for k, v in summary.items() if k in parameters},
        "derived": derived,
        "converged": True,
        "note": " ".join(notes),
        "posterior": {
            "kind": "neural",
            "parameters": summary,
            "identification": identification,
            "prior": prior_label,
            "prior_kind": prior_kind,
            "draws": N_DRAWS,
            "effective_samples": posterior.effective_samples,
            "in_prior_share": posterior.in_prior_share,
            "n_days": posterior.n_days,
            "network": {"components": trained.n_components, "hidden": list(trained.hidden),
                        "trained_on": training.get("n_train"), "validated_on": validation.get("n"),
                        "sha256": trained.weights_sha256, "created_at": trained.created_at},
        },
        "predictive_check": check,
    }


def _gbm(returns: np.ndarray) -> tuple[dict, dict]:
    """Exact posterior of sigma under the Jeffreys prior p(m, s) ∝ 1/s:
    (n - 1) S^2 / s^2 ~ chi^2_{n-1}, with S^2 the sample variance."""
    nu = returns.size - 1
    scale = math.sqrt(nu * float(np.var(returns, ddof=1)) / DT)  # sigma = scale / sqrt(chi^2_nu)
    mean = scale * math.exp(special.gammaln((nu - 1) / 2) - special.gammaln(nu / 2)) / math.sqrt(2.0)
    second = scale**2 / (nu - 2)
    sd = math.sqrt(max(second - mean**2, 0.0))
    q05 = scale / math.sqrt(stats.chi2.ppf(0.95, nu))
    q95 = scale / math.sqrt(stats.chi2.ppf(0.05, nu))
    check = _gbm_check(returns, scale, nu)
    notes = ["With constant volatility the posterior is available in closed form; no network is needed. "
             "Its 90% interval is exact."]
    if (misfit := _misfit_note(check)) is not None:
        notes.append(misfit)
    return {"sigma": mean}, {
        "method_label": "Exact posterior (Jeffreys prior)",
        "std_errors": {"sigma": sd},
        "converged": True,
        "note": " ".join(notes),
        "posterior": {
            "kind": "exact",
            "parameters": {"sigma": {"mean": mean, "sd": sd, "q05": q05, "q95": q95}},
            "identification": {},
            "prior": "Jeffreys prior",
            "prior_kind": "jeffreys",
            "n_days": int(returns.size),
        },
        "predictive_check": check,
    }


def _gbm_check(returns: np.ndarray, scale: float, nu: int, seed: int = 0) -> dict:
    """Predictive check of GBM: i.i.d. Gaussian histories at posterior draws of sigma."""
    rng = np.random.default_rng(seed)
    recent = returns[-MAX_DAYS:]
    sigmas = scale / np.sqrt(rng.chisquare(nu, N_REPLICATES))
    simulated = sigmas[:, None] * math.sqrt(DT) * rng.standard_normal((N_REPLICATES, recent.size))
    padded, mask = pad_returns([recent, *simulated], MAX_DAYS)
    features = np.asarray(summary_features(jnp.asarray(padded), jnp.asarray(mask)))
    keep = np.array([n not in NOT_CHECKED for n in FEATURE_NAMES])
    observed, replicates = features[0, keep], features[1:, keep]
    names = tuple(n for n in FEATURE_NAMES if n not in NOT_CHECKED)
    return _check_by_aspect(names, observed, np.quantile(replicates, [0.05, 0.5, 0.95], axis=0),
                            predictive_p_values(observed, replicates), N_REPLICATES)


def warm_up() -> None:
    """Load every network and compile one forward pass."""
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.015, MIN_DAYS + 1)))
    returns = np.diff(np.log(prices))
    for model in ("gbm", *NPE_NAME):
        fit(model, "WARMUP", prices, returns, "broad")
