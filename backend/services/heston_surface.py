"""Heston or Bates calibrated to an option chain, at a cost no request can inflate.

Quotes: out-of-the-money mids (calls at or above the forward, puts below) with a
positive bid, 30 days (by default) to 2 years out, at most ``QUOTE_BUCKET`` of the
most liquid. Shorter expiries carry scheduled events (an earnings release
inside a week's options) that neither model has: fitted to them, Heston and
Bates alike run to extreme mean reversion and vol of vol. Weights are 1 / vega^2, so the least-squares fit is approximately one in
implied volatility. mjollnir's Levenberg-Marquardt fit runs in one compiled
program per model with fixed buckets, warmed at startup, and stops by itself.

Bates is Heston with lognormal jumps: on a real chain, pure Heston bends the
steep short-dated skew only with extreme mean reversion and vol of vol, and
the jumps let the diffusion parameters stay plausible.
"""

from __future__ import annotations

import logging
import time
from datetime import date

import numpy as np
from mjollnir.calibration.marketdata import OptionChain
from mjollnir.calibration.risk_neutral.gradient_heston import fit_bates_surface, fit_heston_surface
from mjollnir.jax import configure_runtime
from mjollnir.pricer._jax_iv import implied_vol_batch_np
from scipy.stats import norm

logger = logging.getLogger(__name__)

QUOTE_BUCKET = 256
EXPIRY_BUCKET = 32
MIN_DAYS = 30  # the default shortest expiry; a request may choose another
MAX_DAYS = 730
MIN_QUOTES = 8
DAYS_PER_YEAR = 365.0  # the chain's calendar convention


def _vega(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def chain_quotes(chain: OptionChain, min_days: int = MIN_DAYS) -> tuple[list[tuple[float, float, bool, float]],
                                                                      np.ndarray, dict]:
    """The quotes a fit uses, their weights, and a summary of what was kept."""
    S, r, q = chain.spot_price, chain.risk_free_rate, chain.dividend_yield
    rows = []
    for o in chain.options:
        days = (o.expiry - chain.reference_date).days
        if not min_days <= days <= MAX_DAYS or o.bid <= 0 or o.mid <= 0:
            continue
        T = days / DAYS_PER_YEAR
        forward = S * np.exp((r - q) * T)
        if o.is_call != (o.strike >= forward):  # keep the out-of-the-money side
            continue
        rows.append((o.strike, T, o.is_call, o.mid, (o.open_interest or 0) + (o.volume or 0)))
    if len(rows) < MIN_QUOTES:
        raise ValueError(f"only {len(rows)} usable out-of-the-money quotes; a Heston fit needs {MIN_QUOTES}")

    rows.sort(key=lambda row: -row[4])  # most liquid first
    kept: list[tuple] = []
    expiries: set[float] = set()
    for row in rows:
        if len(kept) == QUOTE_BUCKET:
            break
        if row[1] not in expiries and len(expiries) == EXPIRY_BUCKET:
            continue
        expiries.add(row[1])
        kept.append(row)

    arr = np.array([row[:4] for row in kept], float)
    n = len(arr)
    ivs = implied_vol_batch_np(np.full(n, S), arr[:, 0], arr[:, 1], np.full(n, r), np.full(n, q),
                               arr[:, 3], arr[:, 2] > 0.5)
    usable = np.isfinite(ivs)
    if usable.sum() < MIN_QUOTES:
        raise ValueError("too few quotes have an implied volatility to fit")
    arr, ivs = arr[usable], ivs[usable]
    vega = _vega(S, arr[:, 0], arr[:, 1], r, q, ivs)
    vega = np.maximum(vega, 0.01 * vega.max())  # deep wings: cap their weight
    quotes = [(float(k), float(t), bool(c > 0.5), float(m)) for k, t, c, m in arr]
    summary = {
        "n_quotes": len(quotes),
        "n_expiries": len(set(arr[:, 1])),
        "days_range": [round(float(arr[:, 1].min()) * DAYS_PER_YEAR), round(float(arr[:, 1].max()) * DAYS_PER_YEAR)],
        "moneyness_range": [float((arr[:, 0] / S).min()), float((arr[:, 0] / S).max())],
    }
    return quotes, 1.0 / vega**2, summary


FITS = {"heston": fit_heston_surface, "bates": fit_bates_surface}
# mjollnir's parameter names -> the API's
API_NAMES = {"sigma_v": "xi", "lambda_j": "lambda"}


def fit_chain(chain: OptionChain, model: str = "heston", min_days: int = MIN_DAYS) -> tuple[dict[str, float], dict]:
    """Heston or Bates parameters (API names) and diagnostics for an option chain."""
    if model not in FITS:
        raise ValueError(f"no chain fit for {model!r}: {sorted(FITS)}")
    configure_runtime()
    started = time.perf_counter()
    quotes, weights, summary = chain_quotes(chain, min_days)
    result = FITS[model](
        quotes, chain.spot_price, chain.risk_free_rate, chain.dividend_yield,
        weights=weights, asset=chain.underlying, buckets=(QUOTE_BUCKET, EXPIRY_BUCKET),
        source=f"Levenberg-Marquardt on exact COS gradients, {chain.underlying} chain of "
               f"{chain.reference_date.isoformat()}",
    )
    p = result.param_set.params
    parameters = {API_NAMES.get(k, k): v for k, v in p.items()}
    # JSON has no infinity: a parameter the chain cannot pin down reports null
    std = {API_NAMES.get(k, k): (v if np.isfinite(v) else None) for k, v in result.std_errors.items()}
    feller = 2.0 * p["kappa"] * p["theta"]
    note = (f"Fitted {summary['n_quotes']} out-of-the-money quotes over {summary['n_expiries']} expiries "
            f"({summary['days_range'][0]} to {summary['days_range'][1]} days), weighted to fit implied vols.")
    if p["kappa"] > 20.0 or p["sigma_v"] > 2.0:
        note += (" The fit reaches for extreme mean reversion and vol-of-vol: the shortest expiries' skew is steeper "
                 "than the model can make, usually because a scheduled event such as earnings falls inside them. "
                 "Leaving them out (a longer shortest expiry) gives calmer parameters.")
    if model == "bates":
        note += (f" Jumps: {p['lambda_j']:.2f} a year, a mean log size of {100 * p['mu_j']:+.1f}% "
                 f"with a spread of {100 * p['sigma_j']:.1f}%.")
    diagnostics = {
        "note": note,
        "method": "Levenberg-Marquardt on exact COS gradients, 1/vega^2 weights (an implied-vol fit)",
        "rmse_iv": result.iv_rmse,
        "rmse": result.rmse,
        "std_errors": std,
        "converged": result.converged,
        "params_hash": result.param_set.content_hash(),
        "computation_time": time.perf_counter() - started,
        "feller_satisfied": bool(feller > p["sigma_v"] ** 2),
        "feller_condition": f"2*kappa*theta = {feller:.4f}, xi^2 = {p['sigma_v'] ** 2:.4f}",
        "errorMetrics": {"rmse_iv": result.iv_rmse},
        **summary,
    }
    return parameters, diagnostics


def warm_up() -> None:
    """Compile the fit's one program on a synthetic chain."""
    from mjollnir.synthetic_data.data_provider import OptionQuote

    started = time.perf_counter()
    today = date(2026, 1, 2)
    options = []
    for days in (30, 91, 182):
        T = days / DAYS_PER_YEAR
        for k in np.linspace(80, 120, 9):
            is_call = k >= 100.0
            price = float(np.maximum(0.6, 0.4 * 100 * 0.25 * np.sqrt(T) * np.exp(-((k - 100) / 15) ** 2)))
            options.append(OptionQuote(strike=float(k), expiry=date.fromordinal(today.toordinal() + days),
                                       option_type="call" if is_call else "put", bid=price * 0.98,
                                       ask=price * 1.02, mid=price, last=price, volume=10, open_interest=100))
    chain = OptionChain(underlying="WARMUP", spot_price=100.0, reference_date=today,
                        risk_free_rate=0.04, dividend_yield=0.0, options=options)
    for model in FITS:
        fit_chain(chain, model)
    logger.info("Heston and Bates chain fits compiled in %.1f s", time.perf_counter() - started)
