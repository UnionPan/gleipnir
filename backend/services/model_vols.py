"""Model-implied volatilities on a strike grid, shared by the smile and surface services.

Each strike is priced on its out-of-the-money side (puts below the forward, calls
above), where the whole price is time value, and inverted with mjollnir's bisection
IV. A time value inside the COS pricer's error carries no volatility information
and comes back as NaN rather than as a vol fitted to round-off.
"""
import numpy as np
from mjollnir.jax import configure_runtime, fourier_price_batch, jax_cos_price_merton_multi
from mjollnir.pricer._jax_iv import implied_vol_batch_np

# COS with the default 256 terms prices to ~3e-12 of spot; stay well above it.
COS_PRICE_TOLERANCE = 1e-8


def _out_of_the_money_calls(spot: float, strikes: np.ndarray, t: float, r: float, q: float) -> np.ndarray:
    return strikes >= spot * np.exp((r - q) * t)


def _invert(spot, strikes, t, r, q, prices, is_call) -> np.ndarray:
    return implied_vol_batch_np(S=spot, K=strikes, T=t, r=r, q=q, market_prices=np.asarray(prices),
                                is_call=is_call, price_tolerance=COS_PRICE_TOLERANCE * spot)


def heston_ivs(spot: float, strikes: np.ndarray, t: float, r: float, q: float, *, v0: float,
               kappa: float, theta: float, sigma_v: float, rho: float) -> np.ndarray:
    configure_runtime()
    strikes = np.asarray(strikes, dtype=float)
    is_call = _out_of_the_money_calls(spot, strikes, t, r, q)
    prices = fourier_price_batch(spot, strikes, t, r, q, v0, kappa, theta, sigma_v, rho, is_call)
    return _invert(spot, strikes, t, r, q, prices, is_call)


def merton_ivs(spot: float, strikes: np.ndarray, t: float, r: float, q: float, *, sigma: float,
               lambda_j: float, mu_j: float, sigma_j: float) -> np.ndarray:
    configure_runtime()
    strikes = np.asarray(strikes, dtype=float)
    is_call = _out_of_the_money_calls(spot, strikes, t, r, q)
    prices = jax_cos_price_merton_multi(spot, strikes, t, r, q, sigma, lambda_j, mu_j, sigma_j, is_call)
    return _invert(spot, strikes, t, r, q, prices, is_call)


def finite_or_none(values: np.ndarray) -> list[float | None]:
    """JSON-ready row: undefined model vols become None (a gap), never a placeholder."""
    return [float(v) if np.isfinite(v) else None for v in values]
