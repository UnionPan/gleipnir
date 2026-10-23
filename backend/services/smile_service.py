"""Volatility smile: market implied vols vs. the smiles that pricing models imply.

Model vols come from mjollnir: COS prices inverted on each strike's
out-of-the-money side for Heston and Merton (backend.services.model_vols), and
Hagan's formula for SABR. Strikes where a model vol is undefined are reported as
null: a model IV is never replaced by the market IV.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from mjollnir.processes.sabr import SABR

from backend.schemas.smile import VolSmileComparisonResponse, VolSmileDataPoint, VolSmileRequest
from backend.services.model_vols import finite_or_none, heston_ivs, merton_ivs

RISK_FREE_RATE = 0.05  # simplified flat rate
MIN_TIME_TO_EXPIRY = 1 / 365.25


def _model_ivs(model: str, request: VolSmileRequest, spot: float, strikes: np.ndarray,
               t: float, r: float, atm_iv: float) -> np.ndarray:
    """Implied vols the model assigns to each strike; NaN where undefined."""
    n = len(strikes)
    if model == "black_scholes":
        # One volatility for every strike: the at-the-money market IV.
        return np.full(n, atm_iv)
    if model == "sabr":
        forward = spot * np.exp(r * t)
        # The API takes the ATM lognormal vol; Hagan's alpha carries price units
        # F^(1-beta), so convert at leading order. mjollnir's SABR names Hagan's
        # vol-of-vol `alpha` and the initial vol `sigma0`.
        hagan_alpha = request.sabr_alpha * forward ** (1.0 - request.sabr_beta)
        sabr = SABR(beta=request.sabr_beta, alpha=request.sabr_nu, rho=request.sabr_rho,
                    sigma0=hagan_alpha)
        return np.array([sabr.implied_volatility_hagan(forward, k, t) for k in strikes])
    if model == "heston":
        return heston_ivs(spot, strikes, t, r, 0.0, v0=request.heston_v0, kappa=request.heston_kappa,
                          theta=request.heston_theta, sigma_v=request.heston_sigma_v,
                          rho=request.heston_rho)
    if model == "merton":
        sigma = request.merton_sigma if request.merton_sigma is not None else float(np.sqrt(request.heston_v0))
        return merton_ivs(spot, strikes, t, r, 0.0, sigma=sigma, lambda_j=request.merton_lambda,
                          mu_j=request.merton_mu_j, sigma_j=request.merton_sigma_j)
    raise ValueError(f"unsupported model {model!r}")


class SmileService:
    """Service for volatility smile analysis with model comparison."""

    @staticmethod
    def get_volatility_smile_comparison(request: VolSmileRequest) -> VolSmileComparisonResponse:
        """Market call IVs for one expiry, with each requested model's IVs at the same strikes."""
        ticker = yf.Ticker(request.symbol)
        info = ticker.info
        spot = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
        if spot <= 0:
            raise ValueError(f"Could not fetch spot price for {request.symbol}")
        if request.expiration_date not in ticker.options:
            raise ValueError(f"Expiration {request.expiration_date} not available for {request.symbol}")

        expiry = datetime.strptime(request.expiration_date, "%Y-%m-%d")
        t = max((expiry - datetime.now()).days / 365.25, MIN_TIME_TO_EXPIRY)

        calls = ticker.option_chain(request.expiration_date).calls
        calls = calls[pd.notna(calls["impliedVolatility"]) & (calls["impliedVolatility"] > 0)
                      & pd.notna(calls["lastPrice"])]
        if calls.empty:
            raise ValueError(f"No usable call quotes for {request.symbol} {request.expiration_date}")
        strikes = calls["strike"].to_numpy(dtype=float)
        market_ivs = calls["impliedVolatility"].to_numpy(dtype=float)
        atm_iv = float(market_ivs[np.argmin(np.abs(strikes - spot))])

        model_ivs = {m: finite_or_none(_model_ivs(m, request, spot, strikes, t, RISK_FREE_RATE, atm_iv))
                     for m in request.models}

        order = np.argsort(strikes)
        data_points: list[VolSmileDataPoint] = [
            VolSmileDataPoint(
                strike=float(strikes[i]),
                moneyness=float(strikes[i] / spot),
                market_iv=float(market_ivs[i]),
                calculated_ivs={m: ivs[i] for m, ivs in model_ivs.items()},
            )
            for i in order
        ]
        return VolSmileComparisonResponse(
            symbol=request.symbol,
            expiration_date=request.expiration_date,
            time_to_expiry=t,
            spot_price=spot,
            risk_free_rate=RISK_FREE_RATE,
            data_points=data_points,
            models_used=request.models,
        )
