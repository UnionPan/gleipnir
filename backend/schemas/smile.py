"""Volatility smile schemas with model comparison."""
from pydantic import BaseModel, Field
from typing import Literal


class VolSmileRequest(BaseModel):
    """Request for volatility smile with model comparison."""
    symbol: str
    expiration_date: str
    models: list[Literal["black_scholes", "heston", "sabr", "merton"]] = Field(
        default=["black_scholes"],
        description="Pricing models to compare"
    )
    # Heston parameters
    heston_v0: float | None = Field(default=0.04, description="Heston initial variance")
    heston_theta: float | None = Field(default=0.04, description="Heston long-term variance")
    heston_kappa: float | None = Field(default=2.0, description="Heston mean reversion")
    heston_sigma_v: float | None = Field(default=0.3, description="Heston vol of vol")
    heston_rho: float | None = Field(default=-0.7, description="Heston correlation")

    # SABR parameters
    sabr_alpha: float | None = Field(
        default=0.2,
        description="SABR at-the-money lognormal vol (scale-free; converted to Hagan's alpha = vol * F^(1-beta))")
    sabr_beta: float | None = Field(default=0.7, description="SABR CEV parameter")
    sabr_rho: float | None = Field(default=-0.3, description="SABR correlation")
    sabr_nu: float | None = Field(default=0.4, description="SABR vol of vol")

    # Merton parameters
    merton_lambda: float | None = Field(default=0.1, description="Merton jump intensity")
    merton_mu_j: float | None = Field(default=-0.05, description="Merton mean jump size")
    merton_sigma_j: float | None = Field(default=0.15, description="Merton jump volatility")
    merton_sigma: float | None = Field(
        default=None, description="Merton diffusion vol; defaults to sqrt(heston_v0)")


class VolSmileDataPoint(BaseModel):
    """Single point on volatility smile."""
    strike: float
    moneyness: float
    market_iv: float
    calculated_ivs: dict[str, float | None] = Field(
        default_factory=dict,
        description="Model IV per model; null where the model price carries no vol information"
    )


class VolSmileComparisonResponse(BaseModel):
    """Volatility smile with model comparison."""
    symbol: str
    expiration_date: str
    time_to_expiry: float
    spot_price: float
    risk_free_rate: float
    data_points: list[VolSmileDataPoint]
    models_used: list[str]
