"""The deep-hedging experiment's market, shared by training and evaluation scripts.

A short at-the-money one-year call is hedged daily in a Heston world with no
rates or dividends (so the pricing and the real-world measure coincide), with
the underlying and a grid of listed 5- and 20-day options at 95%, 100% and
105% of the spot at issue. Trading costs: 1 bp of notional on the underlying,
1% on options.
"""

from __future__ import annotations

from .jax.env import GRID_COARSE, DeepHedgingEnvConfig, FloatingOptionGrid
from .jax.pricing import HestonMarketParams
from .utils.contracts import LiabilitySpec

HORIZON = 252
INITIAL_SPOT = 100.0
INITIAL_VARIANCE = 0.04
EVAL_SEED = 9999
EVAL_PATHS = 4096


def heston_easy() -> HestonMarketParams:
    """Feller-safe Heston (2 kappa theta = 0.16 > sigma_v^2 = 0.09), equity-like correlation."""
    return HestonMarketParams(kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0)


def make_env_config(grid: FloatingOptionGrid = GRID_COARSE, horizon: int = HORIZON) -> DeepHedgingEnvConfig:
    return DeepHedgingEnvConfig(
        horizon_steps=horizon,
        option_grid=grid,
        dt=1.0 / 252.0,
        transaction_cost_underlying=1.0e-4,
        transaction_cost_option=1.0e-2,
        risk_aversion=1_000.0,
        scheme="qe",  # positivity-preserving
    )


def short_call(horizon: int = HORIZON) -> LiabilitySpec:
    """The liability: a short at-the-money call expiring at the horizon."""
    return LiabilitySpec(kind="call", strike=INITIAL_SPOT, maturity=horizon, quantity=1.0)


def underlying_only_mask(grid: FloatingOptionGrid = GRID_COARSE) -> tuple[int, ...]:
    return (1,) + (0,) * (grid.n_instruments - 1)
