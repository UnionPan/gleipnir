"""
JAX-native option pricing for the deep hedging environment.

The grid's options are listed contracts (see :func:`contract_schedule`):
strikes fixed at issue, tenors running down to expiry. Every step prices all
of them with the COS method, staying entirely in JAX arrays (no to_numpy()
calls) so the result can feed directly into a ``lax.scan`` rollout body.

The grid is pre-compiled once via ``compile_padded_grid()`` into fixed-shape
arrays (padded to max_strikes_per_maturity) so that ``vmap`` over maturities
and the enclosing ``lax.scan`` both see static shapes.

author: Yunian Pan
email: yp1170@nyu.edu
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

# Public mjollnir.jax API (kernel extracted from options_desk; old private
# names kept as local aliases to minimize the diff).
from mjollnir.jax import (
    HestonCFParams,
    configure_runtime as configure_jax_runtime,
)

from .env import FloatingOptionGrid

configure_jax_runtime()


# ============================================================================
# Market parameter container
# ============================================================================

class HestonMarketParams(NamedTuple):
    """Risk-neutral Heston parameters for option pricing within the env."""
    kappa: float
    theta: float
    sigma_v: float
    rho: float
    r: float       # risk-free rate
    q: float       # dividend yield


# ============================================================================
# Padded grid for JIT-compatible fixed-shape arrays
# ============================================================================

class _PaddedGrid(NamedTuple):
    """Pre-compiled fixed-shape arrays for JIT-compatible grid pricing.

    All arrays have the maturity dimension as axis 0 (length = n_maturities)
    and the strike/moneyness dimension padded to ``max_strikes`` (axis 1).
    Invalid entries are masked with ``moneyness_mask``.

    Instrument layout in the env price vector:
        [underlying, call_m0_k0, put_m0_k0, call_m0_k1, put_m0_k1, ...]
    So maturity ``m`` with ``K`` strikes occupies 2*K slots starting at
    ``instrument_offset[m]``.
    """
    maturities_steps: jnp.ndarray     # (n_maturities,) int — maturity in env steps
    maturities_years: jnp.ndarray     # (n_maturities,) float — maturity in years
    moneyness: jnp.ndarray            # (n_maturities, max_strikes) — padded moneyness
    moneyness_mask: jnp.ndarray       # (n_maturities, max_strikes) bool — True = valid
    instrument_offset: jnp.ndarray    # (n_maturities,) int — start index in price vector
    n_instruments: int                # total instruments (1 + 2 * total_strikes)
    max_strikes: int                  # max strikes per maturity (padding width)
    n_maturities: int

    # Flat instrument table (scatter-free pricing) — see compile_padded_grid
    # NOTE: bool arrays are stored as float32 (1.0/0.0) because the MPS
    # backend corrupts bool dtype inside JIT closures.
    flat_maturity_idx: jnp.ndarray    # (n_options,) int — maturity index for each option
    flat_moneyness: jnp.ndarray       # (n_options,) — moneyness for each option
    flat_is_call: jnp.ndarray         # (n_options,) float32 — 1.0 for call, 0.0 for put
    flat_valid: jnp.ndarray           # (n_options,) float32 — 1.0 for real instrument

    # Pre-built action masks — depends only on (grid, horizon), not state
    # Shape: (horizon+1, n_instruments) bool.  Built once by compile_padded_grid.
    action_masks: np.ndarray | None   # None when horizon unknown at compile time


def compile_padded_grid(
    grid: FloatingOptionGrid,
    dt: float,
    horizon_steps: int | None = None,
) -> _PaddedGrid:
    """Convert a ``FloatingOptionGrid`` into padded arrays for JIT pricing.

    Called once at environment construction time — not inside a scan body.

    Also builds a flat instrument table for scatter-free pricing: each of
    the ``n_options`` option slots maps to (maturity_index, moneyness, is_call).

    If ``horizon_steps`` is provided, pre-computes action masks for the
    entire horizon so they don't need to be rebuilt every simulation call.
    """
    maturities = list(grid.maturities)
    n_maturities = len(maturities)
    max_strikes = max(len(grid.moneyness_by_maturity[m]) for m in maturities)

    mat_steps = np.array(maturities, dtype=np.int32)
    mat_years = mat_steps.astype(np.float64) * dt

    moneyness_arr = np.zeros((n_maturities, max_strikes), dtype=np.float64)
    mask_arr = np.zeros((n_maturities, max_strikes), dtype=bool)

    offset = 1  # instrument 0 is the underlying
    offsets = np.zeros(n_maturities, dtype=np.int32)

    # Build flat instrument table (excluding underlying at index 0)
    # Store booleans as float32 — MPS corrupts bool in JIT closures
    n_options = grid.n_options  # 2 * total_strikes
    flat_mat_idx = np.zeros(n_options, dtype=np.int32)
    flat_money = np.zeros(n_options, dtype=np.float64)
    flat_call = np.zeros(n_options, dtype=np.float32)
    flat_valid = np.zeros(n_options, dtype=np.float32)

    flat_pos = 0
    for i, m in enumerate(maturities):
        strikes_m = list(grid.moneyness_by_maturity[m])
        n_k = len(strikes_m)
        moneyness_arr[i, :n_k] = strikes_m
        mask_arr[i, :n_k] = True
        offsets[i] = offset
        for k_idx in range(n_k):
            # call
            flat_mat_idx[flat_pos] = i
            flat_money[flat_pos] = strikes_m[k_idx]
            flat_call[flat_pos] = 1.0
            flat_valid[flat_pos] = 1.0
            flat_pos += 1
            # put
            flat_mat_idx[flat_pos] = i
            flat_money[flat_pos] = strikes_m[k_idx]
            flat_call[flat_pos] = 0.0
            flat_valid[flat_pos] = 1.0
            flat_pos += 1
        offset += 2 * n_k

    # Pre-compute action masks if horizon is known
    masks = None
    if horizon_steps is not None:
        from .env import build_action_mask
        masks = np.stack([
            build_action_mask(grid, t, horizon_steps)
            for t in range(horizon_steps + 1)
        ], axis=0)

    return _PaddedGrid(
        maturities_steps=jnp.array(mat_steps),
        maturities_years=jnp.array(mat_years),
        moneyness=jnp.array(moneyness_arr),
        moneyness_mask=jnp.array(mask_arr),
        instrument_offset=jnp.array(offsets),
        n_instruments=grid.n_instruments,
        max_strikes=max_strikes,
        n_maturities=n_maturities,
        flat_maturity_idx=jnp.array(flat_mat_idx),
        flat_moneyness=jnp.array(flat_money),
        flat_is_call=jnp.array(flat_call),
        flat_valid=jnp.array(flat_valid),
        action_masks=masks,
    )


# ============================================================================
# Listed contracts: fixed strikes, decaying tenors
# ============================================================================

def _enforce_vanilla_price_bounds(
    option_prices: jnp.ndarray,
    upper_bounds: jnp.ndarray,
) -> jnp.ndarray:
    """Sanitize non-finite COS output and enforce vanilla price bounds."""
    cleaned = jnp.nan_to_num(
        option_prices,
        nan=0.0,
        posinf=jnp.inf,
        neginf=0.0,
    )
    cleaned = jnp.maximum(cleaned, 0.0)
    return jnp.minimum(cleaned, upper_bounds)


def contract_schedule(maturities_steps: jnp.ndarray, time_index) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Where each maturity's contract is in its life at ``time_index``.

    A maturity of ``m`` steps lists one contract at a time: it is issued at a
    step with phase 0 (strikes set from that step's spot), trades for ``m``
    steps while its tenor runs down from ``m`` to 1, and at phase ``m`` it
    expires (tenor 0, priced at intrinsic value, not tradable); the next
    contract is issued on the following step. Holding a slot across steps is
    therefore holding one contract, whose price is a martingale under the
    pricing measure.

    Returns:
        (tau_steps, is_issue): ``(n_maturities,)`` remaining steps of the live
        contract (0 on its expiry step) and whether this step issues a new one.
    """
    period = maturities_steps + 1
    phase = jnp.mod(time_index, period)
    return maturities_steps - phase, phase == 0


def price_option_contracts(
    spot: jnp.ndarray,
    v0: jnp.ndarray,
    strikes: jnp.ndarray,
    tau_years: jnp.ndarray,
    padded_grid: _PaddedGrid,
    market: HestonMarketParams,
    N_cos: int = 128,
    L_cos: float = 12.0,
) -> jnp.ndarray:
    """Price the listed contracts at the current state.

    All options are priced in one broadcast over (n_options, N_cos): the
    Heston characteristic function is evaluated once per maturity at that
    maturity's remaining tenor, and each option gathers its maturity's
    coefficients. Contracts at expiry (tenor 0) are worth their intrinsic
    value.

    Args:
        spot: scalar JAX array, current spot
        v0: scalar JAX array, current instantaneous variance
        strikes: ``(n_options,)`` the contracts' strikes, fixed at issue
        tau_years: ``(n_maturities,)`` remaining tenor of each maturity's contract
        padded_grid: pre-compiled grid from ``compile_padded_grid``
        market: risk-neutral Heston parameters
        N_cos: COS expansion terms (static for JIT)
        L_cos: COS truncation, in standard deviations of the log return

    Returns:
        ``(n_instruments,)`` price vector; index 0 is the underlying (= spot),
        then the options in the grid's interleaved call/put order.
    """
    mu = market.r - market.q
    log_S0 = jnp.log(spot)

    # --- Step 1: per-maturity COS quantities ---
    expired = tau_years <= 0.0                                  # (n_mat,)
    T_all = jnp.where(expired, 1.0 / 252.0, tau_years)          # placeholder tenor where priced at intrinsic
    # The range is sized at the larger of today's and the long-run variance:
    # variance mean-reverts over the option's life, and a v0 near zero would
    # otherwise shrink the range below the density's support.
    v_range = jnp.maximum(v0, market.theta)
    c1 = log_S0 + (mu - 0.5 * v_range) * T_all
    c2 = jnp.maximum(v_range * T_all, 1e-12)
    a_all = c1 - L_cos * jnp.sqrt(c2)
    b_all = c1 + L_cos * jnp.sqrt(c2)
    ba_all = b_all - a_all

    k_grid = jnp.arange(N_cos)
    u_k_all = k_grid[None, :] * jnp.pi / ba_all[:, None]       # (n_mat, N_cos)

    cf_params = HestonCFParams(
        v0=v0, kappa=market.kappa, theta=market.theta,
        sigma_v=market.sigma_v, rho=market.rho, mu=mu,
    )
    d = jnp.sqrt(
        (cf_params.rho * cf_params.sigma_v * 1j * u_k_all - cf_params.kappa)**2
        + cf_params.sigma_v**2 * (1j * u_k_all + u_k_all**2)
    )
    g = (
        (cf_params.kappa - cf_params.rho * cf_params.sigma_v * 1j * u_k_all - d)
        / (cf_params.kappa - cf_params.rho * cf_params.sigma_v * 1j * u_k_all + d)
    )
    exp_dt = jnp.exp(-d * T_all[:, None])
    C = (
        cf_params.mu * 1j * u_k_all * T_all[:, None]
        + (cf_params.kappa * cf_params.theta / cf_params.sigma_v**2)
        * (
            (cf_params.kappa - cf_params.rho * cf_params.sigma_v * 1j * u_k_all - d)
            * T_all[:, None]
            - 2.0 * jnp.log((1.0 - g * exp_dt) / (1.0 - g))
        )
    )
    D = (
        (cf_params.kappa - cf_params.rho * cf_params.sigma_v * 1j * u_k_all - d)
        / cf_params.sigma_v**2
    ) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    cf_all = jnp.exp(C + D * v0 + 1j * u_k_all * log_S0)       # (n_mat, N_cos)

    # Fourier density coefficients: F_k = Re{cf * exp(-i*u_k*a)}
    F_k_all = jnp.real(cf_all * jnp.exp(-1j * u_k_all * a_all[:, None]))

    # --- Step 2: gather per-option quantities from the maturity tables ---
    mat_idx = padded_grid.flat_maturity_idx                     # (n_options,)
    is_call = padded_grid.flat_is_call                          # (n_options,) 1.0 call / 0.0 put
    T_opt = T_all[mat_idx]
    a_opt = a_all[mat_idx]
    b_opt = b_all[mat_idx]
    ba_opt = ba_all[mat_idx]
    F_k_opt = F_k_all[mat_idx]                                  # (n_options, N_cos)
    log_K = jnp.log(strikes)

    # --- Step 3: COS payoff coefficients (n_options, N_cos) ---
    w = k_grid[None, :] * jnp.pi / ba_opt[:, None]

    def _F_antideriv(x_col):
        """Antiderivative of exp(x)*cos(w*(x-a)). x_col: (n_options,)."""
        x = x_col[:, None]
        a = a_opt[:, None]
        return jnp.exp(x) / (1.0 + w**2) * (
            jnp.cos(w * (x - a)) + w * jnp.sin(w * (x - a))
        )

    # strikes outside the truncation range: clip the integration limits into it
    log_K_in = jnp.clip(log_K, a_opt, b_opt)
    safe_k = jnp.where(k_grid == 0, 1.0, k_grid)

    chi_call = _F_antideriv(b_opt) - _F_antideriv(log_K_in)
    psi_call = (ba_opt[:, None] / (safe_k[None, :] * jnp.pi)) * (
        jnp.sin(w * (b_opt[:, None] - a_opt[:, None]))
        - jnp.sin(w * (log_K_in[:, None] - a_opt[:, None]))
    )
    psi_call = jnp.where(k_grid[None, :] == 0, (b_opt - log_K_in)[:, None], psi_call)
    V_call = 2.0 / ba_opt[:, None] * (chi_call - strikes[:, None] * psi_call)

    chi_put = _F_antideriv(log_K_in) - _F_antideriv(a_opt)
    psi_put = (ba_opt[:, None] / (safe_k[None, :] * jnp.pi)) * jnp.sin(
        w * (log_K_in[:, None] - a_opt[:, None])
    )
    psi_put = jnp.where(k_grid[None, :] == 0, (log_K_in - a_opt)[:, None], psi_put)
    V_put = 2.0 / ba_opt[:, None] * (-chi_put + strikes[:, None] * psi_put)

    V_k = is_call[:, None] * V_call + (1.0 - is_call[:, None]) * V_put
    half_weight = jnp.where(k_grid == 0, 0.5, 1.0)

    # --- Step 4: price = exp(-rT) * sum(F_k * V_k * half_weight) ---
    option_prices = jnp.exp(-market.r * T_opt) * jnp.sum(
        F_k_opt * V_k * half_weight[None, :], axis=-1
    )
    call_upper = spot * jnp.exp(-market.q * T_opt)
    put_upper = strikes * jnp.exp(-market.r * T_opt)
    upper_bound = is_call * call_upper + (1.0 - is_call) * put_upper
    option_prices = _enforce_vanilla_price_bounds(option_prices, upper_bound)

    intrinsic = is_call * jnp.maximum(spot - strikes, 0.0) + (1.0 - is_call) * jnp.maximum(strikes - spot, 0.0)
    option_prices = jnp.where(expired[mat_idx], intrinsic, option_prices) * padded_grid.flat_valid

    return jnp.concatenate([spot[jnp.newaxis], option_prices])
