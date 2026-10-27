"""
Simulator alpha diagnostic.

Generates the same eval batch the LSTM/BS-delta evals use (seed 9999, 4096
paths, 252 steps, GRID_COARSE), then probes the simulator+pricer for
structural alpha that any agent could harvest as "free PnL":

    1. Spot martingale check:  E[S_T] − S_0 should be ≈ 0 under Q (r=q=0).
    2. Payoff consistency:     E[max(S_T − K, 0)] should match the
                               t=0 Heston-COS price of the liability.
    3. Per-contract hold:      E[P_expiry^i − P_0^i] for each of the 13
                               grid slots (1 underlying + 12 listed
                               options), each held to its own expiry.
                               Should be ≈ 0 in a self-consistent simulator.
    3b. Predictability:        today's variance must not forecast a held
                               contract's next move (the leak the old
                               rolling grid had: z ≈ 40).
    4. Do-nothing baseline:    confirms mean_pnl=0 for positions=0
                               (sanity check on the eval pipeline).

Standard errors reported on every figure so we can distinguish noise
from structural bias.

Run:
    python scripts/diagnose_simulator_alpha.py
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import jax

from gleipnir.deep_hedging.jax.env import (
    DeepHedgingEnvConfig, GRID_COARSE,
)
from gleipnir.deep_hedging.jax.pricing import (
    HestonMarketParams, compile_padded_grid,
)
from gleipnir.deep_hedging.jax.rollout import (
    simulate_heston_market_batch,
)


def fmt_alpha(label: str, mean: float, sem: float, width: int = 50) -> str:
    z = mean / sem if sem > 1e-12 else float("nan")
    flag = "  ← STRUCTURAL" if not math.isnan(z) and abs(z) > 3 else ""
    return f"  {label:<{width}s} = {mean:+.4f}  ± {sem:.4f}  (z={z:+5.2f}){flag}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paths", type=int, default=4096)
    parser.add_argument("--eval-seed", type=int, default=9999)
    parser.add_argument("--horizon", type=int, default=252)
    parser.add_argument("--initial-spot", type=float, default=100.0)
    parser.add_argument("--initial-variance", type=float, default=0.04)
    args = parser.parse_args()

    grid = GRID_COARSE
    env_config = DeepHedgingEnvConfig(
        horizon_steps=args.horizon,
        option_grid=grid,
        dt=1.0 / 252.0,
        transaction_cost_underlying=1.0e-4,
        transaction_cost_option=1.0e-2,
        risk_aversion=1_000.0,
        scheme="qe",
    )
    market = HestonMarketParams(
        kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, r=0.0, q=0.0,
    )
    padded_grid = compile_padded_grid(
        grid, dt=env_config.dt, horizon_steps=args.horizon,
    )

    print("=" * 78)
    print("Simulator alpha diagnostic")
    print(f"  n_paths={args.n_paths}  eval_seed={args.eval_seed}  horizon={args.horizon}")
    print(f"  Heston(kappa={market.kappa}, theta={market.theta}, sigma_v={market.sigma_v}, rho={market.rho})")
    print(f"  S_0={args.initial_spot}  v_0={args.initial_variance}  r=q=0")
    print("=" * 78)

    key = jax.random.PRNGKey(args.eval_seed)
    keys = jax.random.split(key, args.n_paths)
    traj = simulate_heston_market_batch(
        config=env_config, market=market, padded_grid=padded_grid,
        initial_spot=args.initial_spot, initial_variance=args.initial_variance,
        keys=keys,
    )

    spots = np.asarray(traj.spots, dtype=np.float64)              # (B, T+1)
    prices = np.asarray(traj.instrument_prices, dtype=np.float64) # (B, T+1, N)
    masks = np.asarray(traj.action_masks, dtype=bool)             # (T+1, N)
    B = args.n_paths
    n_inst = grid.n_instruments

    # ── 1. Spot martingale check ────────────────────────────────────────
    print("\n[1] Spot martingale check  (E[S_T] - S_0 should be ≈ 0)")
    spot_change = spots[:, -1] - args.initial_spot
    mean_sp = float(spot_change.mean())
    sem_sp = float(spot_change.std(ddof=1) / math.sqrt(B))
    print(fmt_alpha("E[S_T] - S_0", mean_sp, sem_sp))
    print(f"    relative bias: {mean_sp / args.initial_spot * 1e4:+.1f} bp of spot per year")

    # ── 2. Payoff consistency ───────────────────────────────────────────
    K = 100.0
    payoffs = np.maximum(spots[:, -1] - K, 0.0)
    mean_payoff = float(payoffs.mean())
    sem_payoff = float(payoffs.std(ddof=1) / math.sqrt(B))
    # "Theoretical" price of the liability (1Y ATM call) — read off the
    # COS price of the closest grid option at t=0. The liability isn't
    # in the grid (252-step), but we can use Black-Scholes with v_0 as
    # a sanity bracket.
    sigma0 = math.sqrt(args.initial_variance)
    T = args.horizon * env_config.dt
    d1 = (math.log(args.initial_spot / K) + 0.5 * sigma0 * sigma0 * T) / (sigma0 * math.sqrt(T))
    d2 = d1 - sigma0 * math.sqrt(T)
    from scipy.stats import norm as _norm
    bs_price = args.initial_spot * _norm.cdf(d1) - K * _norm.cdf(d2)
    print("\n[2] Payoff consistency  (E[payoff] vs analytical price at t=0)")
    print(fmt_alpha(f"E[max(S_T - {K:.0f}, 0)]", mean_payoff, sem_payoff))
    print(f"    BS reference price (constant σ=√v_0={sigma0:.4f}): {bs_price:.4f}")
    diff = mean_payoff - bs_price
    print(f"    empirical − BS: {diff:+.4f}  "
          f"(Heston correction is normally small for ATM 1Y, "
          f"so |diff| ≲ 0.10 is expected)")

    # ── 3. Per-instrument "buy 1 at t=0, hold" alpha ────────────────────
    # This is the most diagnostic test. For each instrument i, compute:
    #   PnL_i = -price_0[i] - tc[i] * price_0[i]    (entry cost)
    #         + sum_{t : mask[t,i]} 0                (no rebalancing)
    #         + price_T_eff[i]                       (final mark)
    #         - tc[i] * price_T_eff[i]               (exit cost)
    # where T_eff is the last time index where the instrument is tradable.
    print("\n[3] Per-contract buy-and-hold alpha  (E[P_expiry^i - P_0^i], before costs)")
    print("    Each option is bought at issue (t=0) and held to its own expiry, where it settles at")
    print("    intrinsic value; the underlying is held to the horizon. A self-consistent simulator")
    print("    gives zero at r = q = 0.")
    print(f"    {'instrument':>30s}  {'price_0':>9s}  {'mean_pnl':>10s}  {'sem':>8s}  {'z':>6s}")

    # Build instrument labels (slot 0 = underlying)
    labels = ["underlying (S)"]
    for m in grid.maturities:
        for k in grid.moneyness_by_maturity[m]:
            labels.append(f"call  m={m:>2d} K/S={k:.3f}")
            labels.append(f"put   m={m:>2d} K/S={k:.3f}")

    instrument_alphas = {}
    for i in range(n_inst):
        mask_i = masks[:, i]  # (T+1,)
        # the first contract's expiry: the first step the slot cannot trade
        end = args.horizon if mask_i.all() else int(np.where(~mask_i)[0][0])
        p0 = prices[:, 0, i]
        pnl = prices[:, end, i] - p0
        mean = float(pnl.mean())
        sem = float(pnl.std(ddof=1) / math.sqrt(B))
        z = mean / sem if sem > 1e-12 else float("nan")
        flag = "  ← STRUCTURAL" if not math.isnan(z) and abs(z) > 3 else ""
        print(f"    {labels[i]:>30s}  {float(p0.mean()):>9.4f}  {mean:>+10.4f}  {sem:>8.4f}  {z:>+6.2f}{flag}")
        instrument_alphas[labels[i]] = (mean, sem)

    print("\n[3b] Predictability  (does today's variance forecast a held contract's next move?)")
    print("    Slope of the one-step price change on (theta - v_t), over steps where the same")
    print("    contract is held; |z| > 3 means an agent that sees v_t can trade on it.")
    variances = np.asarray(traj.variances)
    predictable = 0
    for i in range(1, n_inst):
        held = masks[:-1, i]
        gain = (prices[:, 1:, i] - prices[:, :-1, i])[:, held].ravel()
        gap = (args.initial_variance - variances[:, :-1])[:, held].ravel()
        slope, intercept = np.polyfit(gap, gain, 1)
        resid = gain - slope * gap - intercept
        z = slope / (resid.std() / (gap.std() * math.sqrt(gap.size)))
        predictable += abs(z) > 3
        print(f"    {labels[i]:>30s}  slope {slope:+.4f}  (z={z:+6.2f}){'  ← PREDICTABLE' if abs(z) > 3 else ''}")

    # ── 4. Do-nothing eval ──────────────────────────────────────────────
    print("\n[4] Do-nothing baseline  (positions=0 forever)")
    print("    By construction:  mean_pnl = 0,  mean_hedging_error = -mean_payoff,")
    print(f"    std_hedging_error = std_payoff = {payoffs.std(ddof=0):.4f}")
    print("    (no observed value can disagree with this — pure pipeline check.)")

    # ── 5. Headline summary ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Headline:")
    print(f"  spot drift over 1Y:   {mean_sp:+.4f}  (z={mean_sp/sem_sp:+.2f})")
    print(f"  payoff vs BS price:   {diff:+.4f}")
    n_struct = sum(
        1 for (m, s) in instrument_alphas.values()
        if s > 1e-12 and abs(m / s) > 3
    )
    print(f"  instruments with |z|>3 buy-and-hold alpha: {n_struct} / {len(instrument_alphas)}")
    print(f"  options whose next move today's variance predicts (|z|>3): {predictable} / {n_inst - 1}")
    if n_struct or predictable:
        print("  → the simulator leaks alpha a learned hedger can harvest")
    else:
        print("  → no leak found: held contracts are martingales, unconditionally and given v_t")
    print("=" * 78)


if __name__ == "__main__":
    main()
