#!/usr/bin/env python3
"""Export the Learning to Hedge page's artifacts: trained hedgers' evaluations, with provenance.

Deep hedging (``backend/data/rl/deep_hedging.json``). The LSTM checkpoints of a
``scripts/validate_lstm_vs_bs.py`` run (underlying-only and with the option
grid, one per seed) and the classical hedges (Black-Scholes delta,
delta-gamma, delta-gamma-vega) are evaluated on the same 4,096 paths of the
experiment's market (``gleipnir.deep_hedging.experiments``): per-hedger
metrics (per seed for the LSTMs), hedging-error histograms on shared bins,
three sample paths with each hedger's positions, and the training curves.

Schema 2 adds the series the page charts, in units of the spot (100 at
issue). A hedger's book starts with ``setup.initial_cash``, the cash the
evaluation credits before the first trade (0: the premium is not credited, so
hedging errors and wealth are net of it; ``setup.fair_premium`` is the call's
day-0 value). Its hedge account on day t is that cash, plus the gains of the
positions held over the days before t, less the costs of every trade through
day t's: the book at the close of day t, after that day's rebalancing. There
is no trade on the expiry day, so the day-T account is the terminal P&L, and
less the payoff it is the hedging error.

- ``setup.option_grid``: the listed-option slots in instrument order (the
  order of ``positions[..., 1:]``): kind, tenor in days, moneyness at issue.
- ``samples[i].liability_value``: day 0 to T, the true-model value of the call
  the desk owes (positive): its Heston price (COS,
  ``mjollnir.jax.fourier_price``) at the market's parameters and the path's
  spot and simulated variance, and the payoff on the expiry day.
- ``samples[i].portfolio[hedger]``: the hedge account, day 0 to T.
- ``samples[i].hedging_error[hedger]``: the path's hedging error, the last
  account value less the payoff.
- ``samples[i].option_holdings[hedger]``: per option slot, the holdings after
  each day's trade (days 0 to T-1), for the hedgers that hold options.
- ``wealth_bands[hedger]``: per day, the 5/25/50/75/95% quantiles over all
  evaluation paths of the mark-to-market P&L, hedge account less liability
  value; on day T, the hedging error's quantiles.
- ``terminal_histogram``: the uncentred hedging errors on bins shared by all
  hedgers (``histogram`` is the centred ones).

AIS cliquet (``backend/data/rl/cliquet_ais.json``). The registered three-seed
run of aispg's concurrent cliquet-hedging benchmark, summarized per method
(mean and standard deviation over seeds) with its learning curves.

Both files record where every number came from: run ids, checkpoint and
result-file SHA-256s, the evaluation seed, the commit of this repository.

Usage (a GPU helps for the deep-hedging evaluation)::

    uv run --extra web --extra rl --extra cuda python scripts/export_rl_demos.py \\
        --run runs/validate_lstm_vs_bs/<run-id> \\
        --aispg ../aispg/experiments/results/cliquet_concurrent_learning.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import jax
import numpy as np
import torch
from mjollnir.jax import fourier_price

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gleipnir.deep_hedging.agents.analytical_delta_hedger import build_option_grid_spec  # noqa: E402
from gleipnir.deep_hedging.experiments import (  # noqa: E402
    EVAL_PATHS,
    EVAL_SEED,
    INITIAL_SPOT,
    INITIAL_VARIANCE,
    heston_easy,
    make_env_config,
    short_call,
)
from gleipnir.deep_hedging.jax.env import GRID_COARSE  # noqa: E402
from gleipnir.deep_hedging.jax.pricing import compile_padded_grid  # noqa: E402
from gleipnir.deep_hedging.training.torch_buehler import BuehlerTrainer  # noqa: E402
from gleipnir.deep_hedging.utils.eval_baseline import analytical_hedge_paths  # noqa: E402

OUT = ROOT / "backend" / "data" / "rl"
PHASES = {"lstm_underlying": "P1_underlying_only", "lstm_options": "P2_with_options"}
HEDGERS = {
    "bs_delta": ("Black-Scholes delta", "Underlying only, at the true variance"),
    "delta_gamma": ("Delta-gamma", "Underlying and one listed option, gamma-neutral"),
    "delta_gamma_vega": ("Delta-gamma-vega", "Underlying and two listed options, gamma- and vega-neutral"),
    "lstm_underlying": ("LSTM, underlying only", "Learned; trades the underlying only"),
    "lstm_options": ("LSTM with options", "Learned; trades the underlying and the listed options"),
}
TAIL = 0.05
N_BINS = 60
SAMPLE_QUANTILES = (0.15, 0.5, 0.85)  # terminal spot quantiles of the sample paths
CURVE_POINTS = 100
BANDS = {"p5": 0.05, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p95": 0.95}
DECIMALS = 4
IDENTITY_TOLERANCE = 1e-3  # recomputed account vs the evaluation's float32 rollout
PRICE_TOLERANCE = 1e-4  # day-0 repricing of the listed options vs the simulator's own prices


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def metrics(errors: np.ndarray, costs: np.ndarray, positions: np.ndarray, prices: np.ndarray) -> dict:
    """Hedging-error statistics; the tail is the mean of the worst 5% of errors around their mean."""
    centred = errors - errors.mean()
    tail = np.sort(centred)[: max(1, int(TAIL * centred.size))]
    trades = np.abs(np.diff(positions, axis=1, prepend=0.0))  # (B, T, N)
    return {
        "mean_error": float(errors.mean()),
        "std_error": float(errors.std()),
        "expected_shortfall_95": float(-tail.mean()),
        "mean_cost": float(costs.mean()),
        "option_turnover": float((trades[..., 1:] * prices[:, :-1, 1:]).sum(axis=(1, 2)).mean()),
    }


def shared_histogram(values: dict[str, np.ndarray]) -> dict:
    """Counts per hedger on bins spanning the pooled 0.1%-99.9% quantiles; the tails land in the end bins."""
    edges = np.linspace(*np.quantile(np.concatenate(list(values.values())), [0.001, 0.999]), N_BINS + 1)
    counts = {name: {"counts": np.histogram(np.clip(v, edges[0], edges[-1]), edges)[0].tolist()}
              for name, v in values.items()}
    return {"edges": np.round(edges, 4).tolist(), **counts}


def summarize_seeds(per_seed: list[dict]) -> dict:
    keys = per_seed[0].keys()
    return {k: {"mean": float(np.mean([m[k] for m in per_seed])), "sd": float(np.std([m[k] for m in per_seed], ddof=1))
                if len(per_seed) > 1 else 0.0, "values": [m[k] for m in per_seed]} for k in keys}


def training_curves(run: Path) -> dict:
    curves: dict[str, dict[str, list]] = {}
    for line in (run / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") != "epoch":
            continue
        name = next(k for k, v in PHASES.items() if v == row["phase"])
        curves.setdefault(name, {}).setdefault(str(int(row["seed"])), []).append(
            (int(row["epoch"]), float(row["std_hedging_error"])))
    out = {}
    for name, seeds in curves.items():
        out[name] = {}
        for seed, points in seeds.items():
            points.sort()
            step = max(1, len(points) // CURVE_POINTS)
            kept = points[::step] + ([points[-1]] if (len(points) - 1) % step else [])
            out[name][seed] = {"epoch": [p[0] for p in kept], "std_error": [round(p[1], 4) for p in kept]}
    return out


def _require(ok: bool, message: str) -> None:
    """A consistency check of the export, kept under ``python -O``."""
    if not ok:
        raise AssertionError(message)


def _rounded(values) -> list | float:
    """Rounded for the JSON, with no negative zeros."""
    return (np.round(np.asarray(values, dtype=np.float64), DECIMALS) + 0.0).tolist()


@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0, 0, None, None))
def _heston_price(market, spot, variance, strike, tau, is_call):
    """Heston COS price of a European option, one per path."""
    return fourier_price(spot, strike, tau, market.r, market.q, variance, market.kappa, market.theta,
                         market.sigma_v, market.rho, is_call)


def check_option_grid(batch, spec: list[dict], market, dt: float) -> None:
    """The slots of ``spec`` are the batch's instruments 1..N-1: each one's day-0
    strike, tenor and price (repriced at the market's parameters) match."""
    spots, variances = np.asarray(batch.spots, np.float64)[:, 0], np.asarray(batch.variances, np.float64)[:, 0]
    strikes = np.asarray(batch.option_strikes, np.float64)[:, 0]
    taus = np.asarray(batch.option_taus, np.float64)[:, 0]
    prices = np.asarray(batch.instrument_prices, np.float64)[:, 0]
    _require(prices.shape[1] == len(spec) + 1, "the grid spec does not cover the batch's instruments")
    for j, slot in enumerate(spec):
        _require(slot["idx"] == j + 1, f"slot {j} is listed as instrument {slot['idx']}")
        _require(np.allclose(strikes[:, j], spots * slot["moneyness"]), f"slot {j}'s strike is not its moneyness")
        _require(np.allclose(taus[:, j], slot["tau_steps"] * dt), f"slot {j}'s tenor is not {slot['tau_steps']} days")
        repriced = np.asarray(_heston_price(market, spots, variances, strikes[:, j], slot["tau_steps"] * dt,
                                            slot["kind"] == "call"))
        gap = np.abs(repriced - prices[:, 1 + j]).max()
        _require(gap < PRICE_TOLERANCE, f"slot {j} is not a {slot['kind']}: day-0 price off by {gap:.2e}")


def liability_values(batch, liability, market, dt: float) -> np.ndarray:
    """(B, T+1) value of the liability the desk owes (positive for a call it is
    short): its Heston price at the market's parameters, the path's spot and
    simulated variance on each day, and its payoff on the expiry day T."""
    spots = np.asarray(batch.spots, dtype=np.float64)
    variances = np.asarray(batch.variances, dtype=np.float64)
    horizon = spots.shape[1] - 1
    _require(liability.kind in ("call", "put"), f"no pricer for a {liability.kind} liability")
    _require(liability.maturity == horizon, "the liability must expire at the horizon")
    strikes = np.full(spots.shape[0], float(liability.strike))
    values = np.empty_like(spots)
    for t in range(horizon):
        values[:, t] = np.asarray(_heston_price(market, spots[:, t], variances[:, t], strikes,
                                                (liability.maturity - t) * dt, liability.kind == "call"))
    sign = 1.0 if liability.kind == "call" else -1.0
    values[:, horizon] = np.maximum(sign * (spots[:, horizon] - liability.strike), 0.0)
    return liability.quantity * values


def hedge_accounts(positions: np.ndarray, prices: np.ndarray, rates: np.ndarray, initial_cash: float) -> np.ndarray:
    """(B, T+1) hedge account at the close of each day, booked as the rollouts do:
    each day's trade at that day's prices less its proportional cost; positions
    (B, T, N) are the holdings after each day's trade, prices (B, T+1, N)."""
    positions, prices = np.asarray(positions, np.float64), np.asarray(prices, np.float64)
    trades = np.diff(positions, axis=1, prepend=0.0)
    costs = (np.asarray(rates, np.float64) * np.abs(trades * prices[:, :-1])).sum(axis=2).cumsum(axis=1)
    gains = (positions * np.diff(prices, axis=1)).sum(axis=2).cumsum(axis=1)
    gains = np.concatenate([np.zeros((positions.shape[0], 1)), gains], axis=1)  # through day t: positions held before it
    costs = np.concatenate([costs, costs[:, -1:]], axis=1)  # through day t's trade; none on day T
    return initial_cash + gains - costs


def export_deep_hedging(run: Path) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    seeds, horizon = summary["seeds"], int(summary["horizon"])
    env_config = make_env_config(horizon=horizon)
    market = heston_easy()
    grid = compile_padded_grid(GRID_COARSE, dt=env_config.dt, horizon_steps=horizon)
    liability = short_call(horizon)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    spec = build_option_grid_spec(GRID_COARSE)
    results: dict[str, dict] = {}
    paths: dict[str, dict] = {}
    checkpoints = {}
    reference = batch = prices = None
    for name, phase in PHASES.items():
        per_seed, per_seed_paths = [], []
        for seed in seeds:
            checkpoint = run / "checkpoints" / f"{phase}_seed{seed}.pt"
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            trainer = BuehlerTrainer.from_heston(
                config=saved["config"], env_config=env_config, market_params=market, padded_grid=grid,
                liability=liability, initial_spot=INITIAL_SPOT, initial_variance=INITIAL_VARIANCE,
                device=device, seed=seed)
            trainer.load_checkpoint(str(checkpoint))
            if reference is None:  # every hedger is evaluated on this batch
                reference = trainer
                batch = trainer._market_sampler(EVAL_PATHS, _key(EVAL_SEED))
                prices = np.asarray(batch.instrument_prices)
            out = trainer.evaluate_paths(n_paths=EVAL_PATHS, seed=EVAL_SEED)
            _require(np.array_equal(out["spots"], batch.spots), f"{phase} seed {seed} was evaluated on other paths")
            per_seed.append(metrics(out["hedging_errors"], out["costs"], out["positions"], prices))
            per_seed_paths.append(out)
            checkpoints[f"{phase}_seed{seed}"] = sha256(checkpoint)
        results[name] = {"seeds": seeds, **summarize_seeds(per_seed)}
        # the seed whose error spread is the median one represents the method on the charts
        order = np.argsort([m["std_error"] for m in per_seed])
        representative = int(order[len(order) // 2])
        results[name]["representative_seed"] = seeds[representative]
        paths[name] = per_seed_paths[representative]

    rates = reference.transaction_cost_rates.cpu().numpy()
    initial_cash = float(reference.initial_cash)
    for name in ("bs_delta", "delta_gamma", "delta_gamma_vega"):
        kind = {"bs_delta": "delta"}.get(name, name)
        out = analytical_hedge_paths(
            reference._market_sampler, env_config.horizon_steps, GRID_COARSE.n_instruments, rates, liability,
            hedge_kind=kind, option_grid_spec=spec, dt=env_config.dt, n_paths=EVAL_PATHS, eval_seed=EVAL_SEED,
            initial_cash=initial_cash)
        _require(np.array_equal(out["spots"], batch.spots), f"{name} was evaluated on other paths")
        results[name] = metrics(out["hedging_errors"], out["costs"], out["positions"], prices)
        paths[name] = out

    centred = {name: p["hedging_errors"] - p["hedging_errors"].mean() for name, p in paths.items()}

    # the chart series: the liability's value on every path and day, each hedger's account against it
    check_option_grid(batch, spec, market, env_config.dt)
    liability_value = liability_values(batch, liability, market, env_config.dt)
    _require(np.ptp(liability_value[:, 0]) < 1e-9, "the paths do not start from one state")
    accounts, wealth_bands = {}, {}
    quantiles = list(BANDS.values())
    for name, p in paths.items():
        payoff_gap = np.abs(liability_value[:, -1] - p["liability_payoffs"]).max()
        _require(payoff_gap < IDENTITY_TOLERANCE, f"{name}: the expiry value misses the payoff by {payoff_gap:.1e}")
        accounts[name] = hedge_accounts(p["positions"], prices, rates, initial_cash)
        gap = np.abs(accounts[name][:, -1] - liability_value[:, -1] - p["hedging_errors"]).max()
        _require(gap < IDENTITY_TOLERANCE, f"{name}: the account at expiry less the payoff misses the hedging error "
                 f"by {gap:.1e}")
        bands = np.quantile(accounts[name] - liability_value, quantiles, axis=0)
        band_gap = np.abs(bands[:, -1] - np.quantile(p["hedging_errors"], quantiles)).max()
        _require(band_gap < IDENTITY_TOLERANCE,
                 f"{name}: the expiry wealth band misses the hedging errors by {band_gap:.1e}")
        wealth_bands[name] = {key: _rounded(band) for key, band in zip(BANDS, bands, strict=True)}
        print(f"{name}: account at expiry less payoff = hedging error to {gap:.1e} on every path, "
              f"bands to {band_gap:.1e}")
    option_traders = [name for name, p in paths.items() if np.round(p["positions"][..., 1:], DECIMALS).any()]

    spots = np.asarray(batch.spots)
    picks = [int(np.argmin(np.abs(spots[:, -1] - np.quantile(spots[:, -1], q)))) for q in SAMPLE_QUANTILES]
    samples = []
    for i in picks:
        sample = {"spot": np.round(spots[i], 3).tolist(), "positions": {}, "option_value": {},
                  "liability_value": _rounded(liability_value[i]), "portfolio": {}, "hedging_error": {},
                  "option_holdings": {}}
        for name, p in paths.items():
            sample["positions"][name] = np.round(p["positions"][i, :, 0], 4).tolist()
            sample["option_value"][name] = np.round((p["positions"][i, :, 1:] * prices[i, :-1, 1:]).sum(axis=1),
                                                    4).tolist()
            sample["portfolio"][name] = _rounded(accounts[name][i])
            sample["hedging_error"][name] = _rounded(p["hedging_errors"][i])
            if name in option_traders:
                sample["option_holdings"][name] = _rounded(p["positions"][i, :, 1:].T)
        samples.append(sample)

    payoffs = paths["bs_delta"]["liability_payoffs"]
    return {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "setup": {
            "market": heston_easy()._asdict(),
            "initial_spot": INITIAL_SPOT, "initial_variance": INITIAL_VARIANCE,
            "horizon_days": env_config.horizon_steps,
            "liability": f"short at-the-money call expiring in {horizon} trading days",
            "options": "listed 5- and 20-day calls and puts struck at 95%, 100% and 105% of the spot at issue",
            "costs": {"underlying": env_config.transaction_cost_underlying, "options": env_config.transaction_cost_option},
            "objective": f"risk aversion {env_config.risk_aversion:g} x variance of the hedging error + expected costs",
            "training": {k: summary[k] for k in ("n_epochs", "batch_size", "learning_rate", "cosine_lr", "seeds")},
            "eval_paths": EVAL_PATHS, "eval_seed": EVAL_SEED,
            "mean_payoff": float(payoffs.mean()),
            "initial_cash": initial_cash,
            "fair_premium": _rounded(liability_value[0, 0]),
            "option_grid": [{"kind": s["kind"], "tenor_days": s["tau_steps"], "moneyness": s["moneyness"]}
                            for s in spec],
        },
        "hedgers": [{"key": k, "label": label, "summary": text} for k, (label, text) in HEDGERS.items()],
        "results": results,
        "histogram": shared_histogram(centred),
        "terminal_histogram": shared_histogram({name: p["hedging_errors"] for name, p in paths.items()}),
        "samples": samples,
        "wealth_bands": wealth_bands,
        "training_curves": training_curves(run),
        "provenance": {"run": run.name, "checkpoints_sha256": checkpoints, "gleipnir_commit": git_commit()},
    }


def _key(seed: int):
    return jax.random.PRNGKey(seed)


AIS_METHODS = {
    "reactive": ("Reactive", "Acts on the current observation only"),
    "sequential": ("Pretrain then freeze", "Recurrent AIS learned first, then frozen while the policy learns"),
    "concurrent": ("Concurrent recurrent AIS", "AIS and policy learned together on two timescales"),
    "oracle_state": ("Exact contract state", "Sees the reset spot and accrued return directly"),
}
AIS_BASELINES = {
    "no_hedge": ("No hedge", "Holds nothing"),
    "static_vanilla": ("Static vanilla basket", "The best fixed exposure to the put-call basket"),
    "oracle_cap_exit": ("Full-state cap exit", "A rule that sees the contract state and exits once capped"),
}
AIS_METRICS = ("terminal_pnl_rmse", "terminal_abs_cvar_95", "mean_trading_cost")
AIS_PROBES = ("period_return_probe_r2", "accrued_return_probe_r2")


def export_cliquet(results_path: Path) -> dict:
    data = json.loads(results_path.read_text())
    env = data["environment"]
    methods = {}
    for key, (label, text) in AIS_METHODS.items():
        runs = data["learned"][key]
        evaluation = [r["evaluation"] for r in runs]
        methods[key] = {
            "label": label, "summary": text, "seeds": [r["seed"] for r in runs],
            **summarize_seeds([{m: e[m] for m in (*AIS_METRICS, *AIS_PROBES)} for e in evaluation]),
            "curves": [{k: r["diagnostics"]["curves"][k] for k in ("iteration", "terminal_pnl_rmse", "reward_mse")}
                       for r in runs],
        }
    baselines = {key: {"label": label, "summary": text, **{m: data["baselines"][key][m] for m in AIS_METRICS}}
                 for key, (label, text) in AIS_BASELINES.items()}
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "setup": {
            "contract": (f"short return cliquet on notional {env['notional']:g}: {len(env['reset_steps'])} resets, "
                         f"each period's return floored at {env['local_floor']:g} and capped at {env['local_cap']:g}, "
                         f"the sum capped at {env['global_cap']:g}"),
            "hedge": (f"a European basket ({env['hedge_call_weight']:g} call, {env['hedge_put_weight']:g} put, "
                      f"strike {env['hedge_strike']:g}) whose exposure the agent chooses each period"),
            "hidden": "the previous reset spot and the accrued capped return are not observed",
            "market": f"Heston calibrated to {data['calibration']['asset']} over {data['calibration']['window']}",
            "calibration_hash": data["calibration"]["content_hash"],
            "proportional_cost": env["proportional_cost"],
            "training": data["training"],
        },
        "baselines": baselines,
        "methods": methods,
        "provenance": {"results_file": results_path.name, "results_sha256": sha256(results_path),
                       "source": "aispg experiments/cliquet_concurrent_learning.py, registered configuration"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="a validate_lstm_vs_bs run directory")
    parser.add_argument("--aispg", type=Path, help="aispg's cliquet_concurrent_learning.json")
    args = parser.parse_args()
    if not args.run and not args.aispg:
        parser.error("nothing to export: pass --run and/or --aispg")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.run:
        path = OUT / "deep_hedging.json"
        path.write_text(json.dumps(export_deep_hedging(args.run.resolve()), separators=(",", ":")) + "\n")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} kB)")
    if args.aispg:
        path = OUT / "cliquet_ais.json"
        path.write_text(json.dumps(export_cliquet(args.aispg.resolve()), indent=1) + "\n")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
