#!/usr/bin/env python3
"""Export the Learning to Hedge page's artifacts: trained hedgers' evaluations, with provenance.

Deep hedging (``backend/data/rl/deep_hedging.json``). The LSTM checkpoints of a
``scripts/validate_lstm_vs_bs.py`` run (underlying-only and with the option
grid, one per seed) and the classical hedges (Black-Scholes delta,
delta-gamma, delta-gamma-vega) are evaluated on the same 4,096 paths of the
experiment's market (``gleipnir.deep_hedging.experiments``): per-hedger
metrics (per seed for the LSTMs), hedging-error histograms on shared bins,
three sample paths with each hedger's positions, and the training curves.

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
from pathlib import Path

import numpy as np
import torch

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


def export_deep_hedging(run: Path) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    seeds, horizon = summary["seeds"], int(summary["horizon"])
    env_config = make_env_config(horizon=horizon)
    market = heston_easy()
    grid = compile_padded_grid(GRID_COARSE, dt=env_config.dt, horizon_steps=horizon)
    liability = short_call(horizon)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    results: dict[str, dict] = {}
    paths: dict[str, dict] = {}
    checkpoints = {}
    reference = None
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
            out = trainer.evaluate_paths(n_paths=EVAL_PATHS, seed=EVAL_SEED)
            reference = reference or trainer
            prices = np.asarray(trainer._market_sampler(EVAL_PATHS, _key(EVAL_SEED)).instrument_prices)
            per_seed.append(metrics(out["hedging_errors"], out["costs"], out["positions"], prices))
            per_seed_paths.append(out)
            checkpoints[f"{phase}_seed{seed}"] = sha256(checkpoint)
        results[name] = {"seeds": seeds, **summarize_seeds(per_seed)}
        # the seed whose error spread is the median one represents the method on the charts
        order = np.argsort([m["std_error"] for m in per_seed])
        representative = int(order[len(order) // 2])
        results[name]["representative_seed"] = seeds[representative]
        paths[name] = per_seed_paths[representative]

    batch = reference._market_sampler(EVAL_PATHS, _key(EVAL_SEED))
    prices = np.asarray(batch.instrument_prices)
    for name in ("bs_delta", "delta_gamma", "delta_gamma_vega"):
        kind = {"bs_delta": "delta"}.get(name, name)
        out = analytical_hedge_paths(
            reference._market_sampler, env_config.horizon_steps, GRID_COARSE.n_instruments,
            reference.transaction_cost_rates.cpu().numpy(), liability, hedge_kind=kind,
            option_grid_spec=build_option_grid_spec(GRID_COARSE), dt=env_config.dt, n_paths=EVAL_PATHS,
            eval_seed=EVAL_SEED)
        results[name] = metrics(out["hedging_errors"], out["costs"], out["positions"], prices)
        paths[name] = out

    centred = {name: p["hedging_errors"] - p["hedging_errors"].mean() for name, p in paths.items()}
    edges = np.linspace(*np.quantile(np.concatenate(list(centred.values())), [0.001, 0.999]), N_BINS + 1)
    histograms = {name: {"counts": np.histogram(np.clip(e, edges[0], edges[-1]), edges)[0].tolist()}
                  for name, e in centred.items()}

    spots = np.asarray(batch.spots)
    picks = [int(np.argmin(np.abs(spots[:, -1] - np.quantile(spots[:, -1], q)))) for q in SAMPLE_QUANTILES]
    samples = []
    for i in picks:
        sample = {"spot": np.round(spots[i], 3).tolist(), "positions": {}, "option_value": {}}
        for name, p in paths.items():
            sample["positions"][name] = np.round(p["positions"][i, :, 0], 4).tolist()
            sample["option_value"][name] = np.round((p["positions"][i, :, 1:] * prices[i, :-1, 1:]).sum(axis=1),
                                                    4).tolist()
        samples.append(sample)

    payoffs = paths["bs_delta"]["liability_payoffs"]
    return {
        "schema_version": 1,
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
        },
        "hedgers": [{"key": k, "label": label, "summary": text} for k, (label, text) in HEDGERS.items()],
        "results": results,
        "histogram": {"edges": np.round(edges, 4).tolist(), **histograms},
        "samples": samples,
        "training_curves": training_curves(run),
        "provenance": {"run": run.name, "checkpoints_sha256": checkpoints, "gleipnir_commit": git_commit()},
    }


def _key(seed: int):
    import jax

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
        path.write_text(json.dumps(export_deep_hedging(args.run.resolve())) + "\n")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} kB)")
    if args.aispg:
        path = OUT / "cliquet_ais.json"
        path.write_text(json.dumps(export_cliquet(args.aispg.resolve()), indent=1) + "\n")
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
