"""Train the learned band hedger for one lab model, through the lab's own simulated worlds.

Scenarios cover what the lab offers: every shipped book of the model, the
lab's maturities (5, 21, 63 trading days), strikes from 85% to 115% of the
spot, calls and puts, trading costs from 0 to 50 bp. The excess drift of
each training scenario is drawn at random around zero, so the band cannot
learn a directional bet: it can only learn when trading is not worth its
cost.

The objective, per scenario, is a mean-variance criterion on the P&L of
the sold option hedged by the band, in units of the option's premium
(floored, so cheap far-from-the-money options do not dominate):
``-E[X] + risk_aversion / 2 * Var(X)``. Trading costs lower E[X]; leaving
the hedge loose raises Var(X); the band learns the trade-off. The risk
aversion is drawn per scenario and given to the network, which learns the
whole family of bands, from cautious (wide aversion: close to delta) to
cost-conscious.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from mjollnir.backtest import hedge_market, simulate_hedge
from mjollnir.hedging import EuropeanOption, model_delta, model_price
from mjollnir.params import ParamSet

from .policy import band_strategy, init_params

DT = 1.0 / 252
SPOT = 100.0
MATURITIES = (5, 21, 63)
MONEYNESS = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
COSTS_BPS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
PREMIUM_FLOOR = 0.25
RISK_AVERSIONS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
VALIDATION_COSTS = (2.0, 5.0, 20.0)
VALIDATION_RISK_AVERSIONS = (1.0, 5.0, 20.0)
VALIDATION_MONEYNESS = (0.9, 1.0, 1.1)


@dataclass
class TrainConfig:
    steps: int = 1200
    scenarios_per_step: int = 8
    paths_per_scenario: int = 512
    learning_rate: float = 3e-3
    excess_drift_sd: float = 0.15
    validation_paths: int = 4096
    seed: int = 0
    log: callable = field(default=print, repr=False)


@dataclass
class Scenario:
    book: ParamSet
    maturity_days: int
    moneyness: float
    is_call: bool
    rate: float
    dividend: float
    premium: float


def _option(s: Scenario) -> EuropeanOption:
    return EuropeanOption(strike=SPOT * s.moneyness, maturity=s.maturity_days * DT, is_call=s.is_call)


def scenario_pool(books: list[tuple[ParamSet, float, float]], log=print) -> list[Scenario]:
    """Every (book, maturity, strike, call/put) with its premium under the book's model."""
    pool, started = [], time.perf_counter()
    for book, rate, dividend in books:
        for days in MATURITIES:
            for moneyness in MONEYNESS:
                for is_call in (True, False):
                    s = Scenario(book, days, moneyness, is_call, rate, dividend, 0.0)
                    s.premium = float(model_price(_option(s), book, SPOT, rate=rate, dividend=dividend, dt=DT))
                    pool.append(s)
    log(f"  {len(pool)} scenarios priced in {time.perf_counter() - started:.0f} s")
    return pool


def _stack(trees):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack([jnp.asarray(x, jnp.float64) for x in xs]), *trees)


def _batch(scenarios: list[Scenario], costs, drifts, risk_aversions):
    """Stacked traced inputs for a batch of same-model, same-maturity scenarios."""
    models, markets, bases = [], [], []
    for s, cost, drift in zip(scenarios, costs, drifts, strict=True):
        hm = hedge_market(s.book, dt=DT, mu=s.rate - s.dividend + drift, rate=s.rate, dividend=s.dividend, spot0=SPOT,
                          initial_cash=s.premium, cost_bps=cost)
        models.append(hm.model)
        markets.append(hm.market)
        bases.append(model_delta(_option(s), s.book, DT, s.rate, s.dividend).params)
    scale = jnp.asarray([max(s.premium, PREMIUM_FLOOR) for s in scenarios])
    return {"model": _stack(models), "market": _stack(markets), "base": _stack(bases),
            "cost_bps": jnp.asarray(costs, jnp.float64), "risk_aversion": jnp.asarray(risk_aversions, jnp.float64),
            "scale": scale,
            "strike": jnp.asarray([SPOT * s.moneyness for s in scenarios]),
            "is_call": jnp.asarray([s.is_call for s in scenarios])}


def make_loss(world, base_fn, horizon: int):
    band = band_strategy(base_fn)

    def scenario_loss(net, b, keys):
        from mjollnir.backtest import HedgeMarket

        def payoff(s):
            return jnp.where(b["is_call"], jnp.maximum(s - b["strike"], 0.0), jnp.maximum(b["strike"] - s, 0.0))

        pnl, costs = simulate_hedge(HedgeMarket(world, b["model"], b["market"]), band,
                                    {"base": b["base"], "net": net, "cost_bps": b["cost_bps"],
                                     "risk_aversion": b["risk_aversion"]}, keys,
                                    horizon_steps=horizon, payoff=payoff)
        x = pnl / b["scale"]
        return -jnp.mean(x) + 0.5 * b["risk_aversion"] * jnp.var(x), (jnp.std(pnl), jnp.mean(costs))

    def loss(net, batch, keys):
        values, aux = jax.vmap(scenario_loss, in_axes=(None, 0, 0))(net, batch, keys)
        return jnp.mean(values), aux

    return jax.jit(jax.value_and_grad(loss, has_aux=True))


def evaluate(world, base_fn, net, scenarios: list[Scenario], cost_bps: float, risk_aversion: float, n_paths: int,
             seed: int) -> dict:
    """The band and the model's own delta on the same paths, per scenario, at zero excess drift."""
    band = band_strategy(base_fn)
    rows = []
    for k, s in enumerate(scenarios):
        hm = hedge_market(s.book, dt=DT, mu=s.rate - s.dividend, rate=s.rate, dividend=s.dividend, spot0=SPOT,
                          initial_cash=s.premium, cost_bps=cost_bps)
        option = _option(s)
        base = model_delta(option, s.book, DT, s.rate, s.dividend)
        keys = jax.random.split(jax.random.PRNGKey(seed + k), n_paths)
        payoff = option.payoff
        delta_pnl, delta_costs = simulate_hedge(hm, base.fn, base.params, keys, horizon_steps=s.maturity_days, payoff=payoff)
        band_pnl, band_costs = simulate_hedge(hm, band, {"base": base.params, "net": net, "cost_bps": cost_bps,
                                                         "risk_aversion": risk_aversion}, keys,
                                              horizon_steps=s.maturity_days, payoff=payoff)
        rows.append({
            "asset": s.book.asset, "maturity_days": s.maturity_days, "moneyness": s.moneyness, "is_call": s.is_call,
            "premium": s.premium,
            "delta": {"mean": float(delta_pnl.mean()), "std": float(delta_pnl.std()), "cost": float(delta_costs.mean())},
            "band": {"mean": float(band_pnl.mean()), "std": float(band_pnl.std()), "cost": float(band_costs.mean())},
        })
    return {"cost_bps": cost_bps, "risk_aversion": risk_aversion, "rows": rows}


def summarize(validation: list[dict]) -> dict:
    """Per cost level: the band's cost saving, and its error spread relative to delta's."""
    out = {}
    for v in validation:
        rows = v["rows"]
        saving = np.array([1 - r["band"]["cost"] / r["delta"]["cost"] for r in rows if r["delta"]["cost"] > 1e-9])
        spread = np.array([r["band"]["std"] / r["delta"]["std"] for r in rows if r["delta"]["std"] > 1e-9])
        gain = np.array([(r["band"]["mean"] - r["delta"]["mean"]) / max(r["premium"], PREMIUM_FLOOR) for r in rows])
        out[f"{v['cost_bps']:g}bp, risk aversion {v['risk_aversion']:g}"] = {"median_cost_saving": float(np.median(saving)),
                                       "median_std_ratio": float(np.median(spread)),
                                       "median_mean_gain_per_premium": float(np.median(gain))}
    return out


def train(model: str, books: list[tuple[ParamSet, float, float]], config: TrainConfig) -> tuple[dict, dict]:
    """Trained band network weights and a report (configuration, validation)."""
    log = config.log
    rng = np.random.default_rng(config.seed)
    pool = scenario_pool(books, log)
    world = None
    by_maturity = {d: [s for s in pool if s.maturity_days == d] for d in MATURITIES}
    sample = by_maturity[MATURITIES[0]][0]
    base_fn = model_delta(_option(sample), sample.book, DT, sample.rate, sample.dividend).fn
    world = hedge_market(sample.book, dt=DT).world
    steps = {d: make_loss(world, base_fn, d) for d in MATURITIES}

    net = init_params(config.seed)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(net)
    key = jax.random.PRNGKey(config.seed)
    started = time.perf_counter()
    for step in range(config.steps):
        days = MATURITIES[step % len(MATURITIES)]
        group = by_maturity[days]
        chosen = [group[i] for i in rng.choice(len(group), config.scenarios_per_step, replace=False)]
        costs = rng.choice(COSTS_BPS, config.scenarios_per_step)
        drifts = rng.normal(0.0, config.excess_drift_sd, config.scenarios_per_step)
        aversions = rng.choice(RISK_AVERSIONS, config.scenarios_per_step)
        batch = _batch(chosen, costs, drifts, aversions)
        key, sub = jax.random.split(key)
        keys = jax.random.split(sub, config.scenarios_per_step * config.paths_per_scenario).reshape(
            config.scenarios_per_step, config.paths_per_scenario, -1)
        (value, (_, mean_costs)), grads = steps[days](net, batch, keys)
        updates, opt_state = optimizer.update(grads, opt_state, net)
        net = optax.apply_updates(net, updates)
        if step % 100 == 0 or step == config.steps - 1:
            log(f"  step {step:5d}  {days:2d}d  loss {float(value):+.4f}  "
                f"cost {float(jnp.mean(mean_costs)):.4f}  ({time.perf_counter() - started:.0f} s)")

    validation_set = [s for s in pool if s.moneyness in VALIDATION_MONEYNESS]
    validation = [evaluate(world, base_fn, net, validation_set, c, g, config.validation_paths, config.seed + 7_000)
                  for c in VALIDATION_COSTS for g in VALIDATION_RISK_AVERSIONS]
    report = {
        "model": model,
        "config": {k: v for k, v in config.__dict__.items() if k != "log"},
        "scenarios": {"assets": sorted({b.asset for b, _, _ in books}), "maturities": list(MATURITIES),
                      "moneyness": list(MONEYNESS), "costs_bps": list(COSTS_BPS),
                      "risk_aversions": list(RISK_AVERSIONS)},
        "summary": summarize(validation),
        "validation": validation,
        "train_seconds": round(time.perf_counter() - started, 1),
    }
    return net, report


def load_shipped_books(lab_dir: Path, model: str) -> list[tuple[ParamSet, float, float]]:
    """The lab's shipped books of ``model`` (a world name), with each asset's rate and dividend yield."""
    manifest = json.loads((lab_dir / "manifest.json").read_text())
    books = []
    for asset in manifest["assets"]:
        for entry in asset["models"].values():
            ps = ParamSet.load(lab_dir / entry["file"])
            if ps.model == model:
                books.append((ps, float(manifest["rate"]), float(asset["dividend_yield"])))
    if not books:
        raise ValueError(f"no shipped {model} books under {lab_dir}")
    return books
