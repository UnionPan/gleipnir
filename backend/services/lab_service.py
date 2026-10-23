"""Counterfactual Hedging Lab: a short-option book run through factual and counterfactual worlds.

The book (premium, hedgers) is fixed by an asset's real-data Heston calibration;
the world is either that calibration or a mjollnir scenario applied to it. All
numerics come from mjollnir.hedging; this module selects inputs from the
shipped ParamSets and reduces results to what the lab page draws.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from mjollnir import __version__ as mjollnir_version
from mjollnir.backtest import BacktestResult
from mjollnir.hedging import EuropeanOption, compare_hedgers, heston_price, implied_vol, standard_hedgers
from mjollnir.jax import configure_runtime
from mjollnir.params import ParamSet
from mjollnir.scenarios import correlation_breakdown, regime_shift, vol_spike

from backend.schemas.lab import (
    AssetInfo,
    Book,
    HedgeRequest,
    HedgeResponse,
    HedgerStats,
    LabCatalog,
    SamplePath,
    ScenarioInfo,
    WorldResult,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "lab"
DT = 1.0 / 252
SPOT0 = 100.0
N_PATHS = 4096  # fixed so every request reuses the warmed-up compiled programs
MATURITIES_DAYS = (5, 21, 63)
RHO_ASSUMPTION = -0.7  # typical equity option-implied spot-vol correlation
HISTOGRAM_BINS = 48
HISTOGRAM_PAD = 0.04
TAIL = 0.05


@dataclass(frozen=True)
class _ScenarioKind:
    info: ScenarioInfo
    build: object  # severity -> mjollnir Scenario


SCENARIOS = {
    "vol_spike": _ScenarioKind(
        ScenarioInfo(
            name="vol_spike", label="Volatility spike",
            summary="Spot vol jumps by the severity multiple; long-run vol and vol-of-vol rise less.",
            severity_min=1.25, severity_max=3.0, severity_default=2.0,
        ),
        vol_spike,
    ),
    "correlation_breakdown": _ScenarioKind(
        ScenarioInfo(
            name="correlation_breakdown", label="Correlation breakdown",
            summary="Spot-vol correlation pinned to a crisis level (the severity is the new rho).",
            severity_min=-0.95, severity_max=-0.5, severity_default=-0.9,
        ),
        correlation_breakdown,
    ),
    "regime_shift": _ScenarioKind(
        ScenarioInfo(
            name="regime_shift", label="Regime shift",
            summary="Mean reversion slows by the severity factor and long-run variance rises.",
            severity_min=1.5, severity_max=5.0, severity_default=3.0,
        ),
        regime_shift,
    ),
}


class UnknownAssetError(KeyError):
    """The requested ticker is not among the shipped calibrations."""


@cache
def _manifest() -> dict:
    return json.loads((DATA_DIR / "manifest.json").read_text())


@cache
def _calibration(ticker: str) -> ParamSet:
    entries = {a["ticker"]: a for a in _manifest()["assets"]}
    if ticker not in entries:
        raise UnknownAssetError(ticker)
    return ParamSet.load(DATA_DIR / entries[ticker]["file"])


def catalog() -> LabCatalog:
    manifest = _manifest()
    assets = []
    for entry in manifest["assets"]:
        ps = _calibration(entry["ticker"])
        assets.append(AssetInfo(
            ticker=entry["ticker"], sector=entry["sector"], params=dict(ps.params),
            params_hash=ps.content_hash(), vol_today=entry["state"]["vol_today"],
            as_of=entry["state"]["as_of"], diagnostics=entry["diagnostics"],
        ))
    return LabCatalog(
        window=manifest["window"], estimator=manifest["estimator"], assets=assets,
        scenarios=[kind.info for kind in SCENARIOS.values()],
        maturities_days=list(MATURITIES_DAYS), n_paths=N_PATHS, rho_assumption=RHO_ASSUMPTION,
    )


def _book(request: HedgeRequest) -> tuple[ParamSet, EuropeanOption]:
    calibration = _calibration(request.asset)
    book = calibration
    if request.rho is not None:
        book = calibration.derive(
            set_={"rho": request.rho},
            note=f"rho assumed {request.rho:g}: daily closes do not identify the spot-vol correlation",
        )
    option = EuropeanOption(
        strike=SPOT0 * request.option.moneyness,
        maturity=request.option.maturity_days * DT,
        is_call=request.option.is_call,
    )
    return book, option


def _scenario(request: HedgeRequest):
    if request.scenario is None:
        return None
    kind = SCENARIOS[request.scenario.name]
    lo, hi = kind.info.severity_min, kind.info.severity_max
    if not lo <= request.scenario.severity <= hi:
        raise ValueError(f"{request.scenario.name} severity must be in [{lo}, {hi}]")
    return kind.build(request.scenario.severity)


def _cvar(pnl: np.ndarray) -> float:
    worst = np.sort(pnl)[: max(1, round(TAIL * pnl.size))]
    return float(worst.mean())


def _world(label: str, description: str, params: ParamSet, results: dict[str, BacktestResult],
           edges: dict[str, np.ndarray], path_index: int) -> WorldResult:
    first = next(iter(results.values()))
    hedgers = {}
    for name, res in results.items():
        bins = edges[name]
        counts, _ = np.histogram(res.pnl, bins=bins)  # out-of-range tails are reported, not folded in
        hedgers[name] = HedgerStats(
            mean=float(res.pnl.mean()), std=float(res.pnl.std()), cvar_5=_cvar(res.pnl),
            mean_cost=float(res.costs.mean()), histogram=counts.tolist(),
            below=int((res.pnl < bins[0]).sum()), above=int((res.pnl > bins[-1]).sum()),
        )
    return WorldResult(
        label=label, description=description, params=dict(params.params),
        params_hash=first.params_hash, hedgers=hedgers,
        sample_path=SamplePath(
            spots=first.spots[path_index].tolist(),
            positions={n: r.positions[path_index].tolist() for n, r in results.items()},
            pnl={n: float(r.pnl[path_index]) for n, r in results.items()},
        ),
    )


def run_hedge(request: HedgeRequest) -> HedgeResponse:
    configure_runtime()
    started = time.perf_counter()
    book, option = _book(request)
    scenario = _scenario(request)
    premium = heston_price(option, book, SPOT0)
    hedgers = standard_hedgers(option, book, dt=DT, spot0=SPOT0)
    common = {"hedgers": hedgers, "n_paths": N_PATHS, "dt": DT, "spot0": SPOT0,
              "cost_bps": request.cost_bps, "seed": request.seed}

    worlds = {"factual": (book, compare_hedgers(book, option, **common))}
    if scenario is not None:
        shocked = scenario.apply(book)
        worlds["counterfactual"] = (shocked, compare_hedgers(book, option, scenario=scenario, **common))
    # Bins per hedger, shared by both worlds: hedged and unhedged P&L differ in
    # spread by an order of magnitude, the worlds of one hedger are what compare.
    edges = {}
    for name in hedgers:
        pooled = np.concatenate([results[name].pnl for _, results in worlds.values()])
        lo, hi = np.quantile(pooled, [0.005, 0.995])
        pad = HISTOGRAM_PAD * (hi - lo)  # keep point masses (e.g. premium kept) off the frame
        edges[name] = np.linspace(lo - pad, hi + pad, HISTOGRAM_BINS + 1)

    factual_results = worlds["factual"][1]
    terminal = next(iter(factual_results.values())).spots[:, -1]
    path_index = int(np.argmin(np.abs(terminal - np.median(terminal))))

    factual = _world("Factual", "As calibrated on real prices", book, factual_results, edges,
                     path_index)
    counterfactual = None
    if scenario is not None:
        shocked, shocked_results = worlds["counterfactual"]
        counterfactual = _world(SCENARIOS[request.scenario.name].info.label, scenario.description,
                                shocked, shocked_results, edges, path_index)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info("lab hedge %s %s in %.0f ms", request.asset,
                getattr(scenario, "name", "factual"), elapsed_ms)
    return HedgeResponse(
        book=Book(
            asset=request.asset, params=dict(book.params), params_hash=book.content_hash(),
            parent_hash=book.parent_hash, rho_source="estimated" if request.rho is None else "assumed",
            strike=option.strike, maturity_days=request.option.maturity_days,
            is_call=option.is_call, premium=premium, implied_vol=implied_vol(option, premium, SPOT0),
        ),
        bin_edges={name: e.tolist() for name, e in edges.items()},
        factual=factual, counterfactual=counterfactual,
        n_paths=N_PATHS, seed=request.seed, cost_bps=request.cost_bps,
        elapsed_ms=elapsed_ms, mjollnir_version=mjollnir_version,
    )


def warm_up() -> None:
    """Compile every maturity's programs so the first visitor does not pay for it."""
    started = time.perf_counter()
    ticker = _manifest()["assets"][0]["ticker"]
    for days in MATURITIES_DAYS:
        run_hedge(HedgeRequest(asset=ticker, option={"maturity_days": days}))
    logger.info("lab warm-up compiled %d maturities in %.1f s", len(MATURITIES_DAYS),
                time.perf_counter() - started)
