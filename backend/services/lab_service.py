"""Counterfactual Hedging Lab: a short-option book run through factual and counterfactual worlds.

The book (premium, hedgers) is fixed by an asset's real-data calibration of
the chosen model; the world is either that calibration or a mjollnir scenario
expressed in the same model. All numerics come from mjollnir (worlds,
hedging, scenarios); this module selects inputs from the shipped ParamSets and
reduces results to what the lab page draws.

Drift policy: every lab book expires within 3 months, so the simulated spot
drifts at r - q with no excess drift. r is the 13-week Treasury bill yield at
the end of the calibration window and q the asset's dividend yield; the book
is priced and hedged at the same r and q, cash earns r and held shares earn q.
An asset's historical drift is reported with its standard error as an
educational estimate and never simulated.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
from mjollnir import __version__ as mjollnir_version
from mjollnir.backtest import BacktestResult
from mjollnir.hedging import MODEL_HEDGERS, EuropeanOption, compare_hedgers, implied_vol, model_price, \
    standard_hedgers
from mjollnir.jax import configure_runtime
from mjollnir.params import ParamSet
from mjollnir.scenarios import SCENARIO_MODELS, scenario_for

from backend.schemas.lab import (
    AssetInfo,
    Book,
    BookOrigin,
    CalibrationBundle,
    DriftEstimate,
    HedgeRequest,
    HedgeResponse,
    HedgerInfo,
    HedgerStats,
    LabCatalog,
    LabModelInfo,
    ModelFit,
    SamplePath,
    SimulatedPaths,
    ScenarioInfo,
    WorldResult,
)
from backend.services import learned_hedger
from backend.services.lab_bundle import bundle_hash, validated_world_params
from backend.services.lab_models import LAB_MODELS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "lab"
DT = 1.0 / 252
SPOT0 = 100.0
N_PATHS = 2048  # fixed so every request reuses the compiled programs
MATURITIES_DAYS = (5, 21, 63)
RHO_ASSUMPTION = -0.7  # typical equity option-implied spot-vol correlation
HISTOGRAM_BINS = 48
HISTOGRAM_PAD = 0.04
TAIL = 0.05
N_SHOWN = 40  # whole paths returned per world, for the lab's animation

HEDGER_LABELS = {
    "heston_delta": "Heston delta",
    "gbm_delta": "Black-Scholes at historical vol",
    "merton_delta": "Merton delta",
    "regime_delta": "Regime delta",
    "garch_delta": "GARCH-forecast delta",
    "rough_delta": "Rough-vol delta",
}


@dataclass(frozen=True)
class _ScenarioKind:
    label: str
    summary: str
    severity: tuple[float, float, float]  # min, max, default


SCENARIOS = {
    "vol_spike": _ScenarioKind("Volatility spike",
                               "Spot vol jumps by the severity multiple; slower-moving volatility rises less.",
                               (1.25, 3.0, 2.0)),
    "correlation_breakdown": _ScenarioKind("Correlation breakdown",
                                           "Spot-vol correlation pinned to a crisis level (the severity is the new rho).",
                                           (-0.95, -0.5, -0.9)),
    "regime_shift": _ScenarioKind("Regime shift",
                                  "Volatility shocks last the severity times longer and long-run variance rises.",
                                  (1.5, 5.0, 3.0)),
    "jump_cascade": _ScenarioKind("Jump cascade",
                                  "Jumps arrive the severity times as often, and the average jump grows.",
                                  (2.0, 10.0, 4.0)),
}


class UnknownAssetError(KeyError):
    """The requested ticker is not among the shipped calibrations."""


def _models_for(scenario: str) -> list[str]:
    worlds = SCENARIO_MODELS[scenario]
    return [name for name, lab in LAB_MODELS.items() if lab.world in worlds]


def _has_rho(model: str) -> bool:
    return LAB_MODELS[model].world in SCENARIO_MODELS["correlation_breakdown"]


@cache
def _manifest() -> dict:
    return json.loads((DATA_DIR / "manifest.json").read_text())


def _asset(ticker: str) -> dict:
    entries = {a["ticker"]: a for a in _manifest()["assets"]}
    if ticker not in entries:
        raise UnknownAssetError(ticker)
    return entries[ticker]


@cache
def _calibration(ticker: str, model: str) -> ParamSet:
    entry = _asset(ticker)["models"].get(model)
    if entry is None:
        raise UnknownAssetError(f"{ticker} has no {model} calibration")
    return ParamSet.load(DATA_DIR / entry["file"])


def catalog() -> LabCatalog:
    manifest = _manifest()
    assets = []
    for entry in manifest["assets"]:
        models = {}
        for model, fit in entry["models"].items():
            ps = _calibration(entry["ticker"], model)
            models[model] = ModelFit(params=dict(ps.params), params_hash=ps.content_hash(),
                                     fit_params=fit["fit"]["parameters"], std_errors=fit["fit"]["std_errors"],
                                     derived=fit["fit"]["derived"], note=fit["fit"]["note"])
        assets.append(AssetInfo(
            ticker=entry["ticker"], sector=entry["sector"], vol_today=entry["state"]["vol_today"],
            as_of=entry["state"]["as_of"], dividend_yield=entry["dividend_yield"],
            historical_drift=DriftEstimate(**entry["historical_drift"]), models=models,
        ))
    models = []
    for name, lab in LAB_MODELS.items():
        hedger, note = MODEL_HEDGERS[lab.world]
        models.append(LabModelInfo(name=name, label=lab.label, summary=lab.summary,
                                   hedger_label=HEDGER_LABELS[hedger], hedger_note=note, has_rho=_has_rho(name)))
    scenarios = [ScenarioInfo(name=name, label=kind.label, summary=kind.summary, severity_min=kind.severity[0],
                              severity_max=kind.severity[1], severity_default=kind.severity[2],
                              models=_models_for(name))
                 for name, kind in SCENARIOS.items()]
    return LabCatalog(
        window=manifest["window"], rate=manifest["rate"], rate_source=manifest["rate_source"], assets=assets,
        models=models, scenarios=scenarios, maturities_days=list(MATURITIES_DAYS), n_paths=N_PATHS,
        rho_assumption=RHO_ASSUMPTION,
    )


@dataclass(frozen=True)
class _Source:
    """What a book is built on: a shipped calibration or a visitor's bundle."""

    asset: str
    params: ParamSet
    rate: float
    dividend: float
    origin: BookOrigin


def _shipped(request: HedgeRequest) -> _Source:
    entry = _asset(request.asset)
    ps = _calibration(request.asset, request.model)
    fit = entry["models"][request.model]["fit"]
    origin = BookOrigin(kind="shipped", window=ps.window or _manifest()["window"], as_of=entry["state"]["as_of"],
                        vol_today=entry["state"]["vol_today"],
                        historical_drift=DriftEstimate(**entry["historical_drift"]),
                        method_label=ps.source or "", note=fit.get("note"), calibration_hash=ps.content_hash())
    return _Source(request.asset, ps, float(_manifest()["rate"]), float(entry["dividend_yield"]), origin)


def _visitor(bundle: CalibrationBundle) -> _Source:
    world = validated_world_params(bundle.model, bundle.parameters, bundle.derived)
    # Built directly (not ParamSet.create): stamped with the bundle's date, so the
    # same bundle always gives the same book hash.
    ps = ParamSet(model=LAB_MODELS[bundle.model].world, measure="P", params=MappingProxyType(world),
                  asset=bundle.asset, window=bundle.window, source=f"Visitor calibration: {bundle.method_label}",
                  note=f"calibration bundle {bundle.hash}", created_at=bundle.as_of,
                  mjollnir_version=mjollnir_version)
    vol_today = bundle.derived.get("vol_today") or _world_vol_today(bundle.model, world)
    origin = BookOrigin(kind="visitor", window=bundle.window, as_of=bundle.as_of, vol_today=vol_today,
                        historical_drift=bundle.historical_drift, method_label=bundle.method_label,
                        note=bundle.note, calibration_hash=bundle_hash(bundle.model_dump()))
    return _Source(bundle.asset, ps, bundle.rate, bundle.dividend_yield, origin)


def _world_vol_today(model: str, world: dict) -> float:
    if model == "heston":
        return math.sqrt(world["v0"])
    if model == "rough_bergomi":
        return math.sqrt(world["xi0"])
    if model == "regime_switching_gbm":
        return world[f"sigma_{int(world['regime0'])}"]
    return world.get("sigma", float("nan"))


def _book(request: HedgeRequest, source: _Source) -> tuple[ParamSet, EuropeanOption]:
    book = source.params
    if request.rho is not None:
        if not _has_rho(request.model):
            raise ValueError(f"{LAB_MODELS[request.model].label} has no spot-vol correlation to assume")
        book = book.derive(
            set_={"rho": request.rho},
            note=f"rho assumed {request.rho:g}: daily closes do not identify the spot-vol correlation",
        )
    option = EuropeanOption(
        strike=SPOT0 * request.option.moneyness,
        maturity=request.option.maturity_days * DT,
        is_call=request.option.is_call,
    )
    return book, option


def _scenario(request: HedgeRequest, book: ParamSet):
    if request.scenario is None:
        return None
    name, severity = request.scenario.name, request.scenario.severity
    if request.model not in _models_for(name):
        raise ValueError(f"{SCENARIOS[name].label} has no meaning for {LAB_MODELS[request.model].label}")
    lo, hi, _ = SCENARIOS[name].severity
    if not lo <= severity <= hi:
        raise ValueError(f"{name} severity must be in [{lo}, {hi}]")
    return scenario_for(name, severity, book)


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
        paths=_shown_paths(results),
    )


def _shown_paths(results: dict[str, BacktestResult]) -> SimulatedPaths:
    """The first N_SHOWN paths whole, rounded for the wire (spots to 1e-3, the rest to 1e-4)."""
    first = next(iter(results.values()))
    terminal = first.spots[:N_SHOWN, -1]
    return SimulatedPaths(
        spots=np.round(first.spots[:N_SHOWN], 3).tolist(),
        positions={n: np.round(r.positions[:N_SHOWN], 4).tolist() for n, r in results.items()},
        pnl={n: np.round(r.pnl[:N_SHOWN], 4).tolist() for n, r in results.items()},
        highlight=int(np.argmin(np.abs(terminal - np.median(terminal)))),
    )


def _hedger_infos(model: str, book: ParamSet, sigma: float, learned: bool, cost_bps: float,
                  risk_aversion: float) -> list[HedgerInfo]:
    name, note = MODEL_HEDGERS[LAB_MODELS[model].world]
    # A GBM book priced by its own model has implied vol = sigma: its delta is Black-Scholes delta.
    same = "bs_delta" if model == "gbm" and abs(book.params["sigma"] - sigma) < 1e-6 else None
    infos = [
        HedgerInfo(name="no_hedge", role="none", label="No hedge", note="Keeps the premium and pays the payoff."),
        HedgerInfo(name="bs_delta", role="bs", label="Black-Scholes delta",
                   note=f"Delta at the premium's implied vol ({100 * sigma:.1f}%), fixed for the life of the option."),
        HedgerInfo(name=name, role="model", label=HEDGER_LABELS[name], note=note + ".", same_as=same),
    ]
    if learned:
        infos.append(HedgerInfo(
            name=learned_hedger.NAME, role="learned", label="Learned band",
            note=(f"Follows the {HEDGER_LABELS[name]} but trades only when its position leaves a band a network "
                  f"learned offline for this model, at {cost_bps:g} bp and risk aversion {risk_aversion:g}."),
        ))
    return infos


def run_hedge(request: HedgeRequest) -> HedgeResponse:
    configure_runtime()
    started = time.perf_counter()
    source = _visitor(request.calibration) if request.calibration is not None else _shipped(request)
    rate, dividend = source.rate, source.dividend
    book, option = _book(request, source)
    scenario = _scenario(request, book)
    premium = model_price(option, book, SPOT0, rate=rate, dividend=dividend, dt=DT, seed=request.seed)
    sigma = implied_vol(option, premium, SPOT0, rate, dividend)
    hedgers = standard_hedgers(option, book, dt=DT, spot0=SPOT0, rate=rate, dividend=dividend, premium=premium)
    band = learned_hedger.strategy(option, book, dt=DT, rate=rate, dividend=dividend, cost_bps=request.cost_bps,
                                   risk_aversion=request.risk_aversion)
    if band is not None:
        hedgers[learned_hedger.NAME] = band
    common = {"hedgers": hedgers, "n_paths": N_PATHS, "dt": DT, "spot0": SPOT0, "rate": rate,
              "dividend": dividend, "premium": premium, "cost_bps": request.cost_bps, "seed": request.seed}

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

    factual = _world("Factual", "As calibrated on real prices", book, factual_results, edges, path_index)
    counterfactual = None
    if scenario is not None:
        shocked, shocked_results = worlds["counterfactual"]
        counterfactual = _world(SCENARIOS[request.scenario.name].label, scenario.description,
                                shocked, shocked_results, edges, path_index)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info("lab hedge %s %s (%s) %s in %.0f ms", source.asset, request.model, source.origin.kind,
                getattr(scenario, "name", "factual"), elapsed_ms)
    rho_source = None if not _has_rho(request.model) else ("estimated" if request.rho is None else "assumed")
    return HedgeResponse(
        book=Book(
            asset=source.asset, model=request.model, origin=source.origin, params=dict(book.params),
            params_hash=book.content_hash(),
            parent_hash=book.parent_hash, rho_source=rho_source, strike=option.strike,
            maturity_days=request.option.maturity_days, is_call=option.is_call, premium=premium,
            implied_vol=sigma, rate=rate, dividend_yield=dividend,
        ),
        hedgers=_hedger_infos(request.model, book, sigma, band is not None, request.cost_bps, request.risk_aversion),
        bin_edges={name: e.tolist() for name, e in edges.items()},
        factual=factual, counterfactual=counterfactual,
        n_paths=N_PATHS, seed=request.seed, cost_bps=request.cost_bps,
        elapsed_ms=elapsed_ms, mjollnir_version=mjollnir_version,
    )


def warm_up() -> None:
    """Compile every model's programs at every maturity, so no visitor pays for a
    compile (a scenario reuses its factual world's program)."""
    started = time.perf_counter()
    ticker = _manifest()["assets"][0]["ticker"]
    for model in LAB_MODELS:
        for days in MATURITIES_DAYS:
            run_hedge(HedgeRequest(asset=ticker, model=model, option={"maturity_days": days}))
    logger.info("lab warm-up compiled %d models x %d maturities in %.1f s", len(LAB_MODELS),
                len(MATURITIES_DAYS), time.perf_counter() - started)
