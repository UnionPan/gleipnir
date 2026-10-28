"""The lab's learned hedger: a no-trade band around each model's delta, trained offline.

Networks are trained per world by ``python -m gleipnir.lab_hedger.cli`` on the
lab's shipped books and shipped under ``backend/data/lab/hedgers/<world>/``
with their validation report and weights' SHA-256. A book whose model has no
trained network simply runs without the learned hedger.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from gleipnir.lab_hedger import band_strategy
from gleipnir.lab_hedger.artifact import load
from mjollnir.backtest import ParametricStrategy
from mjollnir.hedging import EuropeanOption, model_delta
from mjollnir.params import ParamSet

HEDGERS_DIR = Path(__file__).resolve().parents[1] / "data" / "lab" / "hedgers"
NAME = "learned_band"


@cache
def _network(world: str) -> tuple[dict, dict] | None:
    directory = HEDGERS_DIR / world
    return load(directory) if (directory / "params.msgpack").exists() else None


def available(world: str) -> bool:
    return _network(world) is not None


def strategy(option: EuropeanOption, book: ParamSet, *, dt: float, rate: float, dividend: float, cost_bps: float,
             risk_aversion: float) -> ParametricStrategy | None:
    """The band hedger for ``book``, or None when its model has no trained network."""
    trained = _network(book.model)
    if trained is None:
        return None
    net, _ = trained
    base = model_delta(option, book, dt, rate, dividend)
    return ParametricStrategy(band_strategy(base.fn), {"base": base.params, "net": net, "cost_bps": float(cost_bps),
                                                      "risk_aversion": float(risk_aversion)})


def report(world: str) -> dict | None:
    trained = _network(world)
    return None if trained is None else trained[1]
