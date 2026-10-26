"""Calibration bundles: a Calibration-tab fit, carried to the Hedging Lab by the visitor's browser.

A bundle is everything the lab needs to run a book on the visitor's own
calibration: the asset, the model and its fitted parameters, today's state
(variance, regime), the rate and dividend yield at fit time, the window and
the drift estimate. Its hash is a SHA-256 over its canonical JSON, so the
receipt on the Calibration tab and the book in the lab show the same
fingerprint, and any edit in transit changes it.

The browser holds the bundle, so the lab trusts nothing in it: every
parameter is bounded to what the worlds simulate well.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from backend.core.config import settings
from backend.services.lab_models import LAB_MODELS, world_params
from backend.services.market_data import todays_dividend_yield, treasury_bill_rate

HASHED_FIELDS = ("asset", "model", "parameters", "derived", "std_errors", "historical_drift", "window", "as_of",
                 "rate", "dividend_yield", "method_label", "note")
MAX_REGIMES = 3

# World-parameter bounds: what the lab's worlds and hedgers handle without loss of accuracy.
BOUNDS = {
    "sigma": (0.01, 3.0), "kappa": (0.01, 300.0), "theta": (1e-4, 4.0), "sigma_v": (0.01, 10.0),
    "rho": (-0.99, 0.99), "v0": (1e-4, 4.0), "lambda_j": (0.0, 300.0), "mu_j": (-1.0, 1.0),
    "sigma_j": (1e-3, 1.0), "omega": (1e-12, 1e-2), "alpha": (1e-6, 1.0), "beta": (1e-6, 1.0),
    "h0": (1e-8, 0.02), "xi0": (1e-4, 4.0), "eta": (0.01, 5.0), "H": (0.01, 0.5),
}


def bundle_hash(bundle: dict) -> str:
    content = {k: bundle.get(k) for k in HASHED_FIELDS}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_bundle(ticker: str, model: str, parameters: dict, diagnostics: dict, window: str, as_of: str) -> dict | None:
    """The bundle for a fit, or None for a model the lab does not run (OU)."""
    if model not in LAB_MODELS:
        return None
    bundle = {
        "asset": ticker.upper(),
        "model": model,
        "parameters": {k: float(v) for k, v in parameters.items() if k != "mu"},
        "derived": {k: v for k, v in (diagnostics.get("derived") or {}).items() if v is not None},
        "std_errors": {k: v for k, v in (diagnostics.get("std_errors") or {}).items() if v is not None},
        "historical_drift": diagnostics.get("drift"),
        "window": window,
        "as_of": as_of,
        "rate": treasury_bill_rate(settings.DEFAULT_RISK_FREE_RATE),
        "dividend_yield": todays_dividend_yield(ticker),
        "method_label": diagnostics.get("method_label") or "",
        "note": diagnostics.get("note"),
    }
    bundle["hash"] = bundle_hash(bundle)
    return bundle


def validated_world_params(model: str, parameters: dict, derived: dict) -> dict:
    """The bundle's world parameters, refused if any falls outside what the lab simulates."""
    world = world_params(model, parameters, derived)
    for name, value in world.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} is not a finite number")
        key = "sigma" if name.startswith("sigma_") and name[6:].isdigit() else name
        if key in BOUNDS:
            lo, hi = BOUNDS[key]
            if not lo <= value <= hi:
                raise ValueError(f"{name} = {value:g} is outside the lab's range [{lo:g}, {hi:g}]")
    if model == "regime_switching_gbm":
        n = sum(1 for k in world if k.startswith("sigma_") and k[6:].isdigit())
        if not 1 <= n <= MAX_REGIMES:
            raise ValueError(f"the lab runs 1 to {MAX_REGIMES} regimes, not {n}")
        rows = np.array([[world[f"p_{i}{j}"] for j in range(n)] for i in range(n)])
        if np.any(rows < 0) or not np.allclose(rows.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("regime transition probabilities must be non-negative and sum to 1 by row")
    if model == "garch" and world["alpha"] + world["beta"] >= 0.999:
        raise ValueError("GARCH persistence alpha + beta must be below 0.999")
    return world
