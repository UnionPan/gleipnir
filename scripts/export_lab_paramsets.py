#!/usr/bin/env python3
"""Export real-data ParamSets of every lab model for the Counterfactual Hedging Lab.

Inputs (local artifacts, not committed):

* ``runs/calibration/sp500-batched/``: the universe run's window (manifest) and
  GICS sectors (heston_qmle.parquet).
* ``data/price_lake/prices/<TICKER>.parquet``: daily adjusted closes.

Each asset is fitted with every model the lab runs, by the Calibration tab's
default (``backend.services.physical_fits``): the Bayesian fit under the
asset's sector prior, whose posterior means are the book's parameters. Each
fit is mapped onto its mjollnir world (``backend.services.lab_models``). The
drift is left out: books follow the drift policy and drift at r - q. The
historical drift is kept per asset as the educational estimate the market
card shows.

Rates and dividends: the 13-week Treasury bill yield (^IRX) on the window's
last day, and each asset's dividend yield from yfinance (needs the network).

Outputs, under ``backend/data/lab/``: ``<TICKER>/<model>.json`` ParamSets and
``manifest.json``.

Usage: ``uv run --extra web python scripts/export_lab_paramsets.py``
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from mjollnir.params import ParamSet

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.services import physical_fits  # noqa: E402
from backend.services.lab_models import LAB_MODELS, world_params  # noqa: E402
from backend.services.market_data import fetch_dividend_yield  # noqa: E402

RUN = ROOT / "runs" / "calibration" / "sp500-batched"
PRICES = ROOT / "data" / "price_lake" / "prices"
OUT = ROOT / "backend" / "data" / "lab"

ASSETS = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM", "XOM", "JNJ", "KO"]
REALIZED_WINDOW = 21
DT = 1.0 / 252


def _closes(ticker: str, start: str, end: str) -> pd.Series:
    frame = pd.read_parquet(PRICES / f"{ticker}.parquet")
    closes = frame["adj_close"].copy()
    closes.index = pd.to_datetime(frame.index)
    return closes.sort_index().loc[start:end].dropna()


def _treasury_bill_rate(on: str) -> float:
    """Continuously compounded 13-week T-bill yield on (or just before) ``on``."""
    day = date.fromisoformat(on)
    history = yf.Ticker("^IRX").history(start=(day - timedelta(days=10)).isoformat(),
                                        end=(day + timedelta(days=1)).isoformat())
    if history.empty:
        raise SystemExit(f"no ^IRX quote near {on}")
    discount_yield = float(history["Close"].iloc[-1]) / 100.0
    return math.log1p(discount_yield)


def _finite(d: dict | None) -> dict:
    return {k: v for k, v in (d or {}).items() if v is not None and math.isfinite(v)}


def main() -> int:
    run = json.loads((RUN / "manifest.json").read_text())
    start, end = run["start"], run["end"]
    window = f"{start}..{end}"
    rate = _treasury_bill_rate(end)
    sectors = pd.read_parquet(RUN / "heston_qmle.parquet").set_index("ticker")["sector"]

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    assets = []
    for ticker in ASSETS:
        closes = _closes(ticker, start, end)
        prices, dates = closes.to_numpy(), [d.strftime("%Y-%m-%d") for d in closes.index]
        log_returns = np.diff(np.log(prices))
        vol_today = float(math.sqrt(log_returns[-REALIZED_WINDOW:].var(ddof=1) / DT))
        models, drift = {}, None
        for model, lab in LAB_MODELS.items():
            parameters, diagnostics, _ = physical_fits.fit(model, None, prices, dates, include_drift=False,
                                                           ticker=ticker, prior="sector")
            if diagnostics.get("converged") is False:
                raise SystemExit(f"{ticker} {model}: the fit did not converge")
            drift = diagnostics["drift"]
            ps = ParamSet.create(
                lab.world, "P", world_params(model, parameters, diagnostics.get("derived")),
                asset=ticker, window=window,
                source=f"{diagnostics['method_label']} on daily closes; drift per the policy (r - q)",
                note="Exported for the gleipnir Counterfactual Hedging Lab.",
            )
            path = OUT / ticker / f"{model}.json"
            ps.save(path)
            models[model] = {
                "file": f"{ticker}/{model}.json",
                "hash": ps.content_hash(),
                "fit": {"parameters": parameters, "std_errors": _finite(diagnostics.get("std_errors")),
                        "derived": _finite(diagnostics.get("derived")), "note": diagnostics.get("note")},
            }
        sector = str(sectors.get(ticker, "Unknown"))
        assets.append({
            "ticker": ticker,
            "sector": sector,
            "dividend_yield": fetch_dividend_yield(ticker),
            "state": {"as_of": end, "vol_today": vol_today, "window_days": REALIZED_WINDOW},
            "historical_drift": drift,
            "n_observations": int(log_returns.size),
            "models": models,
        })
        print(f"  {ticker:5s} vol today {vol_today:.2f}  q {assets[-1]['dividend_yield']:.4f}")

    manifest = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": window,
        "rate": rate,
        "rate_source": f"13-week Treasury bill (^IRX) on {end}, continuously compounded",
        "assets": assets,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(assets)} assets x {len(LAB_MODELS)} models to {OUT.relative_to(ROOT)}; r = {rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
