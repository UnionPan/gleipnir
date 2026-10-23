#!/usr/bin/env python3
"""Export real-data Heston ParamSets for the Counterfactual Hedging Lab.

Inputs (local artifacts, not committed):

* ``runs/calibration/sp500-batched/heston_qmle.parquet`` from mjollnir's
  calibration pipeline (``mjollnir-calibrate``): P-measure Heston QMLE on five
  years of daily closes for the S&P 500.
* ``data/price_lake/prices/<TICKER>.parquet``: the same daily price history.

Outputs, under ``backend/data/lab/``:

* one mjollnir ``ParamSet`` JSON per asset;
* ``manifest.json`` with the per-asset metadata the lab displays.

Parameters vs state. QMLE estimates the dynamics (kappa, theta, sigma_v, mu),
but the lab simulates forward from *today*, so ``v0`` is replaced by today's
variance, taken as the trailing 21-day realized variance. It is model-free and
stable; a Heston particle filter's end-of-window estimate was evaluated and
rejected because it moves by up to 2x on a single large daily return.

Correlation. Daily closes do not identify the spot-variance correlation rho:
QMLE returns rho ~ 0 for every name (an amortized NPE posterior returns its
prior instead). The ParamSets keep the QMLE value; the lab exposes rho as an
explicit, labelled assumption derived from these sets at request time.

Usage: ``uv run python scripts/export_lab_paramsets.py``
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from mjollnir.params import ParamSet

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "calibration" / "sp500-batched"
PRICES = ROOT / "data" / "price_lake" / "prices"
OUT = ROOT / "backend" / "data" / "lab"

ASSETS = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM", "XOM", "JNJ", "KO"]
DYNAMICS = ["kappa", "theta", "sigma_v", "rho", "mu"]
DT = 1.0 / 252
REALIZED_WINDOW = 21


def _closes(ticker: str, start: str, end: str) -> pd.Series:
    frame = pd.read_parquet(PRICES / f"{ticker}.parquet")
    closes = frame["adj_close"].copy()
    closes.index = pd.to_datetime(frame.index)
    return closes.sort_index().loc[start:end].dropna()


def main() -> int:
    manifest_in = json.loads((RUN / "manifest.json").read_text())
    start, end = manifest_in["start"], manifest_in["end"]
    window = f"{start}..{end}"
    table = pd.read_parquet(RUN / "heston_qmle.parquet").set_index("ticker")
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.json"):
        stale.unlink()

    assets = []
    for ticker in ASSETS:
        row = table.loc[ticker]
        dynamics = {k: float(row[k]) for k in DYNAMICS}
        if not bool(row["converged"]) or not all(map(math.isfinite, dynamics.values())):
            raise SystemExit(f"{ticker}: QMLE did not converge or is not finite")

        log_returns = np.diff(np.log(_closes(ticker, start, end).to_numpy()))
        v_today = float(log_returns[-REALIZED_WINDOW:].var(ddof=1) / DT)

        ps = ParamSet.create(
            "heston", "P", {"v0": v_today, **dynamics},
            asset=ticker, window=window,
            source=(
                "dynamics: mjollnir heston_qmle on yfinance daily closes; "
                f"v0: trailing {REALIZED_WINDOW}-day realized variance at {end}"
            ),
            note="Exported for the gleipnir Counterfactual Hedging Lab.",
        )
        filename = f"{ticker}.json"
        ps.save(OUT / filename)
        assets.append({
            "ticker": ticker,
            "sector": str(row["sector"]),
            "file": filename,
            "hash": ps.content_hash(),
            "state": {"as_of": end, "vol_today": math.sqrt(v_today), "window_days": REALIZED_WINDOW},
            "diagnostics": {
                "feller_ratio": float(row["feller_ratio"]),
                "variance_proxy_r2": float(row["variance_proxy_r2"]),
                "n_observations": int(row["n_observations"]),
            },
        })

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": window,
        "estimator": "Quasi-MLE on daily closes (mjollnir heston_qmle)",
        "assets": assets,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(assets)} ParamSets to {OUT.relative_to(ROOT)}")
    for a in assets:
        print(f"  {a['ticker']:5s} vol today {a['state']['vol_today']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
