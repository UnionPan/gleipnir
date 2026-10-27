#!/usr/bin/env python3
"""Empirical-Bayes sector priors for the Bayesian fits on the Calibration tab.

Every S&P 500 stock's five-year history is read by each model's posterior
network (backend/data/npe/<model>/) under the broad training prior. A sector's
prior is then the Gaussian (in the network's unconstrained coordinates) that
best explains its stocks' histories together: the marginal likelihood
prod_i E_{posterior_i}[N(u; m, S)] is maximized over (m, S) by Monte Carlo EM,
whose E-step reweights each stock's posterior draws by the current prior.

So a visitor's fit starts from what the sector's stocks say and is narrowed
by the asset's own history: a weakly identified parameter (Heston's kappa,
regime durations) borrows strength from the sector, a well identified one
(the level of volatility) follows the asset.

Small sectors are shrunk toward the universe's prior (a pseudo-count of
SHRINK_STOCKS stocks), and every prior keeps a floor on its spread so that no
sector becomes dogmatic. The prior's truncation to the training box is
ignored in the EM, which is exact while the sector's prior sits inside it.
A stock whose posterior falls mostly outside the training box is one the
model does not describe; it is left out of that model's priors and counted.

Inputs (local, not committed): data/price_lake/prices/<TICKER>.parquet and
the GICS sectors in runs/calibration/sp500-batched/heston_qmle.parquet.
Output: backend/data/npe/sector_priors.json.

Usage (GPU): ``MJOLLNIR_JAX_BACKEND=cuda uv run --extra web --extra cuda python scripts/fit_sector_priors.py``
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from mjollnir.inference import load_artifact
from mjollnir.inference.posterior import MAX_LEN
from mjollnir.inference.summaries import pad_returns, summary_features
from mjollnir.inference.train import posterior_draws

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.services.bayesian_fits import MIN_DAYS, NPE_DIR, NPE_NAME  # noqa: E402

PRICES = ROOT / "data" / "price_lake" / "prices"
SECTORS = ROOT / "runs" / "calibration" / "sp500-batched" / "heston_qmle.parquet"
DRAWS = 2048
EM_ITERATIONS = 200
EM_TOLERANCE = 1e-6
SHRINK_STOCKS = 10.0
FLOOR_SHARE = 0.05       # a prior's sd stays above 5% of the training box's width
SUMMARY_DRAWS = 20_000
MIN_IN_BOX = 0.05        # a stock whose posterior mostly leaves the box is one the model does not describe


def load_histories() -> tuple[list[str], list[np.ndarray], dict[str, str], str]:
    sectors = pd.read_parquet(SECTORS, columns=["ticker", "sector"]).dropna()
    tickers, histories, first, last = [], [], [], []
    for ticker in sorted(sectors["ticker"]):
        path = PRICES / f"{ticker}.parquet"
        if not path.exists():
            continue
        close = pd.read_parquet(path, columns=["close"])["close"].dropna()
        close = close[close > 0]
        returns = np.diff(np.log(close.to_numpy(dtype=float)))[-MAX_LEN:]
        if returns.size < MIN_DAYS:
            continue
        tickers.append(ticker)
        histories.append(returns)
        first.append(close.index[-returns.size - 1])
        last.append(close.index[-1])
    window = f"{min(first):%Y-%m-%d}..{max(last):%Y-%m-%d}"
    return tickers, histories, dict(zip(sectors["ticker"], sectors["sector"], strict=True)), window


def em_gaussian(u: np.ndarray, inside: np.ndarray, start_mean: np.ndarray, start_cov: np.ndarray,
                floor: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Gaussian hyperprior maximizing prod_i mean_j N(u_ij; m, S) over in-box draws."""
    mean, cov = start_mean.copy(), start_cov.copy()
    for iteration in range(1, EM_ITERATIONS + 1):
        diff = u - mean
        solved = np.linalg.solve(cov, diff.reshape(-1, diff.shape[-1]).T).T.reshape(diff.shape)
        log_w = np.where(inside, -0.5 * np.sum(diff * solved, axis=-1), -np.inf)
        w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        new_mean = np.einsum("ij,ijd->d", w, u) / len(u)
        centred = u - new_mean
        new_cov = np.einsum("ij,ijd,ije->de", w, centred, centred) / len(u)
        new_cov += np.diag(floor**2)
        change = np.max(np.abs(new_mean - mean)) + np.max(np.abs(new_cov - cov))
        mean, cov = new_mean, new_cov
        if change < EM_TOLERANCE:
            return mean, cov, iteration
    return mean, cov, EM_ITERATIONS


def natural_summary(model, mean: np.ndarray, cov: np.ndarray, key) -> dict:
    """The prior's 5/50/95% quantiles in natural units (truncated to the training box)."""
    z = jax.random.multivariate_normal(key, jnp.asarray(mean), jnp.asarray(cov), (SUMMARY_DRAWS,))
    z = np.asarray(z)[model.in_prior(np.asarray(z))]
    x = np.asarray(model.to_natural(jnp.asarray(z)))
    q = np.quantile(x, [0.05, 0.5, 0.95], axis=0)
    return {name: {"q05": float(q[0, k]), "median": float(q[1, k]), "q95": float(q[2, k])}
            for k, name in enumerate(model.params)}


def main() -> int:
    tickers, histories, sector_of, window = load_histories()
    print(f"{len(tickers)} stocks, window {window}, devices {jax.devices()}")
    padded, mask = pad_returns(histories, MAX_LEN)
    features = np.asarray(summary_features(jnp.asarray(padded), jnp.asarray(mask)))
    finite = np.all(np.isfinite(features), axis=1)
    sector = np.array([sector_of[t] for t in tickers])
    out = {"universe": "S&P 500", "window": window, "n_stocks": int(finite.sum()),
           "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "method": ("Monte Carlo EM for a Gaussian hyperprior in each network's unconstrained coordinates; "
                      f"sectors shrunk toward the universe with {SHRINK_STOCKS:g} pseudo-stocks; "
                      f"sd floor {FLOOR_SHARE:.0%} of the training box"),
           "tickers": {t: s for t, s, ok in zip(tickers, sector, finite, strict=True) if ok},
           "models": {}}
    for name in NPE_NAME.values():
        artifact = load_artifact(NPE_DIR / name)
        model = artifact.model
        u = posterior_draws(artifact, features[finite], jax.random.PRNGKey(0), DRAWS)
        inside = model.in_prior(u.reshape(-1, model.dim)).reshape(u.shape[:2])
        share = inside.mean(axis=1)
        keep = share >= MIN_IN_BOX
        dropped = [t for t, ok in zip(np.asarray(tickers)[finite], keep, strict=True) if not ok]
        print(f"{name}: in-box share median {np.median(share):.2f}, 5th pct {np.quantile(share, 0.05):.2f}; "
              f"left out {len(dropped)}: {' '.join(dropped[:20])}")
        u, inside, model_sector = u[keep], inside[keep], sector[finite][keep]
        width = np.asarray(model.high) - np.asarray(model.low)
        floor = FLOOR_SHARE * width
        box_mean, box_cov = (np.asarray(model.high) + np.asarray(model.low)) / 2, np.diag(width**2 / 12)
        u_mean, u_cov, iters = em_gaussian(u, inside, box_mean, box_cov, floor)
        print(f"{name}: universe EM {iters} iterations, prior sd {np.sqrt(np.diag(u_cov)).round(3)}")
        priors = {"_universe": {"n": len(u), "mean": u_mean.tolist(), "cov": u_cov.tolist(),
                                "natural": natural_summary(model, u_mean, u_cov, jax.random.PRNGKey(1)),
                                "left_out": dropped}}
        for k, s in enumerate(sorted(set(model_sector))):
            rows = model_sector == s
            n = int(rows.sum())
            m_hat, c_hat, iters = em_gaussian(u[rows], inside[rows], u_mean, u_cov, floor)
            mean = (n * m_hat + SHRINK_STOCKS * u_mean) / (n + SHRINK_STOCKS)
            cov = (n * c_hat + SHRINK_STOCKS * u_cov) / (n + SHRINK_STOCKS)
            priors[s] = {"n": n, "mean": mean.tolist(), "cov": cov.tolist(),
                         "natural": natural_summary(model, mean, cov, jax.random.PRNGKey(2 + k))}
            medians = {p: round(v["median"], 3) for p, v in priors[s]["natural"].items()}
            print(f"  {s:24s} n={n:3d} EM {iters:3d}  medians {medians}")
        out["models"][name] = priors
    path = NPE_DIR / "sector_priors.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
