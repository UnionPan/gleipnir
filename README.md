# Gleipnir

A learning lab of proof-of-concept demos for market calibration and control.
Each demo is a complete loop on real data: calibrate a market model, choose a counterfactual world, run a controller through it, and compare against a classical baseline.
The pages explain why each result looks the way it does.

Live at [app.yunianpan.com](https://app.yunianpan.com).
The numerics come from [mjollnir](https://github.com/UnionPan/mjollnir), a JAX library for differentiable, reproducible market simulation, calibration and pricing.
Gleipnir keeps only what is specific to the demos: the web app, the agents and training code, and the shipped calibrations.

## Demos

**Counterfactual Hedging Lab.**
Sell an option on a real S&P 500 name, priced and hedged under one of six calibrated models (Heston, Black-Scholes, Merton jumps, regime switching, GARCH, rough Bergomi), then run the same book through the factual world and a stressed one (volatility spike, correlation breakdown, regime shift, jump cascade, wherever the model gives it a meaning).
No hedge, Black-Scholes delta and the model's own delta are compared on 2,048 common paths per world, with the spot drifting at r - q, and every result traceable to the parameter set that produced it.

**Calibration.**
Fit physical-measure models (GBM, OU, Heston, rough Bergomi, Merton, GARCH, regime-switching GBM) to price history, and risk-neutral models (Heston, SABR, Dupire) to live option chains.

**Market.**
Indices, the largest US stocks and commodity futures, from Yahoo Finance.

## Run it locally

Requirements: [uv](https://docs.astral.sh/uv/), Node.js 24, and optionally an NVIDIA GPU.
mjollnir is installed from the commit pinned in `uv.lock`; a checkout at `../mjollnir` is picked up automatically for co-development.

```bash
make install   # Python env (uv) and frontend dependencies (npm)
make dev       # API on :8000 and web app on :3000 with live reload; Ctrl-C stops both
```

`make dev` prints the URLs once both servers answer, including the SSH-tunnel command when you are connected remotely.
See [docs/LOCAL_SETUP.md](/docs/LOCAL_SETUP.md) for details and troubleshooting.

## Tests and lint

```bash
make test-backend    # pytest (live-market-data tests are opt-in: uv run pytest -m network)
make test-frontend   # Jest
make lint            # ruff
cd frontend && npm run lint
```

## Layout

| Path | Contents |
|------|----------|
| `backend/` | FastAPI app: lab, calibration, market-data and pricing endpoints |
| `backend/data/lab/` | Shipped calibrations of every lab model for eight assets (regenerate with `scripts/export_lab_paramsets.py`) |
| `frontend/` | React and TypeScript web app |
| `src/gleipnir/deep_hedging/` | Hedging agents (LSTM, Transformer, POMARL), training loops, JAX rollouts, baselines |
| `scripts/` | Research experiments, `dev.py` (the `make dev` runner) |
| `deployment/` | Dockerfiles and Cloud Build configs |
| `docs/` | Setup, deployment and API notes; design specs |

## Deployment

Pushes to `main` build the images with Cloud Build and deploy them to Cloud Run.
See [docs/DEPLOYMENT.md](/docs/DEPLOYMENT.md).

## License

MIT
