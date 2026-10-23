# Gleipnir repositioning and mjollnir consolidation

Date: 2026-10-23.
Status: Phase 0 approved for execution; later phases are designed here and detailed in their own plans.

## 1. Positioning

The two repos have distinct, user-confirmed roles.

**mjollnir** is the portable JAX substrate for quant research: differentiable, reproducible simulation, calibration, pricing, scenarios, and backtesting.
It keeps library discipline: semantic versioning, golden parity tests, and a documentation site.
Its examples also make it a secondary learning tool.

**gleipnir** (formerly `option_pricer_DLC`) is a learning tool built from proof-of-concept demos for market calibration and control/trading.
Each demo is a complete loop on real data: calibrate, choose a counterfactual scenario, train or evaluate a controller, and compare against a classical baseline.
Each demo explains why its result looks the way it does and names the mjollnir calls that produced it.
The demos are served on the portfolio site (frontend `app.yunianpan.com`, API `api.yunianpan.com`).

The name pairs with mjollnir: in Dota 2 both items upgrade from Maelstrom, and in Norse myth Gleipnir is the fetter that bound Fenrir.

### Boundary rule

Anything reusable lives in mjollnir: environments, backtest primitives, pricers, calibrators, scenario transforms.
gleipnir keeps only demo glue, agents and training code, trained artifacts, web pages, and explanations.
The paper's research code stays in `aispg`; gleipnir showcases its results through exported artifacts.

## 2. Findings that motivate this work (2026-10-22)

1. The live Deep RL Hedging page was fabricated: every "agent" was Black-Scholes delta scaled by a constant plus noise.
   Removed and deployed in `e44e2dc` before any other work.
2. `src/options_desk` carried 34.5k lines of stale pre-v0.2 copies of mjollnir subpackages (`calibration`, `processes`, `pricer`, `derivatives`, `simulations`).
   Only three JAX kernel modules were shims, and mjollnir was not a declared dependency.
   The only code unique to this repo is `deep_hedging` (9.2k lines) and `utils` (2.7k lines).
3. The local commits that switch the kernel to mjollnir were never deployed, because the backend image does not install mjollnir.
4. `/api/v1/smile/compare` silently returns the market IV as the Heston and Merton model IV.
   `implied_volatility_from_price` imports `options_desk.pricing` and `options_desk.core`, which were deleted long ago, and a blanket `except Exception` substitutes the market value.
   Reproduced on production: 408 of 408 model IVs equal the market IV.
5. Backend CI runs `pytest ... || true`, so tests have never gated a merge or deploy.
6. The Backtesting page reimplements a hedging backtest in a 1,370-line NumPy loop with global `np.random.seed`, duplicating `mjollnir.backtest` and `mjollnir.scenarios`.
7. In this repo's RL results, the LSTM Buehler hedger beats Black-Scholes delta (hedging-error std 0.82 vs 1.70), while the POMARL and PPO agents diverge once options are tradable.

## 3. Phase 0: foundation (approved)

### D1. mjollnir as a pinned dependency

gleipnir declares `mjollnir` in `pyproject.toml` and resolves it from GitHub through `[tool.uv.sources]`, pinned by `uv.lock` to an exact commit.
The backend image installs from the lockfile, so production and CI run the same mjollnir.
Local co-development of both repos overrides the pin with an editable install of `../mjollnir`.
mjollnir currently has no git tags.
Tagging its current release (the version in mjollnir's `pyproject.toml`) is an outward-facing step for the user to approve, after which the source pins the tag.

### D2. uv replaces the requirements files

`pyproject.toml` plus `uv.lock` replace `requirements*.txt` and `backend/requirements.txt`.
JAX device choices become extras, matching mjollnir.
The local environment becomes a uv-managed `.venv`, replacing the conda `options-desk` env, which has a stale editable path and conflicting CUDA 12 and CUDA 13 JAX plugins.

### D3. Rename the package and delete duplicates

`src/options_desk` becomes `src/gleipnir`, keeping `deep_hedging` and `utils` with their internal structure unchanged.
The five duplicated subpackages and the empty `core`, `pricing`, and `data` packages are deleted, and every import moves to the corresponding `mjollnir.*` path.
Console entry points for calibration already exist in mjollnir (`mjollnir-calibrate` and friends) and are removed here.
Restructuring `deep_hedging` internals is out of scope for this phase.

### D4. Backend on mjollnir

Backend services import mjollnir directly.
`backend/services/advanced_pricing.py` (duplicate Heston, SABR, and Merton pricers) is replaced by mjollnir pricers plus one IV inversion that reports failure explicitly.
The smile service stops substituting market IV on failure: an IV that cannot be inverted is returned as `null` and the response says why.
The backtesting service is not rewritten here (Phase 1); it only switches imports.

### D5. CI gates for real

Backend CI drops `|| true`, runs the full test suite, and lints with ruff.
Failures found by turning the gate on are fixed, not skipped.

### D6. Branding and metadata

The site brand changes from "AIS Opt" to "Gleipnir"; the FastAPI title, README, and package metadata follow.
The README is rewritten around the positioning in section 1.
Package metadata gets the correct author email and repository URLs, and tool configs target Python 3.12.
Cloud Run service names (`options-desk-*`) and domains stay unchanged.

### Phase 0 definition of done

- `uv sync` builds a clean environment; `import gleipnir` and `import backend.main` succeed.
- No module under `src/gleipnir` duplicates a mjollnir module.
- The full test suite passes and gates CI.
- `/api/v1/smile/compare` returns model IVs that differ from market IVs, verified end to end against a locally running backend.
- The backend image builds with mjollnir installed, and the deployed API passes the same end-to-end checks.

## 4. Later phases (designed, not yet approved)

### Phase 1: counterfactual backtesting

The Backtesting page is rebuilt on `mjollnir.backtest` and `mjollnir.scenarios`.
The flow is: calibrated `ParamSet` from real data, a named counterfactual scenario (vol spike, jump cascade, regime shift, correlation breakdown), a batched simulation, and classical hedges compared across the factual and counterfactual worlds.
Results carry mjollnir's provenance hash, so every chart on the page is reproducible.

### Phase 2: real RL demos

Two trained controllers replace the removed fake page.
The LSTM Buehler deep hedger from this repo is compared against Black-Scholes delta, delta-gamma, and delta-gamma-vega hedges.
The AIS cliquet-hedging agent from `aispg` is shown with its reactive, frozen-AIS, and exact-state references, and the page explains approximate information states.
Whether each demo serves live inference or precomputed evaluation artifacts is decided in that phase's plan, weighing backend image size and cold-start latency.

### Phase 3: mjollnir polish

Release tagging, coverage beyond the current 35% gate, the open TODOs (for example `rmse_iv` in `calibration/risk_neutral/heston_calibrator.py`), and widening the `jax<0.10` pin now that 0.11 is stable.
Work here is prioritized by what the gleipnir demos need.
