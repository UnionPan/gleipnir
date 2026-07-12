# Design: Extract `mjollnir` — a standalone differentiable market-sim & calibration library

**Date:** 2026-07-12
**Status:** Approved-in-progress (design decisions locked; name chosen)
**Source:** `option_pricer_DLC` monorepo, `src/options_desk/`

> **NAME: `mjollnir`** (Thor's hammer; in DotA the upgrade of *Maelstrom* — chain-lightning
> lineage fitting a stochastic sim). Import root: `mjollnir`, e.g. `from mjollnir.jax import ...`.

## 1. Purpose

Extract five self-contained subpackages from the monorepo into a standalone, pip-installable
Python/JAX library that is a **stable, reproducible, differentiable substrate** for two
downstream consumers that today reach into monorepo internals:

1. **RL research (deep hedging / POMARL).** Needs a jittable, batchable, differentiable
   market simulator whose dynamics do not silently drift between runs. Once the simulator
   is a versioned dependency, experiments pin a version and trust the dynamics are frozen.
2. **Backtesting & calibration.** Sourcing calibrated params (physical- & risk-neutral-measure),
   historical backtests, and synthetic option-chain generation for evaluation.

**Non-goals:** exchange connectivity, tick data, latency guarantees, C++ hot paths. Research-grade,
"at some scale," not production market infrastructure.

## 2. Ground truth from reconnaissance

Measured against the monorepo on 2026-07-12 (not the goal doc's estimates):

| Subpackage    | LOC (measured) | Role |
|---------------|----------------|------|
| `processes`   | 8.1k  | SDE model layer (GBM, Heston, SABR, 3/2, 4/2, Merton, Kou, rough Bergomi, VG, NIG, SLV, regime-switching, Vasicek/CIR/HW), JAX-accelerated |
| `pricer`      | 5.4k  | COS/Fourier + MGF pricing kernels (incl. differentiable JAX pricer) |
| `derivatives` | 1.5k  | Vanilla contract definitions |
| `calibration` | 18.0k | Physical (QMLE, particle filters, GARCH, OHLC vol, NPE) + risk-neutral (Heston-COS, SABR, Dupire, SVI/SSVI) + fetchers + synthetic generators + CLIs |
| `simulations` | 3.2k  | Gym-style Heston/Merton/rough-Bergomi hedging environments |

**Dependency DAG is clean.** The five packages import *only from each other* — verified zero
`options_desk.*` edges to `utils`/`core`/`data`/`pricing`/`deep_hedging`/`pomarl`/`backend`.
The sibling `utils` (2.7k LOC) is **not** used by the bundle and stays in the monorepo.
`core`/`pricing` are empty; top-level `data` is a 9-line stub.

**The one internal knot:** `simulations` imports `calibration.data.synthetic_equity` /
`synthetic_merton_equity` (5 sites across `heston_env.py`, `merton_env.py`, `heston_cache.py`).

**True third-party surface** (measured, supersedes the goal's partial list):
`numpy, scipy, pandas, jax/jaxlib, gymnasium, flax, optax, joblib, zarr, scikit-learn,
yfinance, pandas-datareader, requests, matplotlib`.
Notable: `flax`/`optax` are pulled in by **calibration's NPE**, not just RL — they are core
deps, not optional. `py_vollib` (listed in the goal) is **not imported** by the bundle -> dropped
unless a use surfaces during extraction. `gymnasium` is required by `simulations`.
`torch` is **absent** from the bundle (confirmed) — stays with `deep_hedging`.

## 3. The API contract (derived from real consumers, not invented)

**Mode 1 — JAX simulation kernel (RL substrate). #1 priority.**
`deep_hedging/jax/{rollout,env,pricing}.py` and `deep_hedging/training/torch_buehler.py` import:
- `processes._jax_backend.configure_jax_runtime`
- `processes._jax_qe.qe_heston_step`
- `pricer._jax_fourier_pricer.*` (differentiable COS/Fourier pricing)

**Mode 2 — calibration objects + CLIs (backtesting).**
`scripts/*` (thin wrappers) and `backend/services/calibration_service.py` import:
- `calibration.cli.{build_universes, calibrate, train_npe}` (each exposes `main()`)
- `calibration.physical.heston_particle_filter.HestonParticleFilter`

## 4. Design decisions (locked)

### D1 — Promote the private JAX kernel to a stable public API (#1 priority)
Expose the three load-bearing internals as a first-class public surface:
```python
from mjollnir.jax import qe_heston_step, fourier_price, configure_runtime  # RL substrate
from mjollnir.calibration import HestonParticleFilter, calibrate_universe    # param source
```
`mjollnir.jax` re-exports the (renamed-public) kernels; the underscored modules remain the
implementation but are no longer the contract. Public names are frozen and covered by the
parity test (D5). Internals may be refactored freely as long as public behavior is stable.

### D2 — Resolve the `simulations -> calibration.data` knot: shared `synthetic` layer

Two orthogonal axes both use the word "level" — this fix moves the code favorably on both:
- **Namespace axis:** promote it *up* to a **top-level module `mjollnir.synthetic`** — a sibling
  of `calibration`/`simulations`, no longer *nested* at `calibration.data.synthetic_*`.
- **Dependency axis:** that promotion lets it sit as a **low layer** — depending only on `pricer`,
  depended on by both `simulations` and `calibration` (like `numpy`: top-level to import,
  foundational in the stack). Being nested inside `calibration` today is precisely what forces
  `simulations` to reach *up* into calibration — the knot.

Create a new package **`mjollnir.synthetic`** (top-level in namespace, low in the dependency
DAG) sitting just above `pricer` and below both `simulations` and `calibration`. It holds:
- `synthetic_equity`, `synthetic_merton_equity` (+ `synthetic_crypto`), and
- the `OptionChain` / `OptionQuote` dataclasses (from `data_provider`) they share.

Layering (arrows = "is imported by"):

    pricer  ->  synthetic  ->  simulations
    pricer  ->  calibration
    synthetic -> calibration        (calibration uses synthetic for eval)
    synthetic -> simulations        (simulations uses synthetic, NOT calibration)

**Rationale:** the synthesizer depends only *downward* (on `pricer` + local dataclasses),
is shared by both consumers, and — decisively — **it produces the option chains the RL env
prices and hedges**, so it is load-bearing for env dynamics and must live *inside* the frozen,
versioned substrate, not as an outside eval tool that could drift. The market-data *fetchers*
(yfinance/binance/etc.) are a different concern and remain in `calibration.data`.

### D3 — Transition: dependency + thin re-export shims (no big-bang)
The monorepo pins `mjollnir` as a dependency. `deep_hedging` and `backend` are rewritten to
the new public API. Thin re-export shims remain at the old `options_desk.processes._jax_*` /
`options_desk.calibration.*` paths so nothing breaks mid-refactor. Shims emit a
`DeprecationWarning` and are slated for removal after the RL suite is green on the new API.

### D4 — Fresh-copy extraction
Copy the five packages (+ new `synthetic`) into a new clean git repo (`mjollnir`, sibling of
`option_pricer_DLC`), README noting origin. No `git filter-repo`; monorepo history is
intertwined and not worth dragging for a v0.1 research lib.

### D5 — Reproducibility contract + golden parity test
- Explicit PRNG-key threading; no hidden global RNG/state; deterministic output given a key.
- **Golden-key regression test:** a fixed-key rollout of the Heston QE kernel + differentiable
  pricer whose outputs are asserted bit/tolerance-stable, so a version bump cannot silently
  change env dynamics. This is the acceptance gate that matters more than coverage %.

### D6 — Versioned calibrated-parameter store
Calibration outputs (e.g. per-universe Heston params) serialized as versioned artifacts
(schema + version tag), not ad-hoc pickles, so backtests are reproducible against a param set.

## 5. Package layout (target)

    mjollnir/
      pyproject.toml            # deps = measured set (section 2); extras: [cuda],[metal],[data],[dev]
      README.md                 # two consumption modes: RL kernel vs calibration/backtest
      src/mjollnir/
        __init__.py             # __version__
        jax/__init__.py         # PUBLIC: qe_heston_step, fourier_price, configure_runtime
        processes/              # SDE layer (incl. _jax_backend, _jax_qe internals)
        pricer/                 # COS/Fourier/MGF kernels (incl. _jax_fourier_pricer internal)
        derivatives/
        synthetic/              # NEW: synthetic_* generators + OptionChain/OptionQuote
        calibration/            # physical + risk_neutral + data(fetchers) + cli
        simulations/            # gym envs
      tests/
        test_parity_golden.py   # D5 acceptance gate
        ...

Package name `mjollnir`, `requires-python >= 3.10` (jax 0.9 baseline), setuptools build.

## 6. Extraction order (bottom-up, matches the DAG)
1. `processes`  -> 2. `pricer` + `derivatives` -> 3. `synthetic` (new; carve from calibration.data)
-> 4. `calibration/risk_neutral` -> 5. `calibration/physical` -> 6. `calibration/{data,cli}`
-> 7. `simulations` -> 8. `jax` public facade -> 9. parity test -> 10. monorepo shims + consumer rewrite.

## 7. Definition of done (v0.1)
- [ ] New `mjollnir` repo; `pip install -e .` works in a clean env.
- [ ] Five subpackages + `synthetic` extracted; imports rewritten to `mjollnir.*`.
- [ ] `pyproject.toml` scoped to the measured dep set; JAX `[cuda]`/`[metal]` extras preserved; no torch/web deps.
- [ ] Public `mjollnir.jax` exposing Heston step + differentiable pricer + runtime config.
- [ ] `simulations -> calibration.data` knot dissolved via `mjollnir.synthetic`.
- [ ] Golden-key parity test green (D5).
- [ ] Monorepo `deep_hedging` consumes `mjollnir` (or via shim); `test_pomarl_*` suite passes.
- [ ] README documents both consumption modes.

## 8. Open items
- Final package name (user's pick).
- Confirm `py_vollib` truly unused during extraction (drop if so).
- Whether `synthetic_crypto` + crypto fetchers are in scope for v0.1 or deferred.
