# Calibration-to-Lab pipeline: design

Date: 2026-10-24.
Status: approved direction; the user asked for autonomous execution.
Repos: gleipnir (site, API, demos) and mjollnir (library).

## Intent

A visitor picks an asset, calibrates any of the site's models to real data, and gets an honest calibration receipt.
The receipt saves itself as a bundle that the Counterfactual Hedging Lab picks up, so the visitor can sell an option, hedge it, and ask "what if the market had turned out differently?" on their own calibration.
Everything the visitor sees should be defensible to a quant: estimates come with uncertainty, weakly identified quantities say so, and nothing is presented as fitted that was not fitted.

## Decisions made by the user

### Drift policy

- Up to 3 months: excess drift defaults to zero, so the spot drifts at r - q.
- 3 months to 2 years: zero excess drift plus explicit positive and negative drift scenarios.
- Beyond 2 years: a shrunk factor, equilibrium or regime model with wide uncertainty bands.
- Historical drift is shown only as an educational estimate with its standard error and confidence interval.
- An individual stock's raw sample mean is never presented as a reliable calibration.
- RL training randomizes excess drift around zero and tests policy robustness across drift regimes.

### Pipeline

1. Offline prior construction: mjollnir calibrates a broad equity universe, pools estimates globally and by sector, and publishes versioned prior artifacts.
   This expensive work is never publicly triggered.
2. The visitor selects an asset; gleipnir loads its history, its current option chain and its sector prior.
3. Bounded calibration update: history updates the P-measure posterior of the real-world dynamics; option prices calibrate the Q-measure model used for pricing and hedging; excess drift defaults to zero.
4. Calibration receipt: estimates, uncertainty, fit diagnostics, data provenance, assumptions and an artifact hash.
5. Handoff bundle: asset, P posterior, Q calibration, drift policy, market snapshot, diagnostics and version hashes.
6. Lab experiment: price and initialize the option with Q, simulate the factual world with P, perturb P to create counterfactual worlds; the premium and the hedgers' assumptions stay fixed for a controlled comparison.
7. Compute boundary: public infrastructure performs only cached retrieval, bounded single-asset inference and capped JAX simulations; universe fitting, neural training and particle-filter validation stay offline.

### Handoff

- The bundle saves automatically when a calibration finishes; no download, upload or link.
- The visitor is told plainly that it is saved and that the next step is the Hedging Lab tab.
- In the lab the visitor's calibrations appear in the asset picker, the newest preselected.

### Scope

- Every model on the Calibration tab stays a visitor option (P: GBM, OU, Heston, rough Bergomi, regime-switching GBM, Merton, GARCH; Q: Heston, SABR, Dupire), and each must actually work.
- Every P model can feed the lab, with a hedger built on that model's own delta (and gamma where useful).
- The Calibration tab's own simulator stays and shows the range of paths, not just a few lines.
- Calibration should rely largely on Bayesian neural inference; the classical estimators remain as a comparison.
- Compute for calibration and simulation is capped so a visitor cannot exhaust Cloud Run resources.

## What exists today (recon, 2026-10-24)

Measured on a CPU-only API pinned to 2 cores (production's size), AAPL, KO and TSLA, 2023-07-01 to 2026-07-01.

- Heston "particle filter" runs the filter once at hard-coded starting values (kappa 2, xi 0.5, theta = daily return variance) and returns those starting values as the calibration.
- Rough Bergomi "particle filter" returns the moment-matching estimate unchanged; H sits at its 0.49 bound because rolling-window volatility proxies are smooth.
- OU is fitted to raw price levels (AAPL: long-run level 262, sigma 57.7 price units), and the simulator drops the fitted long-run level.
- Regime-switching GBM reports a bear drift of -722%/yr from 9 days.
- Merton finds about 105 jumps a year of 2% size: the likelihood absorbs fat tails as frequent small jumps.
- GARCH reports daily units while the other models report annual units.
- Only GBM and regime switching report standard errors.
- Heston Q calibration runs differential evolution with maxiter 1000, re-pricing the chain by COS each step: minutes per request and unbounded.
- The lab simulates with each asset's historical QMLE drift (NVDA 58.5%/yr), against the drift policy.
- mjollnir's NPE (amortized posterior, mixture density network over hand-built summaries) covers Heston P only; its v0 and rho posteriors were found to be close to the prior.
- mjollnir's backtest engine supports Heston only; the hedging layer assumes zero rates.

## Architecture

### mjollnir: amortized Bayesian calibration (`mjollnir.inference`)

- One framework, one plug-in per model: a prior in unconstrained coordinates, a batched JAX simulator of daily returns, summary features, and a posterior network.
- Posterior network: conditional mixture density network (the existing NPE design) over learned plus hand-built summaries; trained offline on GPU under a deliberately broad prior.
- Sector priors enter by importance reweighting posterior samples by p_sector(theta) / p_broad(theta), so one trained network serves every sector and the global prior.
- Variable history length is handled with masks (already in the Heston NPE).
- Every trained artifact ships with its offline validation: parameter recovery, simulation-based-calibration coverage of its 50% and 90% intervals, and posterior contraction per parameter (how much the data moved it from the prior).
  The receipt shows these, so "rho is barely identified from daily closes" is a measured statement.
- Artifacts are versioned files with a content hash; gleipnir pins them by hash.
- Q side: the same framework on implied-volatility surfaces (Heston, SABR) with simulated quote noise, followed by a bounded least-squares polish on the visitor's actual quotes.
  Dupire stays a direct, capped extraction.
- Classical estimators stay available as "classical" methods, each fixed to be honest: Heston QMLE; rough Bergomi H from the log-absolute-return variogram with its known noise floor; OU on log prices; GARCH reported in annual terms (long-run vol, persistence, half-life); Merton with jump-variance share reported; regime drifts with intervals.
  Standard errors for likelihood-based estimators come from a shared Hessian (sandwich for QMLE) helper.

### mjollnir: offline priors

- `calibrate-universe` over the S&P 500 (classical estimators), then `pool_parameters` (positive-part James-Stein by GICS sector) into a versioned prior artifact per model: global and per-sector means and dispersions in unconstrained coordinates.
- The priors also define the NPE training prior (broadened) and the sector reweighting targets.

### mjollnir: model-agnostic hedging

- Simulation and hedging decouple: any model produces spot paths plus the state its hedger observes (variance, regime, GARCH variance, rough Bergomi forward variance), then a JAX scan runs the strategies over the paths.
- Rates and dividends: cash accrues at r, holdings earn q, spot drifts at r - q plus any excess drift; the liability and deltas use the same r and q.
- Model deltas, computed from the risk-neutral version of each model (drift replaced, volatility dynamics kept):
  GBM and OU on log prices give Black-Scholes delta; Merton gives the Poisson-weighted Black-Scholes delta; Heston uses the existing COS delta; regime-switching GBM uses a COS delta with the regime-switching characteristic function, conditional on the observed regime; GARCH and rough Bergomi use Black-Scholes delta at the model's expected average variance to expiry, labelled as an approximation.
- Deltas that depend on low-dimensional state are precomputed on grids in (time to expiry, state, log-moneyness) and interpolated, so every model hedges at the cost of a table lookup.
- Scenario library per model: volatility spike, correlation breakdown, regime shift and jump cascade map onto each model's own parameters; a scenario that has no meaning for a model is not offered.

### gleipnir API

- Calibration responses carry posterior summaries (mean, standard deviation, 90% interval), diagnostics, provenance and a canonical bundle with its SHA-256 hash.
- Drift appears as an educational estimate with standard error and interval; simulations use the drift policy.
- The lab accepts a bundle in the request body, validates it, recomputes its hash and runs it; the shipped assets become built-in bundles.
- Premium: the market price from the bundle's chain snapshot when present (the Q-calibrated model price otherwise), so the factual world shows the gap between implied and realized behaviour.

### Compute guard

- Input caps on every heavy endpoint: tickers per request, history length, paths, steps, optimizer iterations, chain size.
- A wall-clock budget per request; optimizers stop and report their best result when it runs out.
- A per-instance limit on concurrent heavy jobs; when busy the API answers 503 with Retry-After.
- A per-client token-bucket rate limit on heavy endpoints, keyed on the client IP Cloud Run forwards.
- Caching of identical calibrations (ticker, data date, model, method) for the day.
- Cloud Run: lower request concurrency per instance and a modest max-instances, so abuse cannot scale cost.

### Frontend

- Calibration tab: the same model list; method choice "Bayesian (default)" or "Classical"; a receipt card; a "Saved to the Hedging Lab" confirmation; a marker on the Hedging Lab tab until visited.
- Simulator: a fan chart of path percentiles (5-95, 25-75, median) plus a few sample paths, with the drift policy's scenarios for the chosen horizon.
- Lab: "Your calibrations" in the asset picker; a model-aware market card; the model's own delta hedger next to Black-Scholes delta and no hedge.

## Phases

Each phase ends verified end to end and deployed.

1. Lab drift policy.
2. Compute guard.
3. Honest classical calibration with standard errors.
4. Simulator fan chart and drift scenarios.
5. Model-agnostic hedging with rates and model deltas; the lab runs every P model on shipped calibrations.
6. Offline universe priors.
7. Amortized Bayesian calibration (P, then Q) as the default method.
8. Bundle and handoff.
9. RL demos (separate design).
10. mjollnir polish and release tag.

## Non-goals

- Accounts or server-side storage of visitor data.
- Intraday or tick data.
- Training anything on public infrastructure.

## Anti-goals

- A fitted-looking number that was not fitted.
- A raw historical drift used as a forecast.
- An endpoint whose cost a visitor controls without a cap.

## Calls made by the builder

- Bundles live in the visitor's browser (local storage), up to about 10 assets, the newest per asset replacing the old.
- Mixture density networks over flows, because the existing NPE is one and the parameter counts are small.
- The lab keeps the 5, 21 and 63-day expiries.
- Approximate deltas for GARCH and rough Bergomi rather than nested simulation.
