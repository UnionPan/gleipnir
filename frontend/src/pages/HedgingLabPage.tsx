import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './HedgingLabPage.css';
import { useTheme } from '../contexts/ThemeContext';
import { fetchCatalog, HedgeRequest, hedgerByRole, HedgeResponse, LabCatalog, runHedge, ScenarioName } from '../services/labApi';
import { BUNDLES_EVENT, listBundles, markBundlesSeen, StoredBundle, unseenBundle } from '../services/labBundles';
import LabControls, { LabConfig } from '../components/lab/LabControls';
import MarketCard from '../components/lab/MarketCard';
import ResultsTable from '../components/lab/ResultsTable';
import DistributionPanels from '../components/lab/DistributionPanels';
import SamplePathPanels from '../components/lab/SamplePathPanels';
import { readResult } from '../components/lab/insights';

function initialConfig(catalog: LabCatalog, bundles: StoredBundle[], arrival: string | null): LabConfig {
  const severity = Object.fromEntries(catalog.scenarios.map((s) => [s.name, s.severity_default])) as Record<ScenarioName, number>;
  // A calibration saved since the lab was last opened is what the visitor came for.
  const fresh = bundles.find((b) => b.key === arrival);
  return {
    asset: catalog.assets[0].ticker,
    bundleKey: fresh?.key ?? null,
    model: fresh?.bundle.model ?? 'heston',
    rhoAssumed: true,
    scenario: 'vol_spike',
    severity,
    isCall: true,
    moneyness: 1.0,
    maturityDays: 21,
    costBps: 5,
    riskAversion: 5,
    seed: 7,
  };
}

function toRequest(config: LabConfig, catalog: LabCatalog, bundles: StoredBundle[]): HedgeRequest {
  const hasRho = catalog.models.find((m) => m.name === config.model)?.has_rho ?? false;
  const bundle = bundles.find((b) => b.key === config.bundleKey)?.bundle;
  return {
    ...(bundle ? { calibration: bundle } : { asset: config.asset }),
    model: config.model,
    rho: hasRho && config.rhoAssumed ? catalog.rho_assumption : null,
    scenario: config.scenario ? { name: config.scenario, severity: config.severity[config.scenario] } : null,
    option: { is_call: config.isCall, moneyness: config.moneyness, maturity_days: config.maturityDays },
    cost_bps: config.costBps,
    risk_aversion: config.riskAversion,
    seed: config.seed,
  };
}

const HedgingLabPage: React.FC = () => {
  const { theme } = useTheme();
  const [catalog, setCatalog] = useState<LabCatalog | null>(null);
  const [config, setConfig] = useState<LabConfig | null>(null);
  const [result, setResult] = useState<HedgeResponse | null>(null);
  const [resultScenario, setResultScenario] = useState<ScenarioName | null>(null);
  const [lastRun, setLastRun] = useState<string>('');
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [bundles, setBundles] = useState<StoredBundle[]>(() => listBundles());
  // Read once per visit: effects may run twice (StrictMode), and marking it seen must not lose it.
  const [arrival] = useState<string | null>(() => unseenBundle());
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => {
    const reload = () => setBundles(listBundles());
    window.addEventListener(BUNDLES_EVENT, reload);
    window.addEventListener('storage', reload);  // saved from another tab
    return () => {
      window.removeEventListener(BUNDLES_EVENT, reload);
      window.removeEventListener('storage', reload);
    };
  }, []);

  const run = useCallback(async (cfg: LabConfig, cat: LabCatalog, saved: StoredBundle[]) => {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    const request = toRequest(cfg, cat, saved);
    setRunning(true);
    setError(null);
    const started = performance.now();
    const timer = window.setInterval(() => setElapsed((performance.now() - started) / 1000), 100);
    try {
      const response = await runHedge(request, controller.signal);
      setResult(response);
      setResultScenario(cfg.scenario);
      setLastRun(JSON.stringify(request));
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      if (message !== 'cancelled') setError(message);
    } finally {
      window.clearInterval(timer);
      if (inFlight.current === controller) setRunning(false);
    }
  }, []);

  useEffect(() => {
    const start = async () => {
      try {
        const cat = await fetchCatalog();
        const saved = listBundles();
        const cfg = initialConfig(cat, saved, arrival);
        markBundlesSeen();
        setCatalog(cat);
        setConfig(cfg);
        await run(cfg, cat, saved);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    void start();
    return () => inFlight.current?.abort();
  }, [run, arrival]);

  const insight = useMemo(() => (result ? readResult(result, resultScenario) : null), [result, resultScenario]);
  const stale = !!(catalog && config && lastRun && JSON.stringify(toRequest(config, catalog, bundles)) !== lastRun);
  const worldCount = config?.scenario ? 2 : 1;

  return (
    <div className="lab-page">
      <header className="page-header">
        <h1>Counterfactual Hedging Lab</h1>
        <p className="page-subtitle">
          Sell an option, hedge it, then ask: what if the market had turned out differently?
        </p>
      </header>

      {error && (
        <div className="lab-error" role="alert">
          <span>{error}</span>
          <button type="button" className="error-close" onClick={() => setError(null)} aria-label="Dismiss">×</button>
        </div>
      )}

      {catalog && config && (
        <LabControls catalog={catalog} bundles={bundles} config={config} onChange={setConfig}
          onRun={() => void run(config, catalog, bundles)} running={running} stale={stale} />
      )}

      {running && catalog && (
        <div className="lab-progress" role="status">
          <span className="lab-progress-bar" aria-hidden />
          Simulating {catalog.n_paths.toLocaleString()} paths × {worldCount} world{worldCount > 1 ? 's' : ''} × 3 hedgers
          <span className="lab-progress-time">{elapsed.toFixed(1)} s</span>
        </div>
      )}

      {!result && !error && (
        <p className="lab-empty">{catalog ? 'Running the first experiment…' : 'Loading calibrations…'}</p>
      )}

      {result && insight && catalog && (
        <div className={`lab-results-area ${running ? 'lab-refetching' : ''}`} aria-busy={running}>
          <div className="lab-top-grid">
            <MarketCard catalog={catalog} result={result} />
            <ResultsTable result={result} insight={insight} />
          </div>
          <DistributionPanels result={result} theme={theme} />
          <SamplePathPanels result={result} theme={theme} />
          <section className="lab-panel" aria-label="How this works">
            <div className="lab-titlebar">How this works</div>
            <div className="lab-panel-body lab-explainer">
              <div>
                <h3>The book</h3>
                <p>
                  You sell one option at the chosen model's price under the factual calibration and set up the hedgers
                  then. The premium and the hedgers' assumptions never change; only the world does.
                </p>
              </div>
              <div>
                <h3>The worlds</h3>
                <p>
                  The factual world simulates the calibrated model forward from today, the spot drifting at r − q. A
                  counterfactual applies a stress scenario in the same model, with the same random numbers, so
                  differences come from the world and not from luck.
                </p>
              </div>
              <div>
                <h3>The hedgers</h3>
                <p>
                  Black-Scholes delta uses one volatility for the whole life of the option. The model's own hedger
                  uses what its model watches ({hedgerByRole(result, 'model').label}): it adapts when that moves, as
                  long as its model is right.
                  {result.hedgers.some((h) => h.role === 'learned') && (
                    <> The learned band follows that delta but skips trades not worth their cost: a network trained
                      offline through this model&apos;s simulated world decides how far the position may drift.</>
                  )}
                </p>
              </div>
              <div>
                <h3>What this simplifies</h3>
                <p>
                  Daily rebalancing and proportional trading costs. The model's hedger observes the simulated state
                  (variance, regime) a real desk can only estimate. GARCH and rough-volatility hedgers are labelled
                  approximations. Calibrated on daily prices only.
                </p>
              </div>
            </div>
          </section>
          <footer className="lab-footer">
            {result.n_paths.toLocaleString()} paths per world · seed {result.seed} · {(result.elapsed_ms / 1000).toFixed(2)} s
            on the server · mjollnir {result.mjollnir_version}
          </footer>
        </div>
      )}
    </div>
  );
};

export default HedgingLabPage;
