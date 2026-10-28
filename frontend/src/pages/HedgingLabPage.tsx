import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './HedgingLabPage.css';
import { useTheme } from '../contexts/ThemeContext';
import { fetchCatalog, HedgeRequest, hedgerByRole, HedgeResponse, LabCatalog, runHedge, ScenarioName } from '../services/labApi';
import { BUNDLES_EVENT, listBundles, markBundlesSeen, StoredBundle, unseenBundle } from '../services/labBundles';
import LabControls, { LabConfig } from '../components/lab/LabControls';
import MarketCard from '../components/lab/MarketCard';
import ResultsTable from '../components/lab/ResultsTable';
import DistributionPanels from '../components/lab/DistributionPanels';
import SimulationHero, { WorldView } from '../components/lab/SimulationHero';
import FollowPath from '../components/lab/FollowPath';
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

const MATURITY_WORDS: Record<number, string> = { 5: 'one-week', 21: 'one-month', 63: 'three-month' };

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
  const [view, setView] = useState<WorldView>('both');
  const [path, setPath] = useState(0);
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
      setPath(response.factual.paths?.highlight ?? 0);
      setView('both');
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
  const modelLabel = result && catalog ? catalog.models.find((m) => m.name === result.book.model)?.label : '';

  return (
    <div className="lab-page">
      <header className="page-header">
        <h1>Counterfactual Hedging Lab</h1>
        <p className="page-subtitle">
          Sell an option on a real stock, hedge it in thousands of simulated futures, then ask what if the market changed.
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
          Simulating {catalog.n_paths.toLocaleString()} futures{worldCount > 1 ? ' in two worlds' : ''}
          <span className="lab-progress-time">{elapsed.toFixed(1)} s</span>
        </div>
      )}

      {!result && !error && (
        <p className="lab-empty">{catalog ? 'Running the first experiment…' : 'Loading calibrations…'}</p>
      )}

      {result && insight && catalog && (
        <div className={`lab-results-area ${running ? 'lab-refetching' : ''}`} aria-busy={running}>
          <section className="lab-panel" aria-label="Simulation">
            <div className="lab-titlebar">
              {result.book.asset} · {modelLabel} · short {MATURITY_WORDS[result.book.maturity_days] ?? ''}{' '}
              {result.book.is_call ? 'call' : 'put'} at {result.book.strike.toFixed(0)}
            </div>
            <div className="lab-panel-body">
              <div className="lab-verdict">
                <p className="lab-headline">{insight.headline}</p>
                <details className="lab-why">
                  <summary>Why?</summary>
                  <p>{insight.detail}</p>
                </details>
              </div>
              {result.counterfactual && (
                <div className="lab-view" role="radiogroup" aria-label="World shown">
                  {([['factual', result.factual.label], ['counterfactual', result.counterfactual.label],
                    ['both', 'Both']] as [WorldView, string][]).map(([v, label]) => (
                    <button key={v} type="button" role="radio" aria-checked={view === v}
                      className={`lab-view-option ${view === v ? 'active' : ''}`} onClick={() => setView(v)}>
                      {v !== 'both' && <span className={`lab-key lab-key-${v}`} aria-hidden />}
                      {label}
                    </button>
                  ))}
                  <span className="lab-view-note">Same random draws in both worlds: only the market differs.</span>
                </div>
              )}
              {result.factual.paths
                ? <SimulationHero result={result} theme={theme} view={view} selected={path} onSelect={setPath} />
                : <DistributionPanels result={result} theme={theme} />}
            </div>
          </section>

          {result.factual.paths && (
            <FollowPath result={result} theme={theme} view={view} path={path} onPath={setPath} />
          )}

          <details className="lab-disclosure">
            <summary>
              Market and model <span>{result.book.asset}, premium {result.book.premium.toFixed(2)}, implied vol{' '}
                {(100 * result.book.implied_vol).toFixed(1)}%</span>
            </summary>
            <MarketCard catalog={catalog} result={result} />
          </details>
          <details className="lab-disclosure">
            <summary>All numbers <span>every hedger in every world, over all {result.n_paths.toLocaleString()} paths</span></summary>
            <ResultsTable result={result} />
          </details>
          <details className="lab-disclosure">
            <summary>How this works</summary>
            <div className="lab-explainer">
              <div>
                <h3>The book</h3>
                <p>
                  You sell one option at the model's price and hedge it until expiry. The premium and the hedgers'
                  assumptions are fixed on day 0; only the world changes.
                </p>
              </div>
              <div>
                <h3>The worlds</h3>
                <p>
                  Factual: the model fitted to the stock's daily prices, drifting at r − q. What if: a stress applied to
                  that model, run on the same random numbers, so every difference comes from the world.
                </p>
              </div>
              <div>
                <h3>The hedgers</h3>
                <p>
                  Black-Scholes delta keeps one volatility for the option's life. {hedgerByRole(result, 'model').label}{' '}
                  follows what its model watches.
                  {result.hedgers.some((h) => h.role === 'learned') && (
                    <> The learned band follows that delta but skips trades not worth their cost, as far as a network
                      trained offline on this model allows.</>
                  )}
                </p>
              </div>
              <div>
                <h3>What it simplifies</h3>
                <p>
                  Daily rebalancing, proportional costs, and a hedger that sees the simulated state (variance, regime)
                  a real desk can only estimate.
                </p>
              </div>
            </div>
          </details>
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
