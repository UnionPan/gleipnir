import React, { useCallback, useEffect, useRef, useState } from 'react';
import './HedgingLabPage.css';
import './CalibrationPage.css';
import { useTheme } from '../contexts/ThemeContext';
import {
  calibrationApi,
  CalibrationResponse,
  OHLCVData,
  OptionChainData,
  SimulationResponse,
  VolSurfaceResponse,
} from '../services/calibrationApi';
import type { CalibrationBundle } from '../services/labApi';
import { saveBundle } from '../services/labBundles';
import CalibrationBar, { CalibrationSetup, parseTickers } from '../components/calibration/CalibrationBar';
import FutureFan from '../components/calibration/FutureFan';
import ReproductionCheck from '../components/calibration/ReproductionCheck';
import FitDetails from '../components/calibration/FitDetails';
import OHLCVChart from '../components/calibration/OHLCVChart';
import SavedToLab from '../components/calibration/SavedToLab';
import VolatilitySurface3D from '../components/calibration/VolatilitySurface3D';
import Disclosure from '../components/common/Disclosure';
import { headline, keyNumbers, missedFeatures, modelLabel } from '../components/calibration/story';

// Simulated futures drift at r - q; with no option chain at hand, q is 0.
const RISK_FREE_RATE = 0.04;
const SIM_PATHS = 1000;
const SIM_PATHS_SHOWN = 50;
const SIM_SEED = 7;
const DEFAULT_HORIZON = 252;
// What Q fits keep from the chain: liquid quotes near the money.
const Q_FILTER = { min_volume: 10, min_open_interest: 50, max_spread_pct: 0.5, moneyness_range: [0.8, 1.2] as [number, number] };
const Q_MAXITER = 200;

const isoDate = (d: Date) => d.toISOString().slice(0, 10);

function windowDates(setup: CalibrationSetup, today: Date = new Date()): { start: string; end: string } {
  if (setup.windowYears === null) return { start: setup.startDate, end: setup.endDate };
  const start = new Date(today);
  start.setFullYear(start.getFullYear() - setup.windowYears);
  return { start: isoDate(start), end: isoDate(today) };
}

const INITIAL: CalibrationSetup = {
  measure: 'P',
  tickers: 'SPY',
  model: 'heston',
  method: 'bayesian',
  prior: 'sector',
  includeDrift: true,
  windowYears: 5,
  startDate: '',
  endDate: '',
  riskFreeRate: 0.05,
  minDays: 30,
};

/** What a run fitted: everything the results area shows comes from here, never from the live setup. */
interface Fitted {
  measure: CalibrationSetup['measure'];
  results: CalibrationResponse[];
  histories: Record<string, OHLCVData>;
  chain: OptionChainData | null;
  surface: VolSurfaceResponse | null;
  surfaceError: string | null;
  bundles: CalibrationBundle[];
  storageFailed: boolean;
}

const simKey = (ticker: string, horizon: number) => `${ticker}:${horizon}`;

const shortDate = (iso: string) => new Date(`${iso.slice(0, 10)}T12:00:00Z`)
  .toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function simulate(result: CalibrationResponse, history: OHLCVData, horizon: number, signal: AbortSignal) {
  return calibrationApi.simulate({
    model: result.model,
    parameters: result.parameters,
    s0: history.close[history.close.length - 1],
    nSteps: horizon,
    nPaths: SIM_PATHS,
    dt: 1 / 252,
    riskFreeRate: RISK_FREE_RATE,
    dividendYield: 0,
    maxPathsReturn: SIM_PATHS_SHOWN,
    randomSeed: SIM_SEED,
  }, signal);
}

interface Props {
  onOpenLab?: () => void;
}

const CalibrationPage: React.FC<Props> = ({ onOpenLab }) => {
  const { theme } = useTheme();
  const [setup, setSetup] = useState<CalibrationSetup>(INITIAL);
  const [fitted, setFitted] = useState<Fitted | null>(null);
  const [lastRun, setLastRun] = useState('');
  const [active, setActive] = useState('');
  const [horizon, setHorizon] = useState(DEFAULT_HORIZON);
  const [sims, setSims] = useState<Record<string, SimulationResponse>>({});
  const [simulating, setSimulating] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [phase, setPhase] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef<AbortController | null>(null);
  const simFlight = useRef<AbortController | null>(null);

  useEffect(() => () => {
    inFlight.current?.abort();
    simFlight.current?.abort();
  }, []);

  const run = useCallback(async () => {
    inFlight.current?.abort();
    simFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    const { signal } = controller;
    const tickers = parseTickers(setup.tickers);
    const model = modelLabel(setup.model);
    const started = performance.now();
    const timer = window.setInterval(() => setElapsed((performance.now() - started) / 1000), 100);
    setError(null);
    setElapsed(0);
    try {
      if (setup.measure === 'P') {
        const { start, end } = windowDates(setup);
        setPhase(`Downloading ${tickers.join(', ')} daily prices`);
        const data = await calibrationApi.fetchOHLCV(tickers, start, end, signal);
        const histories = Object.fromEntries(data.map((d) => [d.ticker, d]));
        setPhase(`Fitting ${model} to ${data[0]?.dates.length.toLocaleString('en-US') ?? 'the'} trading days`);
        const results = await calibrationApi.calibrate({
          tickers, model: setup.model, startDate: start, endDate: end, method: setup.method,
          includeDrift: setup.includeDrift, prior: setup.prior,
        }, signal);
        const bundles = results.map((r) => r.lab_bundle).filter((b): b is CalibrationBundle => !!b);
        const storageFailed = !bundles.every((b) => saveBundle(b));
        const first = results[0];
        setPhase(`Simulating ${SIM_PATHS.toLocaleString('en-US')} futures of ${first.ticker}`);
        const sim = await simulate(first, histories[first.ticker], horizon, signal);
        setFitted({ measure: 'P', results, histories, chain: null, surface: null, surfaceError: null, bundles, storageFailed });
        setSims({ [simKey(first.ticker, horizon)]: sim });
        setActive(first.ticker);
      } else {
        const ticker = tickers[0];
        setPhase(`Downloading ${ticker} option chain`);
        const chain = await calibrationApi.fetchOptionChain(ticker, undefined, setup.riskFreeRate, undefined, signal);
        setPhase(`Fitting ${model} to ${ticker} option prices`);
        const result = await calibrationApi.calibrateQMeasure({
          ticker, model: setup.model, riskFreeRate: setup.riskFreeRate, calibrationMethod: setup.method,
          maxiter: Q_MAXITER, filterParams: Q_FILTER, minDays: setup.minDays,
        }, signal);
        setPhase('Drawing the volatility surface');
        let surface: VolSurfaceResponse | null = null;
        let surfaceError: string | null = null;
        try {
          surface = await calibrationApi.generateVolSurface({
            ticker, model: result.model, parameters: result.parameters, spotPrice: chain.spot_price,
            riskFreeRate: setup.riskFreeRate, dividendYield: chain.dividend_yield || 0, nStrikes: 30, nMaturities: 20,
            strikeRange: [0.7, 1.3], maturityRange: [0.05, 2.0],
          }, signal);
        } catch (e) {
          // the fit stands without its surface; a cancelled run is handled below
          if (signal.aborted) throw e;
          surfaceError = message(e);
        }
        setFitted({
          measure: 'Q', results: [{ ...result, measure: 'Q-measure', method: setup.method }], histories: {}, chain,
          surface, surfaceError, bundles: [], storageFailed: false,
        });
        setSims({});
        setActive(ticker);
      }
      setLastRun(JSON.stringify(setup));
      setSimError(null);
    } catch (e) {
      if (message(e) !== 'cancelled') setError(message(e));
    } finally {
      window.clearInterval(timer);
      if (inFlight.current === controller) setPhase(null);
    }
  }, [setup, horizon]);

  // A horizon or a ticker not simulated yet: simulate it, keeping the last fan on screen meanwhile.
  const showFutures = useCallback(async (ticker: string, days: number) => {
    setActive(ticker);
    setHorizon(days);
    if (!fitted || fitted.measure !== 'P' || sims[simKey(ticker, days)]) return;
    const result = fitted.results.find((r) => r.ticker === ticker);
    if (!result) return;
    simFlight.current?.abort();
    const controller = new AbortController();
    simFlight.current = controller;
    setSimulating(true);
    setSimError(null);
    try {
      const sim = await simulate(result, fitted.histories[ticker], days, controller.signal);
      setSims((prev) => ({ ...prev, [simKey(ticker, days)]: sim }));
    } catch (e) {
      if (message(e) !== 'cancelled') setSimError(message(e));
    } finally {
      if (simFlight.current === controller) setSimulating(false);
    }
  }, [fitted, sims]);

  const download = () => {
    if (!fitted) return;
    const data = fitted.measure === 'P' ? fitted.results : fitted.results[0];
    const link = document.createElement('a');
    link.href = `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(data, null, 2))}`;
    link.download = `calibration_${fitted.measure === 'P' ? 'pmeasure' : 'qmeasure'}_${isoDate(new Date())}.json`;
    link.click();
  };

  const running = phase !== null;
  const stale = !!lastRun && JSON.stringify(setup) !== lastRun;
  const result = fitted?.results.find((r) => r.ticker === active) ?? fitted?.results[0];
  const history = result ? fitted?.histories[result.ticker] : undefined;
  // the fan on screen: this ticker at this horizon, or while that one loads, the last one simulated for it
  const sim = result ? (sims[simKey(result.ticker, horizon)]
    ?? Object.entries(sims).find(([k]) => k.startsWith(`${result.ticker}:`))?.[1]) : undefined;

  return (
    <div className="lab-page calib-page">
      <header className="page-header">
        <h1>Model Calibration</h1>
        <p className="page-subtitle">
          Fit a model of how a stock moves to its real prices, check what the fit gets right, and watch the futures it
          implies.
        </p>
      </header>

      {error && (
        <div className="lab-error" role="alert">
          <span>{error}</span>
          <button type="button" className="error-close" onClick={() => setError(null)} aria-label="Dismiss">×</button>
        </div>
      )}

      <CalibrationBar setup={setup} onChange={setSetup} onRun={() => void run()} running={running} stale={stale} />

      {running && (
        <div className="lab-progress" role="status">
          <span className="lab-progress-bar" aria-hidden />
          {phase}
          <span className="lab-progress-time">{elapsed.toFixed(1)} s</span>
        </div>
      )}

      {!fitted && !running && (
        <div className="calib-start">
          <p>
            Pick a stock and a model, then <strong>Calibrate</strong>: the model is fitted to the stock&apos;s daily prices,
            checked against histories simulated from the fit, and run forward from today&apos;s price.
          </p>
          <button type="button" className="lab-run" onClick={() => void run()}>
            Calibrate {modelLabel(setup.model)} to {parseTickers(setup.tickers).join(', ') || 'a stock'}
          </button>
        </div>
      )}

      {fitted && result && (
        <div className={`lab-results-area ${running ? 'lab-refetching' : ''}`} aria-busy={running}>
          <section className="lab-panel" aria-label="Calibration result">
            <div className="lab-titlebar">
              {result.ticker} · {modelLabel(result.model)} ·{' '}
              {fitted.measure === 'P' && history
                ? `${history.dates.length.toLocaleString('en-US')} trading days to ${shortDate(history.dates[history.dates.length - 1])}`
                : `option chain of ${shortDate(fitted.chain?.reference_date ?? result.timestamp)}`}
            </div>
            <div className="lab-panel-body">
              {fitted.results.length > 1 && (
                <div className="lab-view" role="radiogroup" aria-label="Stock shown">
                  {fitted.results.map((r) => (
                    <button key={r.ticker} type="button" role="radio" aria-checked={r.ticker === result.ticker}
                      className={`lab-view-option ${r.ticker === result.ticker ? 'active' : ''}`}
                      onClick={() => void showFutures(r.ticker, horizon)}>{r.ticker}</button>
                  ))}
                </div>
              )}
              <div className="lab-verdict">
                <p className="lab-headline">{headline(result)}</p>
              </div>
              <dl className="calib-numbers">
                {keyNumbers(result).map((k) => (
                  <div key={k.key} title={k.hint}>
                    <dt>{k.label}</dt>
                    <dd>{k.value}</dd>
                  </div>
                ))}
              </dl>

              {fitted.measure === 'P' && history && (sim
                ? (
                  <FutureFan ticker={result.ticker} model={modelLabel(result.model)} dates={history.dates}
                    closes={history.close} simulation={sim} theme={theme} horizon={horizon} simulating={simulating}
                    onHorizon={(days) => void showFutures(result.ticker, days)} />
                )
                : <p className="lab-empty">{simError ?? 'Simulating futures…'}</p>)}
              {simError && sim && <p className="lab-error-inline" role="alert">{simError}</p>}

              {fitted.measure === 'Q' && fitted.chain && (fitted.surface
                ? <VolatilitySurface3D surface={fitted.surface} chain={fitted.chain} theme={theme} />
                : <p className="lab-empty">The volatility surface could not be drawn: {fitted.surfaceError}</p>)}

              {fitted.measure === 'P' && (
                <SavedToLab bundles={fitted.bundles} storageFailed={fitted.storageFailed} onOpenLab={onOpenLab} />
              )}
            </div>
          </section>

          {result.diagnostics.predictive_check && (
            <Disclosure title={`Does ${modelLabel(result.model)} reproduce ${result.ticker}'s history?`}
              hint={(() => {
                const n = result.diagnostics.predictive_check.aspects.length;
                const missed = missedFeatures(result).length;
                return `${n - missed} of ${n} features, against ${result.diagnostics.predictive_check.n_replicates} simulated histories`;
              })()}>
              <ReproductionCheck check={result.diagnostics.predictive_check} ticker={result.ticker} theme={theme} />
            </Disclosure>
          )}
          <Disclosure title="How sure is the fit?"
            hint={result.diagnostics.posterior ? 'the posterior, its 90% intervals, what the data pin down' : 'estimates and their errors'}>
            <FitDetails result={result} theme={theme} />
          </Disclosure>
          {fitted.measure === 'P' && history && (
            <Disclosure title="Price history"
              hint={`daily candles and volume, ${shortDate(history.dates[0])} to ${shortDate(history.dates[history.dates.length - 1])}`}>
              <OHLCVChart data={history} theme={theme} />
            </Disclosure>
          )}
          <Disclosure title="How this works">
            {fitted.measure === 'P' ? (
              <div className="lab-explainer">
                <div>
                  <h3>The fit</h3>
                  <p>
                    The model&apos;s parameters are read off the stock&apos;s daily returns. A Bayesian fit uses a
                    network trained offline on millions of simulated histories: it returns the whole posterior in one
                    pass, and its intervals were checked for coverage on fresh simulations.
                  </p>
                </div>
                <div>
                  <h3>The prior</h3>
                  <p>
                    Before its own history is read, a stock is taken to look like its S&amp;P 500 sector. Stocks outside
                    the S&amp;P 500 start from the network&apos;s broad training range.
                  </p>
                </div>
                <div>
                  <h3>The check</h3>
                  <p>
                    Histories simulated from the fit are compared with the real one, feature by feature. What the model
                    cannot produce shows up there, however tightly its parameters are estimated.
                  </p>
                </div>
                <div>
                  <h3>The futures</h3>
                  <p>
                    Futures start at today&apos;s price and drift at the risk-free rate less dividends, r − q: a
                    stock&apos;s past return is far too uncertain to forecast with. Each fit is also saved for the
                    Hedging Lab.
                  </p>
                </div>
              </div>
            ) : (
              <div className="lab-explainer">
                <div>
                  <h3>The fit</h3>
                  <p>
                    Option prices say how the market prices risk today, not how the stock will move: a fit to them is
                    risk-neutral. {result.diagnostics.method_label ?? ''}
                  </p>
                </div>
                <div>
                  <h3>The quotes</h3>
                  <p>
                    Only liquid quotes near the money are kept: at least {Q_FILTER.min_volume} contracts traded,{' '}
                    {Q_FILTER.min_open_interest} open, a spread under {100 * Q_FILTER.max_spread_pct}% of the price, and
                    strikes within {100 * (1 - Q_FILTER.moneyness_range[0])}% of the spot.
                  </p>
                </div>
                <div>
                  <h3>The surface</h3>
                  <p>
                    The volatility the fitted model implies for each strike and expiry, with the market&apos;s own quotes
                    as dots: where they leave the surface, the model misprices.
                  </p>
                </div>
              </div>
            )}
          </Disclosure>
          <footer className="lab-footer">
            <button type="button" className="lab-link" onClick={download}>Download the fit as JSON</button>
            {result.diagnostics.posterior?.network && (
              <span title={`Network weights SHA-256 ${result.diagnostics.posterior.network.sha256}`}>
                {' '}· network {result.diagnostics.posterior.network.sha256.slice(0, 8)}
              </span>
            )}
            {' '}· {fitted.measure === 'P' ? 'real-world (P) measure' : 'risk-neutral (Q) measure'}
          </footer>
        </div>
      )}
    </div>
  );
};

export default CalibrationPage;
