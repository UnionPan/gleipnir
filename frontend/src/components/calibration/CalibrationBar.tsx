import React, { useState } from 'react';
import type { PriorChoice } from '../../services/calibrationApi';

export type Measure = 'P' | 'Q';

export interface CalibrationSetup {
  measure: Measure;
  /** What the visitor typed: one ticker, or several separated by commas. */
  tickers: string;
  model: string;
  method: string;
  prior: PriorChoice;
  includeDrift: boolean;
  /** Years of history up to today; null reads the custom dates. */
  windowYears: number | null;
  startDate: string;
  endDate: string;
  riskFreeRate: number;
  /** Option-price fits: the shortest expiry used, in days. */
  minDays: number;
}

interface Choice {
  value: string;
  label: string;
  summary: string;
}

export const P_MODELS: Choice[] = [
  { value: 'heston', label: 'Heston', summary: 'Volatility moves at random and reverts to a long-run level.' },
  { value: 'gbm', label: 'Black-Scholes (GBM)', summary: 'One constant volatility: the textbook baseline.' },
  { value: 'garch', label: 'GARCH(1,1)', summary: 'Each day\'s volatility reacts to the last shock, and the reaction persists.' },
  { value: 'rough_bergomi', label: 'Rough Bergomi', summary: 'Volatility with rough, erratic paths, as measured in real markets.' },
  { value: 'merton_jump', label: 'Merton jump-diffusion', summary: 'Smooth moves plus occasional sudden jumps.' },
  { value: 'regime_switching_gbm', label: 'Regime-switching GBM', summary: 'The stock switches between a calm and a turbulent regime.' },
  { value: 'ou', label: 'Ornstein-Uhlenbeck', summary: 'The log price reverts to a fitted level: a statement about the past.' },
];

export const Q_MODELS: Choice[] = [
  { value: 'heston', label: 'Heston', summary: 'Stochastic volatility fitted to today\'s option prices.' },
  { value: 'bates', label: 'Bates (Heston with jumps)', summary: 'Stochastic volatility plus sudden jumps: what a steep short-dated skew needs.' },
  { value: 'sabr', label: 'SABR', summary: 'Hagan\'s smile for each expiry, with β fixed at 0.5.' },
  { value: 'dupire', label: 'Dupire local volatility', summary: 'A local-volatility surface read straight off the option prices.' },
];

const BAYESIAN = 'Bayesian (neural posterior)';
// The default first: every model but OU is fitted as a posterior unless the classical estimator is chosen.
export const METHODS: Record<Measure, Record<string, { value: string; label: string }[]>> = {
  P: {
    gbm: [{ value: 'bayesian', label: 'Bayesian (exact posterior)' }, { value: 'mle', label: 'Maximum likelihood' }],
    ou: [{ value: 'mle', label: 'Maximum likelihood on log prices' }],
    heston: [{ value: 'bayesian', label: BAYESIAN }, { value: 'moments', label: 'Method of moments + bootstrap' }],
    rough_bergomi: [{ value: 'bayesian', label: BAYESIAN }, { value: 'variogram', label: 'Variogram + bootstrap' }],
    regime_switching_gbm: [
      { value: 'bayesian', label: 'Bayesian (neural posterior, 2 regimes)' },
      { value: 'threshold', label: '20% drawdown / rally regimes' },
    ],
    merton_jump: [{ value: 'bayesian', label: BAYESIAN }, { value: 'mle', label: 'Maximum likelihood (multi-start)' }],
    garch: [{ value: 'bayesian', label: 'Bayesian (neural posterior, Student-t)' }, { value: 'mle', label: 'Quasi-MLE (Gaussian)' }],
  },
  Q: {
    heston: [{ value: 'levenberg_marquardt', label: 'Levenberg-Marquardt (exact COS gradients)' }],
    bates: [{ value: 'levenberg_marquardt', label: 'Levenberg-Marquardt (exact COS gradients)' }],
    sabr: [{ value: 'L-BFGS-B', label: 'L-BFGS-B' }, { value: 'differential_evolution', label: 'Differential evolution (global)' }],
    dupire: [{ value: 'extraction', label: 'Direct extraction' }],
  },
};

const POPULAR = [
  ['SPY', 'S&P 500 ETF'], ['QQQ', 'Nasdaq 100 ETF'], ['AAPL', 'Apple'], ['MSFT', 'Microsoft'], ['NVDA', 'NVIDIA'],
  ['GOOGL', 'Alphabet'], ['AMZN', 'Amazon'], ['META', 'Meta'], ['TSLA', 'Tesla'], ['XOM', 'Exxon Mobil'],
  ['JPM', 'JPMorgan Chase'], ['BTC-USD', 'Bitcoin'], ['ETH-USD', 'Ethereum'],
];

const WINDOWS = [1, 2, 5];

export const parseTickers = (text: string) => Array.from(new Set(text.toUpperCase().split(/[\s,;]+/).filter(Boolean)));

/** The setup's model and method after a change of measure or model: the first method is the default. */
export function withModel(setup: CalibrationSetup, measure: Measure, model: string): CalibrationSetup {
  const models = measure === 'P' ? P_MODELS : Q_MODELS;
  const picked = models.some((m) => m.value === model) ? model : models[0].value;
  const methods = METHODS[measure][picked];
  const method = methods.some((m) => m.value === setup.method) ? setup.method : methods[0].value;
  return { ...setup, measure, model: picked, method };
}

interface Props {
  setup: CalibrationSetup;
  onChange: (setup: CalibrationSetup) => void;
  onRun: () => void;
  running: boolean;
  stale: boolean;
}

const CalibrationBar: React.FC<Props> = ({ setup, onChange, onRun, running, stale }) => {
  const [more, setMore] = useState(false);
  const set = <K extends keyof CalibrationSetup>(key: K, value: CalibrationSetup[K]) => onChange({ ...setup, [key]: value });
  const models = setup.measure === 'P' ? P_MODELS : Q_MODELS;
  const model = models.find((m) => m.value === setup.model);
  const methods = METHODS[setup.measure][setup.model] ?? [];
  const tickers = parseTickers(setup.tickers);
  const several = setup.measure === 'Q' && tickers.length > 1;

  return (
    <section className="lab-panel lab-controls" aria-label="Calibration setup">
      <div className="lab-titlebar">Fit a model to a stock</div>
      <form className="lab-bar" onSubmit={(e) => { e.preventDefault(); if (!running) onRun(); }}>
        <div className="lab-field calib-measure">
          <span>Fit to</span>
          <div className="lab-segmented" role="radiogroup" aria-label="What the model is fitted to">
            {([['P', 'Price history'], ['Q', 'Option prices']] as [Measure, string][]).map(([m, label]) => (
              <button key={m} type="button" role="radio" aria-checked={setup.measure === m}
                className={setup.measure === m ? 'active' : ''} onClick={() => onChange(withModel(setup, m, setup.model))}>
                {label}
              </button>
            ))}
          </div>
        </div>
        <label className="lab-field">
          <span>Stock</span>
          <input type="text" list="calib-popular" aria-label="Stock" value={setup.tickers} spellCheck={false} autoComplete="off"
            placeholder="e.g. AAPL" aria-invalid={tickers.length === 0 || several}
            onChange={(e) => set('tickers', e.target.value.toUpperCase())} />
          <datalist id="calib-popular">
            {POPULAR.map(([t, name]) => <option key={t} value={t}>{name}</option>)}
          </datalist>
        </label>
        <label className="lab-field" title={model?.summary}>
          <span>Model</span>
          <select aria-label="Model" value={setup.model} onChange={(e) => onChange(withModel(setup, setup.measure, e.target.value))}>
            {models.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
          </select>
        </label>
        {setup.measure === 'P' && (
          <div className="lab-field">
            <span>History</span>
            <div className="lab-segmented" role="radiogroup" aria-label="Years of history">
              {WINDOWS.map((y) => (
                <button key={y} type="button" role="radio" aria-checked={setup.windowYears === y}
                  className={setup.windowYears === y ? 'active' : ''} onClick={() => set('windowYears', y)}>
                  {y} yr
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="lab-bar-actions">
          <button type="submit" className={`lab-run ${stale ? 'lab-run-stale' : ''}`}
            disabled={running || tickers.length === 0 || several}>
            {running ? 'Calibrating…' : stale ? 'Calibrate with new settings' : 'Calibrate'}
          </button>
          <button type="button" className="lab-link" aria-expanded={more} onClick={() => setMore(!more)}>
            {more ? 'Fewer settings' : 'More settings'}
          </button>
        </div>
      </form>
      <p className="lab-bar-hint">
        {several
          ? 'Option prices are fitted one stock at a time.'
          : <>{model?.label}: {model?.summary}</>}
      </p>
      {more && (
        <div className="lab-more">
          <label className="lab-field">
            <span>Method</span>
            <select aria-label="Method" value={setup.method} onChange={(e) => set('method', e.target.value)}>
              {methods.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </label>
          {setup.measure === 'P' && setup.method === 'bayesian' && setup.model !== 'gbm' && (
            <label className="lab-field" title="Stocks outside the S&P 500 use the broad prior">
              <span>Prior</span>
              <select aria-label="Prior" value={setup.prior} onChange={(e) => set('prior', e.target.value as PriorChoice)}>
                <option value="sector">The stock&apos;s S&amp;P 500 sector</option>
                <option value="broad">Broad (the network&apos;s training range)</option>
              </select>
            </label>
          )}
          {setup.measure === 'P' && (
            <>
              <label className="lab-field">
                <span>From</span>
                <input type="date" aria-label="From" value={setup.startDate} max={setup.endDate || undefined}
                  onChange={(e) => onChange({ ...setup, startDate: e.target.value, windowYears: null })} />
              </label>
              <label className="lab-field">
                <span>To</span>
                <input type="date" aria-label="To" value={setup.endDate} min={setup.startDate || undefined}
                  onChange={(e) => onChange({ ...setup, endDate: e.target.value, windowYears: null })} />
              </label>
              <label className="lab-field lab-check">
                <input type="checkbox" checked={setup.includeDrift} onChange={(e) => set('includeDrift', e.target.checked)} />
                <span>Report the historical drift</span>
              </label>
              <p className="lab-hint lab-more-note">
                Several stocks at once: separate the tickers with commas. A Bayesian fit reads at most five years and
                needs at least one. The historical drift is reported with its error; simulations drift at r − q either
                way.
              </p>
            </>
          )}
          {setup.measure === 'Q' && (
            <>
              <label className="lab-field">
                <span>Risk-free rate</span>
                <input type="number" aria-label="Risk-free rate" step={0.005} min={-0.05} max={0.25}
                  value={setup.riskFreeRate} onChange={(e) => set('riskFreeRate', Number(e.target.value))} />
              </label>
              {(setup.model === 'heston' || setup.model === 'bates') && (
                <label className="lab-field">
                  <span>Shortest expiry</span>
                  <select aria-label="Shortest expiry" value={setup.minDays}
                    onChange={(e) => set('minDays', Number(e.target.value))}>
                    {[7, 14, 30, 60].map((d) => <option key={d} value={d}>{d} days</option>)}
                  </select>
                </label>
              )}
              <p className="lab-hint lab-more-note">
                Options expiring within a few weeks carry scheduled events such as earnings, which neither Heston nor
                Bates has: fitted to them, both run to extreme mean reversion and vol of vol.
              </p>
            </>
          )}
        </div>
      )}
    </section>
  );
};

export default CalibrationBar;
