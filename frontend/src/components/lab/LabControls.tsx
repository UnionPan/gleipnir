import React from 'react';
import type { LabCatalog, ScenarioName } from '../../services/labApi';

export interface LabConfig {
  asset: string;
  rhoAssumed: boolean;
  scenario: ScenarioName | null;
  severity: Record<ScenarioName, number>;
  isCall: boolean;
  moneyness: number;
  maturityDays: number;
  costBps: number;
  seed: number;
}

const MATURITY_LABELS: Record<number, string> = { 5: '1 week', 21: '1 month', 63: '3 months' };

interface Props {
  catalog: LabCatalog;
  config: LabConfig;
  onChange: (config: LabConfig) => void;
  onRun: () => void;
  running: boolean;
  stale: boolean;
}

const LabControls: React.FC<Props> = ({ catalog, config, onChange, onRun, running, stale }) => {
  const set = <K extends keyof LabConfig>(key: K, value: LabConfig[K]) => onChange({ ...config, [key]: value });
  const asset = catalog.assets.find((a) => a.ticker === config.asset);
  const scenario = catalog.scenarios.find((s) => s.name === config.scenario);

  return (
    <section className="lab-panel lab-controls" aria-label="Experiment setup">
      <div className="lab-titlebar">Setup</div>
      <div className="lab-controls-grid">
        <fieldset className="lab-field-group">
          <legend>Market</legend>
          <label className="lab-field">
            <span>Asset</span>
            <select value={config.asset} onChange={(e) => set('asset', e.target.value)}>
              {catalog.assets.map((a) => (
                <option key={a.ticker} value={a.ticker}>
                  {a.ticker} · {a.sector}
                </option>
              ))}
            </select>
          </label>
          <div className="lab-field">
            <span>Spot-vol correlation ρ</span>
            <div className="lab-segmented" role="radiogroup" aria-label="Correlation assumption">
              <button type="button" role="radio" aria-checked={!config.rhoAssumed}
                className={!config.rhoAssumed ? 'active' : ''} onClick={() => set('rhoAssumed', false)}>
                Estimated {asset ? asset.params.rho.toFixed(2) : ''}
              </button>
              <button type="button" role="radio" aria-checked={config.rhoAssumed}
                className={config.rhoAssumed ? 'active' : ''} onClick={() => set('rhoAssumed', true)}>
                Assumed {catalog.rho_assumption.toFixed(2)}
              </button>
            </div>
          </div>
        </fieldset>

        <fieldset className="lab-field-group">
          <legend>World</legend>
          <div className="lab-world-options" role="radiogroup" aria-label="Counterfactual world">
            <label className={`lab-radio ${config.scenario === null ? 'active' : ''}`}>
              <input type="radio" name="world" checked={config.scenario === null}
                onChange={() => set('scenario', null)} />
              Factual only
            </label>
            {catalog.scenarios.map((s) => (
              <label key={s.name} className={`lab-radio ${config.scenario === s.name ? 'active' : ''}`}>
                <input type="radio" name="world" checked={config.scenario === s.name}
                  onChange={() => set('scenario', s.name)} />
                {s.label}
              </label>
            ))}
          </div>
          {scenario && (
            <label className="lab-field">
              <span>
                Severity <strong>{config.severity[scenario.name].toFixed(2)}</strong>
              </span>
              <input type="range" min={scenario.severity_min} max={scenario.severity_max} step={0.05}
                value={config.severity[scenario.name]}
                onChange={(e) => set('severity', { ...config.severity, [scenario.name]: Number(e.target.value) })} />
              <small className="lab-hint">{scenario.summary}</small>
            </label>
          )}
        </fieldset>

        <fieldset className="lab-field-group">
          <legend>Option sold</legend>
          <div className="lab-field">
            <span>Type</span>
            <div className="lab-segmented" role="radiogroup" aria-label="Option type">
              <button type="button" role="radio" aria-checked={config.isCall}
                className={config.isCall ? 'active' : ''} onClick={() => set('isCall', true)}>Call</button>
              <button type="button" role="radio" aria-checked={!config.isCall}
                className={!config.isCall ? 'active' : ''} onClick={() => set('isCall', false)}>Put</button>
            </div>
          </div>
          <label className="lab-field">
            <span>
              Strike <strong>{Math.round(config.moneyness * 100)}%</strong> of spot
            </span>
            <input type="range" min={0.8} max={1.2} step={0.05} value={config.moneyness}
              onChange={(e) => set('moneyness', Number(e.target.value))} />
          </label>
          <label className="lab-field">
            <span>Expiry</span>
            <select value={config.maturityDays} onChange={(e) => set('maturityDays', Number(e.target.value))}>
              {catalog.maturities_days.map((d) => (
                <option key={d} value={d}>{MATURITY_LABELS[d] ?? `${d} days`}</option>
              ))}
            </select>
          </label>
          <label className="lab-field">
            <span>Trading cost (bps)</span>
            <input type="number" min={0} max={50} step={1} value={config.costBps}
              onChange={(e) => set('costBps', Math.min(50, Math.max(0, Number(e.target.value))))} />
          </label>
        </fieldset>
      </div>
      <div className="lab-run-row">
        <button type="button" className="lab-run" onClick={onRun} disabled={running}>
          {running ? 'Simulating…' : 'Run experiment'}
        </button>
        <button type="button" className="lab-secondary" disabled={running}
          onClick={() => onChange({ ...config, seed: config.seed + 1 })}
          title="Same experiment on a fresh set of random paths">
          New random paths
        </button>
        <span className="lab-run-note" aria-live="polite">
          {stale && !running ? 'Settings changed since the last run.' : `${catalog.n_paths.toLocaleString()} paths per world · seed ${config.seed}`}
        </span>
      </div>
    </section>
  );
};

export default LabControls;
