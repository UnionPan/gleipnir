import React from 'react';
import type { LabCatalog, LabModelName, ScenarioName } from '../../services/labApi';
import type { StoredBundle } from '../../services/labBundles';

// The learned band's risk aversion: it was trained across 0.5 to 20.
const RISK_AVERSIONS = [
  { value: 20, label: 'Cautious: close to delta' },
  { value: 5, label: 'Balanced' },
  { value: 1, label: 'Cost-conscious: trades less' },
];

export interface LabConfig {
  asset: string;
  /** A saved calibration (labBundles key) in place of a shipped asset. */
  bundleKey: string | null;
  model: LabModelName;
  rhoAssumed: boolean;
  scenario: ScenarioName | null;
  severity: Record<ScenarioName, number>;
  isCall: boolean;
  moneyness: number;
  maturityDays: number;
  costBps: number;
  riskAversion: number;
  seed: number;
}

const MATURITY_LABELS: Record<number, string> = { 5: '1 week', 21: '1 month', 63: '3 months' };
const BUNDLE_PREFIX = 'bundle:';

interface Props {
  catalog: LabCatalog;
  bundles: StoredBundle[];
  config: LabConfig;
  onChange: (config: LabConfig) => void;
  onRun: () => void;
  running: boolean;
  stale: boolean;
}

const LabControls: React.FC<Props> = ({ catalog, bundles, config, onChange, onRun, running, stale }) => {
  const set = <K extends keyof LabConfig>(key: K, value: LabConfig[K]) => onChange({ ...config, [key]: value });
  const asset = catalog.assets.find((a) => a.ticker === config.asset);
  const saved = bundles.find((b) => b.key === config.bundleKey);
  const model = catalog.models.find((m) => m.name === config.model);
  const estimatedRho = saved ? saved.bundle.parameters.rho : asset?.models[config.model].params.rho;
  const setSource = (value: string) => {
    if (value.startsWith(BUNDLE_PREFIX)) {
      const key = value.slice(BUNDLE_PREFIX.length);
      const picked = bundles.find((b) => b.key === key);
      if (picked) onChange({ ...withModel(picked.bundle.model), bundleKey: key });
    } else {
      onChange({ ...config, asset: value, bundleKey: null });
    }
  };
  const scenarios = catalog.scenarios.filter((s) => s.models.includes(config.model));
  const scenario = scenarios.find((s) => s.name === config.scenario);
  function withModel(name: LabModelName): LabConfig {
    // a scenario that means nothing for the new model falls back to a volatility spike
    const keeps = config.scenario === null
      || catalog.scenarios.some((s) => s.name === config.scenario && s.models.includes(name));
    return { ...config, model: name, scenario: keeps ? config.scenario : 'vol_spike' };
  }

  return (
    <section className="lab-panel lab-controls" aria-label="Experiment setup">
      <div className="lab-titlebar">Setup</div>
      <div className="lab-controls-grid">
        <fieldset className="lab-field-group">
          <legend>Market</legend>
          <label className="lab-field">
            <span>Asset</span>
            <select value={saved ? `${BUNDLE_PREFIX}${saved.key}` : config.asset} onChange={(e) => setSource(e.target.value)}>
              {bundles.length > 0 && (
                <optgroup label="Your calibrations">
                  {bundles.map((b) => (
                    <option key={b.key} value={`${BUNDLE_PREFIX}${b.key}`}>
                      {b.bundle.asset} · {catalog.models.find((m) => m.name === b.bundle.model)?.label ?? b.bundle.model}
                      {' · '}{b.bundle.as_of}
                    </option>
                  ))}
                </optgroup>
              )}
              <optgroup label="Shipped calibrations">
                {catalog.assets.map((a) => (
                  <option key={a.ticker} value={a.ticker}>
                    {a.ticker} · {a.sector}
                  </option>
                ))}
              </optgroup>
            </select>
            {saved && (
              <small className="lab-hint">
                Your fit from the Calibration tab, fingerprint <code>{saved.bundle.hash.slice(0, 8)}</code>.
              </small>
            )}
          </label>
          <label className="lab-field">
            <span>Model</span>
            <select value={config.model} disabled={!!saved}
              onChange={(e) => onChange(withModel(e.target.value as LabModelName))}>
              {catalog.models.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}
            </select>
            {model && <small className="lab-hint">{saved ? 'The model you calibrated. ' : ''}{model.summary}</small>}
          </label>
          {model?.has_rho && (
            <div className="lab-field">
              <span>Spot-vol correlation ρ</span>
              <div className="lab-segmented" role="radiogroup" aria-label="Correlation assumption">
                <button type="button" role="radio" aria-checked={!config.rhoAssumed}
                  className={!config.rhoAssumed ? 'active' : ''} onClick={() => set('rhoAssumed', false)}>
                  Estimated {estimatedRho !== undefined ? estimatedRho.toFixed(2) : ''}
                </button>
                <button type="button" role="radio" aria-checked={config.rhoAssumed}
                  className={config.rhoAssumed ? 'active' : ''} onClick={() => set('rhoAssumed', true)}>
                  Assumed {catalog.rho_assumption.toFixed(2)}
                </button>
              </div>
            </div>
          )}
        </fieldset>

        <fieldset className="lab-field-group">
          <legend>World</legend>
          <div className="lab-world-options" role="radiogroup" aria-label="Counterfactual world">
            <label className={`lab-radio ${config.scenario === null ? 'active' : ''}`}>
              <input type="radio" name="world" checked={config.scenario === null}
                onChange={() => set('scenario', null)} />
              Factual only
            </label>
            {scenarios.map((s) => (
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
          <label className="lab-field" title="How the learned hedger weighs risk against trading costs">
            <span>Learned hedger</span>
            <select value={config.riskAversion} onChange={(e) => set('riskAversion', Number(e.target.value))}>
              {RISK_AVERSIONS.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
            </select>
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
