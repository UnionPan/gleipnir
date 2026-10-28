import React, { useState } from 'react';
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

const MATURITY_LABELS: Record<number, string> = { 5: '1 wk', 21: '1 mo', 63: '3 mo' };
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

/** A row of mutually exclusive buttons. */
function Segmented<T extends string | number | boolean>({ label, value, options, onPick }: {
  label: string; value: T; options: { value: T; label: string }[]; onPick: (v: T) => void;
}) {
  return (
    <div className="lab-segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.value)} type="button" role="radio" aria-checked={o.value === value}
          className={o.value === value ? 'active' : ''} onClick={() => onPick(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

const LabControls: React.FC<Props> = ({ catalog, bundles, config, onChange, onRun, running, stale }) => {
  const [more, setMore] = useState(false);
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
      <div className="lab-titlebar">Sell an option and hedge it</div>
      <div className="lab-bar">
        <label className="lab-field">
          <span>Stock</span>
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
            <optgroup label="Calibrated to real prices">
              {catalog.assets.map((a) => <option key={a.ticker} value={a.ticker}>{a.ticker} · {a.sector}</option>)}
            </optgroup>
          </select>
        </label>
        <label className="lab-field" title={model?.summary}>
          <span>Market model</span>
          <select value={config.model} disabled={!!saved}
            onChange={(e) => onChange(withModel(e.target.value as LabModelName))}>
            {catalog.models.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}
          </select>
        </label>
        <div className="lab-field">
          <span>Option sold</span>
          <Segmented label="Option type" value={config.isCall} onPick={(v) => set('isCall', v)}
            options={[{ value: true, label: 'Call' }, { value: false, label: 'Put' }]} />
        </div>
        <label className="lab-field">
          <span>Strike <strong>{Math.round(config.moneyness * 100)}%</strong> of today's price</span>
          <input type="range" min={0.8} max={1.2} step={0.05} value={config.moneyness}
            onChange={(e) => set('moneyness', Number(e.target.value))} />
        </label>
        <div className="lab-field">
          <span>Expires in</span>
          <Segmented label="Expiry" value={config.maturityDays} onPick={(v) => set('maturityDays', v)}
            options={catalog.maturities_days.map((d) => ({ value: d, label: MATURITY_LABELS[d] ?? `${d} days` }))} />
        </div>
        <label className="lab-field">
          <span>What if…</span>
          <select value={config.scenario ?? ''} onChange={(e) => set('scenario', (e.target.value || null) as ScenarioName | null)}>
            <option value="">Nothing changes</option>
            {scenarios.map((s) => <option key={s.name} value={s.name}>{s.label}</option>)}
          </select>
        </label>
        {scenario && (
          <label className="lab-field" title={scenario.summary}>
            <span>Severity <strong>{config.severity[scenario.name].toFixed(2)}</strong></span>
            <input type="range" min={scenario.severity_min} max={scenario.severity_max} step={0.05}
              value={config.severity[scenario.name]}
              onChange={(e) => set('severity', { ...config.severity, [scenario.name]: Number(e.target.value) })} />
          </label>
        )}
        <div className="lab-bar-actions">
          <button type="button" className={`lab-run ${stale ? 'lab-run-stale' : ''}`} onClick={onRun} disabled={running}>
            {running ? 'Simulating…' : stale ? 'Run with new settings' : 'Run again'}
          </button>
          <button type="button" className="lab-link" aria-expanded={more} onClick={() => setMore(!more)}>
            {more ? 'Fewer settings' : 'More settings'}
          </button>
        </div>
      </div>
      {scenario && <p className="lab-bar-hint">{scenario.label}: {scenario.summary}</p>}
      {more && (
        <div className="lab-more">
          <label className="lab-field">
            <span>Trading cost (bps of each trade)</span>
            <input type="number" min={0} max={50} step={1} value={config.costBps}
              onChange={(e) => set('costBps', Math.min(50, Math.max(0, Number(e.target.value))))} />
          </label>
          <label className="lab-field" title="How the learned hedger weighs risk against trading costs">
            <span>Learned hedger</span>
            <select value={config.riskAversion} onChange={(e) => set('riskAversion', Number(e.target.value))}>
              {RISK_AVERSIONS.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
            </select>
          </label>
          {model?.has_rho && (
            <div className="lab-field">
              <span>Spot-vol correlation ρ</span>
              <Segmented label="Correlation assumption" value={config.rhoAssumed} onPick={(v) => set('rhoAssumed', v)}
                options={[
                  { value: false, label: `Estimated ${estimatedRho !== undefined ? estimatedRho.toFixed(2) : ''}` },
                  { value: true, label: `Assumed ${catalog.rho_assumption.toFixed(2)}` },
                ]} />
            </div>
          )}
          <div className="lab-field">
            <span>Random paths (seed {config.seed})</span>
            <button type="button" className="lab-secondary" disabled={running}
              onClick={() => onChange({ ...config, seed: config.seed + 1 })}>Draw new paths</button>
          </div>
          {model && <p className="lab-hint lab-more-note">{model.label}: {model.summary}</p>}
          {saved && (
            <p className="lab-hint lab-more-note">
              Your fit from the Calibration tab, fingerprint <code>{saved.bundle.hash.slice(0, 8)}</code>.
            </p>
          )}
        </div>
      )}
    </section>
  );
};

export default LabControls;
