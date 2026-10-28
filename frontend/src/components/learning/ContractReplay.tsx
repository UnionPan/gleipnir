import React, { useState } from 'react';
import type { CliquetDemo, CliquetPaths, CliquetPolicyKey } from '../../services/rlApi';
import { chrome, ThemeName } from '../lab/chartTheme';
import { useAnimationClock } from '../lab/viz';
import { cliquetColor } from './learningTheme';
import { ReplayChart } from './YearReplay';

interface Props {
  demo: CliquetDemo;
  paths: CliquetPaths;
  theme: ThemeName;
}

const DURATION = 8000;
const SAMPLE_NAMES = ['Price ends low', 'Price ends near the middle', 'Price ends high'];
export const POLICIES: CliquetPolicyKey[] = [
  'no_hedge', 'static_vanilla', 'oracle_cap_exit', 'reactive', 'sequential', 'concurrent', 'oracle_state',
];
const DEFAULT_SHOWN: CliquetPolicyKey[] = ['static_vanilla', 'reactive', 'concurrent', 'oracle_state'];

export function policyLabel(demo: CliquetDemo, key: CliquetPolicyKey): string {
  return key in demo.methods
    ? demo.methods[key as keyof CliquetDemo['methods']].label
    : demo.baselines[key as keyof CliquetDemo['baselines']].label;
}

/**
 * One contract, period by period: the price and its resets, the payoff the
 * cliquet locks in at each reset, and what each hedge holds and is worth.
 */
const ContractReplay: React.FC<Props> = ({ demo, paths, theme }) => {
  const c = chrome(theme);
  const [sampleIndex, setSampleIndex] = useState(1);
  const [shown, setShown] = useState<Set<CliquetPolicyKey>>(new Set(DEFAULT_SHOWN));
  const clock = useAnimationClock(DURATION, sampleIndex);
  const sample = paths.samples[sampleIndex];
  const steps = paths.steps;
  const upTo = (clock.t / DURATION) * steps;
  const period = Math.min(Math.round(upTo), steps);
  const policies = POLICIES.filter((k) => shown.has(k));
  const cap = paths.notional * paths.global_cap;
  const toggle = (key: CliquetPolicyKey) => setShown((prev) => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  return (
    <div className="learn-replay">
      <div className="learn-replay-head">
        <div className="learn-toggle" role="radiogroup" aria-label="Sample contract">
          {SAMPLE_NAMES.map((name, i) => (
            <button key={name} type="button" role="radio" aria-checked={i === sampleIndex}
              className={i === sampleIndex ? 'active' : ''} onClick={() => setSampleIndex(i)}>{name}</button>
          ))}
        </div>
        <div className="lab-chips" role="group" aria-label="Hedges shown">
          {POLICIES.map((k) => (
            <button key={k} type="button" aria-pressed={shown.has(k)}
              className={`lab-chip ${shown.has(k) ? 'active' : ''}`} onClick={() => toggle(k)}
              style={shown.has(k) ? { borderColor: cliquetColor(theme, k) } : undefined}>
              <span className="learn-chip" style={{ background: cliquetColor(theme, k) }} aria-hidden />
              {policyLabel(demo, k)}
            </button>
          ))}
        </div>
      </div>

      <div className="learn-replay-charts learn-replay-quad">
        <ReplayChart title="Price, and the resets" days={steps} upTo={upTo} theme={theme} height={170} digits={0}
          marks={paths.reset_steps} note="dotted: a reset"
          lines={[{ key: 'spot', values: sample.spot, color: c.ink }]} />
        <ReplayChart title="Payoff locked in so far" days={steps} upTo={upTo} theme={theme} height={170} digits={1}
          marks={paths.reset_steps} reference={cap} referenceLabel="cap"
          lines={[{ key: 'locked', values: sample.locked_in, color: c.ink, width: 2.4 }]} />
        <ReplayChart title="Each hedge's running P&L" days={steps} upTo={upTo} theme={theme} height={170} digits={1}
          reference={0}
          lines={policies.map((k) => ({ key: k, values: sample.pnl[k], color: cliquetColor(theme, k),
            dash: k === 'oracle_state' ? '5 3' : undefined }))} />
        <ReplayChart title="Basket held each period" days={steps} upTo={upTo} theme={theme} height={170} digits={1}
          lines={policies.map((k) => ({ key: k, values: sample.exposure[k], color: cliquetColor(theme, k), step: true,
            dash: k === 'oracle_state' ? '5 3' : undefined }))} />
      </div>

      <div className="learn-replay-controls">
        {clock.playing
          ? <button type="button" className="lab-secondary" onClick={clock.pause}>Pause</button>
          : <button type="button" className="lab-secondary" onClick={clock.done ? clock.replay : clock.resume}>
            {clock.done ? 'Replay' : 'Play'}
          </button>}
        <input type="range" min={0} max={steps} step={1} value={period} aria-label="Period"
          onChange={(e) => clock.seek((Number(e.target.value) / steps) * DURATION)} />
        <span className="lab-hero-count">Period {period} of {steps} · day {period * paths.days_per_step}</span>
      </div>
      <p className="lab-readout" aria-live="off">
        <strong>Price {sample.spot[period].toFixed(2)}</strong>
        <span>locked in {sample.locked_in[period].toFixed(2)} of a cap of {cap.toFixed(0)}</span>
        {policies.map((k) => (
          <span key={k}>
            <span className="learn-chip" style={{ background: cliquetColor(theme, k) }} aria-hidden />
            {policyLabel(demo, k)} {sample.pnl[k][period] >= 0 ? '+' : ''}{sample.pnl[k][period].toFixed(2)}
          </span>
        ))}
      </p>
      <p className="lab-hero-note">
        One of {paths.evaluation_paths.toLocaleString('en-US')} contracts. At each reset (every{' '}
        {(paths.reset_steps[1] - paths.reset_steps[0]) * paths.days_per_step} trading days) the period&apos;s return, capped at{' '}
        {(100 * paths.local_cap).toFixed(1)}%, is locked into the payoff, until the total reaches its cap of{' '}
        {(100 * paths.global_cap).toFixed(0)}%. Once capped, the desk owes nothing more and should stop hedging, but
        only a hedge that remembers the resets can tell. The running P&amp;L is the premium of{' '}
        {paths.premium.toFixed(2)}, plus the basket&apos;s gains, less the payoff locked in.
      </p>
    </div>
  );
};

export default ContractReplay;
