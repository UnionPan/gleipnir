import React from 'react';
import type { CliquetBaselineKey, CliquetDemo, CliquetMethodKey } from '../../services/rlApi';
import { chrome, ThemeName } from '../lab/chartTheme';
import { extent, linear, linePath, ticks, useAnimationClock, useWidth } from '../lab/viz';
import { slotColor } from './learningTheme';
import Bars from './Bars';
import Disclosure from './Disclosure';

interface Props {
  demo: CliquetDemo;
  theme: ThemeName;
}

const BASELINES: CliquetBaselineKey[] = ['no_hedge', 'static_vanilla', 'oracle_cap_exit'];
const METHODS: CliquetMethodKey[] = ['reactive', 'sequential', 'concurrent', 'oracle_state'];
// The three methods that must learn from observations take palette slots; the exact-state
// learner is the reference, drawn in neutral ink, as are the rule-based baselines.
const SLOT: Partial<Record<CliquetMethodKey, number>> = { reactive: 0, sequential: 1, concurrent: 2 };
const DURATION = 7000;
const HEIGHT = 240;
const PAD = { left: 44, right: 12, top: 10, bottom: 34 };

function meanCurve(curves: { iteration: number[]; terminal_pnl_rmse: number[] }[]) {
  const n = Math.min(...curves.map((c) => c.iteration.length));
  const x = curves[0].iteration.slice(0, n);
  const y = x.map((_, i) => curves.reduce((s, c) => s + c.terminal_pnl_rmse[i], 0) / curves.length);
  return { x, y };
}

/** The learners' error while they learn, replayed. */
const TrainingReplay: React.FC<Props> = ({ demo, theme }) => {
  const c = chrome(theme);
  const clock = useAnimationClock(DURATION, demo);
  const [ref, width] = useWidth<HTMLDivElement>(600);
  const curves = METHODS.map((k) => ({ key: k, ...meanCurve(demo.methods[k].curves) }));
  const last = curves[0].x[curves[0].x.length - 1];
  const x = linear([0, last], [PAD.left, width - PAD.right]);
  const domain = extent(curves.flatMap((cv) => cv.y), 0.06);
  const y = linear(domain, [HEIGHT - PAD.bottom, PAD.top]);
  const upTo = (clock.t / DURATION) * (curves[0].x.length - 1);
  const iteration = Math.round(curves[0].x[Math.min(Math.round(upTo), curves[0].x.length - 1)]);
  const color = (k: CliquetMethodKey) => (SLOT[k] === undefined ? c.muted : slotColor(theme, SLOT[k] as number));
  const staticBasket = demo.baselines.static_vanilla.terminal_pnl_rmse;
  return (
    <figure className="lab-chart" ref={ref}>
      <figcaption>
        <span className="lab-chart-title">Terminal error while learning</span>
        <span className="lab-hero-count">iteration {iteration.toLocaleString('en-US')}</span>
      </figcaption>
      <div className="lab-chips" aria-hidden>
        {METHODS.map((k) => (
          <span key={k} className="lab-chip active" style={{ borderColor: color(k) }}>
            <span className="learn-chip" style={{ background: color(k) }} />{demo.methods[k].label}
          </span>
        ))}
      </div>
      <svg width={width} height={HEIGHT} role="img" aria-label="Terminal error of each learner by training iteration">
        {ticks(domain[0], domain[1], 4).map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={width - PAD.right} y1={y(v)} y2={y(v)} stroke={c.grid} />
            <text x={PAD.left - 5} y={y(v) + 4} textAnchor="end" fill={c.muted} fontSize={10}>{v}</text>
          </g>
        ))}
        {ticks(0, last, 4).map((v) => (
          <text key={v} x={x(v)} y={HEIGHT - PAD.bottom + 15} textAnchor="middle" fill={c.muted} fontSize={10}>{v}</text>
        ))}
        <text x={(PAD.left + width - PAD.right) / 2} y={HEIGHT - 3} textAnchor="middle" fill={c.inkSecondary} fontSize={11}>
          Training iteration
        </text>
        {staticBasket > domain[0] && staticBasket < domain[1] && (
          <g>
            <line x1={PAD.left} x2={width - PAD.right} y1={y(staticBasket)} y2={y(staticBasket)} stroke={c.inkSecondary}
              strokeDasharray="5 4" />
            <text x={width - PAD.right - 2} y={y(staticBasket) - 5} textAnchor="end" fill={c.inkSecondary} fontSize={11}
              stroke={c.surface} strokeWidth={4} paintOrder="stroke">best static basket</text>
          </g>
        )}
        {curves.map((cv) => {
          const xs = linear([0, cv.x.length - 1], [x(cv.x[0]), x(cv.x[cv.x.length - 1])]);
          return (
            <path key={cv.key} d={linePath(cv.y, xs, y, upTo)} fill="none" stroke={color(cv.key)}
              strokeWidth={cv.key === 'oracle_state' ? 1.5 : 2} strokeDasharray={cv.key === 'oracle_state' ? '5 3' : undefined} />
          );
        })}
      </svg>
      <div className="learn-replay-controls">
        {clock.playing
          ? <button type="button" className="lab-secondary" onClick={clock.pause}>Pause</button>
          : <button type="button" className="lab-secondary" onClick={clock.done ? clock.replay : clock.resume}>
            {clock.done ? 'Replay training' : 'Play'}
          </button>}
      </div>
      <p className="lab-hero-note">
        Mean of {demo.setup.training.seeds.length} seeds. Pretrain then freeze spends its first half learning only the
        summary, so its policy starts late.
      </p>
    </figure>
  );
};

const CliquetSection: React.FC<Props> = ({ demo, theme }) => {
  const c = chrome(theme);
  const { setup, methods, baselines } = demo;
  const color = (k: CliquetMethodKey) => (SLOT[k] === undefined ? c.muted : slotColor(theme, SLOT[k] as number));
  const errorRows = [
    ...BASELINES.map((k) => ({ key: k, label: baselines[k].label, value: baselines[k].terminal_pnl_rmse, color: c.baseline })),
    ...METHODS.map((k) => ({ key: k, label: methods[k].label, value: methods[k].terminal_pnl_rmse.mean,
      sd: methods[k].terminal_pnl_rmse.sd, color: color(k) })),
  ];
  const probeRows = (key: 'period_return_probe_r2' | 'accrued_return_probe_r2') => METHODS.map((k) => ({
    key: k, label: methods[k].label, value: methods[k][key].mean, sd: methods[k][key].sd, color: color(k),
  }));

  return (
    <div className="learn-tab">
      <div className="lab-verdict">
        <p className="lab-headline">
          The desk owes a payoff that depends on the contract&apos;s past, which it cannot see. A learned memory recovers
          part of it and cuts the error by {(100 * (1 - methods.concurrent.terminal_pnl_rmse.mean / baselines.static_vanilla.terminal_pnl_rmse)).toFixed(0)}%
          {' '}against the best static hedge.
        </p>
        <details className="lab-why">
          <summary>What is a learned memory here?</summary>
          <p>
            An approximate information state (AIS): a summary of the history good enough to predict the next reward and
            observation. A recurrent network learns it by self-prediction on a fast timescale while the hedging policy
            learns on top of it, by policy gradient, on a slow one. The paper this comes from bounds how much the
            summary&apos;s prediction errors can cost the policy.
          </p>
        </details>
      </div>

      <section className="lab-panel" aria-label="Learning">
        <div className="lab-titlebar">Learning to hedge a cliquet</div>
        <div className="lab-panel-body">
          <TrainingReplay demo={demo} theme={theme} />
        </div>
      </section>

      <section className="lab-panel" aria-label="Results">
        <div className="lab-titlebar">After training</div>
        <div className="lab-panel-body learn-bars-row">
          <Bars title="Error at settlement (lower is better)" theme={theme} rows={errorRows} digits={3}
            note="Root mean square P&L error on shared paths. Grey: fixed rules and the exact-state reference. Whiskers: spread across seeds." />
          <div className="learn-bars-stack">
            <Bars title="Remembers the return since the last reset (R²)" theme={theme}
              rows={probeRows('period_return_probe_r2')} />
            <Bars title="Remembers the capped return accrued so far (R²)" theme={theme}
              rows={probeRows('accrued_return_probe_r2')}
              note="How well a linear read-out of each learner's state recovers the hidden contract state. 1: all of it." />
          </div>
        </div>
      </section>

      <Disclosure title="The contract and the market">
        <p className="lab-caption">
          A short {setup.contract}, hedged with {setup.hedge}, in a {setup.market} market (proportional cost{' '}
          {(1e4 * setup.proportional_cost).toFixed(0)} bp). What the desk owes depends on the contract&apos;s history,
          but {setup.hidden}: an agent acting on today&apos;s observation alone cannot tell apart two paths the contract
          treats very differently.
        </p>
      </Disclosure>
      <Disclosure title="All numbers" hint="mean ± sd across seeds">
        <div className="lab-table-scroll">
          <table className="lab-results learn-results">
            <thead>
              <tr>
                <th scope="col">Method</th>
                <th scope="col">Error, root mean square</th>
                <th scope="col">Worst 5%, mean absolute</th>
                <th scope="col">Trading cost</th>
              </tr>
            </thead>
            <tbody>
              {BASELINES.map((k) => (
                <tr key={k}>
                  <th scope="row">{baselines[k].label}<span className="learn-row-note">{baselines[k].summary}</span></th>
                  <td>{baselines[k].terminal_pnl_rmse.toFixed(3)}</td>
                  <td>{baselines[k].terminal_abs_cvar_95.toFixed(3)}</td>
                  <td>{baselines[k].mean_trading_cost.toFixed(4)}</td>
                </tr>
              ))}
              {METHODS.map((k) => (
                <tr key={k} className="learn-learned-row">
                  <th scope="row">{methods[k].label}<span className="learn-row-note">{methods[k].summary}</span></th>
                  <td>{methods[k].terminal_pnl_rmse.mean.toFixed(3)}<span className="learn-sd"> ± {methods[k].terminal_pnl_rmse.sd.toFixed(3)}</span></td>
                  <td>{methods[k].terminal_abs_cvar_95.mean.toFixed(3)}<span className="learn-sd"> ± {methods[k].terminal_abs_cvar_95.sd.toFixed(3)}</span></td>
                  <td>{methods[k].mean_trading_cost.mean.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Disclosure>
      <Disclosure title="What the evidence does and does not support">
        <div className="learn-ledger">
          <div>
            <h4>Supported</h4>
            <ul>
              <li>The cliquet&apos;s hidden reset spot and accrued return matter for the decision.</li>
              <li>The recurrent summary holds more of that history than the reactive one (the probes).</li>
              <li>Learning summary and policy together beats freezing a pretrained summary, at the same budget.</li>
              <li>Every learned policy beats no hedge and the best static basket by a wide margin.</li>
            </ul>
          </div>
          <div>
            <h4>Not supported</h4>
            <ul>
              <li>That the recurrent learner beats the reactive one: the gap is within the seeds&apos; spread.</li>
              <li>That the exact-state learner is an upper bound, or that the learners have converged.</li>
              <li>Anything about live-market profitability, or generalization beyond three seeds.</li>
            </ul>
          </div>
        </div>
      </Disclosure>
      <p className="learn-provenance">
        {demo.provenance.source} · {demo.provenance.results_file} sha256 {demo.provenance.results_sha256.slice(0, 12)} ·
        calibration {setup.calibration_hash.slice(0, 12)}
      </p>
    </div>
  );
};

export default CliquetSection;
