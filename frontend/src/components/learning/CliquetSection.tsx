import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import type { Data } from 'plotly.js';
import type { CliquetBaselineKey, CliquetDemo, CliquetMethodKey } from '../../services/rlApi';
import { baseLayout, chrome, PLOT_CONFIG, ThemeName } from '../lab/chartTheme';
import { slotColor } from './learningTheme';
import { Chip, labeled, SeriesLegend } from './DeepHedgingSection';
import { useNarrow } from './useNarrow';
import { endLabels } from './endLabels';

interface Props {
  demo: CliquetDemo;
  theme: ThemeName;
}

const BASELINES: CliquetBaselineKey[] = ['no_hedge', 'static_vanilla', 'oracle_cap_exit'];
const METHODS: CliquetMethodKey[] = ['reactive', 'sequential', 'concurrent', 'oracle_state'];
// Learning curves: the three methods that must learn from observations, colored by slot;
// the exact-state learner is the reference, drawn in neutral ink.
const CURVE_SLOT: Partial<Record<CliquetMethodKey, number>> = { reactive: 0, sequential: 1, concurrent: 2 };
const PROBES = [
  { key: 'period_return_probe_r2' as const, label: 'Return since the last reset' },
  { key: 'accrued_return_probe_r2' as const, label: 'Capped return accrued so far' },
];

function meanCurve(curves: { iteration: number[]; terminal_pnl_rmse: number[] }[]) {
  const n = Math.min(...curves.map((c) => c.iteration.length));
  const x = curves[0].iteration.slice(0, n);
  const y = x.map((_, i) => curves.reduce((s, c) => s + c.terminal_pnl_rmse[i], 0) / curves.length);
  return { x, y };
}

const CliquetSection: React.FC<Props> = ({ demo, theme }) => {
  const [asTable, setAsTable] = useState(false);
  const narrow = useNarrow();
  const c = chrome(theme);
  const { setup, methods, baselines } = demo;

  const curveTraces: Data[] = METHODS.map((k) => {
    const { x, y } = meanCurve(methods[k].curves);
    const slot = CURVE_SLOT[k];
    return {
      type: 'scatter', mode: 'lines', name: methods[k].label, x, y,
      line: slot === undefined ? { color: c.muted, width: 1.5, dash: 'dash' } : { color: slotColor(theme, slot), width: 2 },
      hovertemplate: '%{y:.3f}<extra>' + methods[k].label + '</extra>',
    };
  });
  const curveSeries = METHODS.map((k) => {
    const { x, y } = meanCurve(methods[k].curves);
    const slot = CURVE_SLOT[k];
    return { name: methods[k].label, x, y, color: slot === undefined ? c.muted : slotColor(theme, slot) };
  });
  const curveLabels = endLabels(curveSeries);
  // method names break onto lines under their bars instead of slanting: one word a line on phones
  const wrapped = (label: string) => (narrow ? label.replace(/ /g, '<br>') : label.replace(/ (?=\S+$)|(?<=^\S+) /, '<br>'));

  const probeTraces: Data[] = PROBES.map((p, i) => ({
    type: 'bar', name: p.label, x: METHODS.map((k) => wrapped(methods[k].label)), y: METHODS.map((k) => methods[k][p.key].mean),
    error_y: { type: 'data', array: METHODS.map((k) => methods[k][p.key].sd), color: c.muted, thickness: 1 },
    marker: { color: slotColor(theme, i) },
    text: METHODS.map((k) => methods[k][p.key].mean.toFixed(2)), textposition: 'outside', textfont: { color: c.inkSecondary, size: 11 },
    hovertemplate: '%{y:.3f}<extra>' + p.label + '</extra>',
  }));

  return (
    <section className="lab-panel" aria-label="AIS cliquet hedging">
      <div className="lab-titlebar">Approximate information states · hedging a cliquet</div>
      <div className="lab-panel-body">
        <p className="lab-caption">
          A desk is short a {setup.contract}. It hedges with {setup.hedge}, in a {setup.market} market. What it owes
          depends on the contract&apos;s history, but {setup.hidden}: an agent that acts on today&apos;s observation alone
          cannot tell two paths apart that the contract treats very differently.
        </p>
        <p className="lab-caption">
          An approximate information state (AIS) is a learned summary of the history that is enough to predict the
          next reward and observation. Here a recurrent network learns it by self-prediction on a fast timescale while
          the hedging policy learns on top of it, by policy gradient, on a slow one. The paper this comes from bounds
          how much the summary&apos;s prediction errors can cost the policy.
        </p>

        <div className="lab-section-head">
          <h3 className="learn-subhead">Results</h3>
          <button type="button" className="lab-link" onClick={() => setAsTable(!asTable)}>{asTable ? 'Charts' : 'Tables'}</button>
        </div>
        <div className="lab-table-scroll">
          <table className="lab-results learn-results">
            <caption>
              Terminal P&amp;L error after the contract settles, on paths shared by every method. Learned methods: mean
              ± standard deviation over {setup.training.seeds.length} seeds of {setup.training.iterations.toLocaleString('en-US')}{' '}
              iterations ({setup.training.batch_size} paths each).
            </caption>
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

        {asTable ? (
          <div className="lab-table-scroll">
            <table className="lab-data-table">
              <caption>How much of the hidden contract state each learned summary holds (probe R²)</caption>
              <thead>
                <tr>
                  <th scope="col">Method</th>
                  {PROBES.map((p) => <th key={p.key} scope="col">{p.label}</th>)}
                </tr>
              </thead>
              <tbody>
                {METHODS.map((k) => (
                  <tr key={k}>
                    <td>{methods[k].label}</td>
                    {PROBES.map((p) => <td key={p.key}>{methods[k][p.key].mean.toFixed(3)} ± {methods[k][p.key].sd.toFixed(3)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="lab-multiples learn-two">
            <figure className="lab-chart">
              <figcaption>
                <span className="lab-chart-title">How much of the hidden contract state each summary holds</span>
              </figcaption>
              <div className="learn-legend">
                {PROBES.map((p, i) => <span key={p.key}><Chip color={slotColor(theme, i)} />{p.label}</span>)}
              </div>
              <Plot
                data={probeTraces}
                layout={baseLayout(theme, {
                  height: 250, barmode: 'group', bargap: 0.25, bargroupgap: 0.08, hovermode: 'closest',
                  xaxis: { tickangle: 0 },
                  yaxis: { title: { text: 'Linear probe R²' }, range: [0, 1.15] },
                })}
                config={PLOT_CONFIG}
                useResizeHandler
                style={{ width: '100%' }}
              />
            </figure>
            <figure className="lab-chart">
              <figcaption><span className="lab-chart-title">Terminal error while learning (mean of seeds)</span></figcaption>
              <p className="lab-chart-note">
                Pretrain then freeze spends its first half learning only the summary, so its policy starts late.
              </p>
              {narrow && <SeriesLegend series={curveSeries} />}
              <Plot
                data={curveTraces}
                layout={baseLayout(theme, {
                  height: 250, hovermode: 'closest', ...labeled(narrow, curveLabels, 160),
                  xaxis: { title: { text: 'Iteration' }, range: curveLabels.xRange },
                  yaxis: { title: { text: 'Error, root mean square' } },
                })}
                config={PLOT_CONFIG}
                useResizeHandler
                style={{ width: '100%' }}
              />
            </figure>
          </div>
        )}

        <div className="learn-ledger">
          <div>
            <h4>What the evidence supports</h4>
            <ul>
              <li>The cliquet creates history aliasing that matters for the decision: the hidden reset spot and accrued return.</li>
              <li>The recurrent summary holds more of that history than the reactive one (the probes).</li>
              <li>Learning the summary and the policy together beats freezing a pretrained summary, at the same budget.</li>
              <li>Every learned policy improves substantially on no hedge and on the best static basket.</li>
            </ul>
          </div>
          <div>
            <h4>What it does not</h4>
            <ul>
              <li>That the recurrent learner beats the reactive one: the gap is within the seeds&apos; spread.</li>
              <li>That the exact-state learner is an upper bound, or that the learners have converged.</li>
              <li>Anything about live-market profitability, or generalization beyond three seeds.</li>
            </ul>
          </div>
        </div>

        <p className="learn-provenance">
          {demo.provenance.source} · {demo.provenance.results_file} sha256 {demo.provenance.results_sha256.slice(0, 12)} ·
          calibration {setup.calibration_hash.slice(0, 12)}
        </p>
      </div>
    </section>
  );
};

export default CliquetSection;
