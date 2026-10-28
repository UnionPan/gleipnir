import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import type { Data } from 'plotly.js';
import type { DeepHedgingDemo, HedgerKey, SeedStat } from '../../services/rlApi';
import { baseLayout, chrome, PLOT_CONFIG, ThemeName } from '../lab/chartTheme';
import { hedgerColor } from './learningTheme';
import { endLabels, LabeledSeries } from './endLabels';
import { verdict } from './verdict';
import { useNarrow } from './useNarrow';

interface Props {
  demo: DeepHedgingDemo;
  theme: ThemeName;
}

const LEARNED = new Set<HedgerKey>(['lstm_underlying', 'lstm_options']);
const PATH_HEDGERS: HedgerKey[] = ['bs_delta', 'delta_gamma', 'lstm_options'];
const OPTION_HEDGERS: HedgerKey[] = ['delta_gamma', 'lstm_options'];
const PATH_NAMES = ['Spot ends low', 'Spot ends near the middle', 'Spot ends high'];

const isStat = (v: number | SeedStat): v is SeedStat => typeof v === 'object';
const meanOf = (v: number | SeedStat) => (isStat(v) ? v.mean : v);
function Cell({ value, digits = 2 }: { value: number | SeedStat; digits?: number }) {
  if (!isStat(value)) return <td>{value.toFixed(digits)}</td>;
  return (
    <td>
      {value.mean.toFixed(digits)}
      <span className="learn-sd"> ± {value.sd.toFixed(digits)}</span>
    </td>
  );
}

/** One color chip next to every hedger name: identity never rests on color alone. */
export const Chip: React.FC<{ color: string }> = ({ color }) => (
  <span className="learn-chip" style={{ background: color }} aria-hidden />
);

/** The legend that replaces end labels on narrow screens. */
export const SeriesLegend: React.FC<{ series: Pick<LabeledSeries, 'name' | 'color'>[] }> = ({ series }) => (
  <div className="learn-legend">
    {series.map((s) => <span key={s.name}><Chip color={s.color} />{s.name}</span>)}
  </div>
);

/** Layout pieces for a labeled line chart: end labels when wide, a legend above when narrow. */
export function labeled(narrow: boolean, labels: ReturnType<typeof endLabels>, wideRightMargin: number) {
  return narrow
    ? { margin: { l: 48, r: 12, t: 8, b: 40 }, annotations: [] }
    : { margin: { l: 48, r: wideRightMargin, t: 8, b: 40 }, annotations: labels.annotations };
}

const days = (n: number) => [...Array(n).keys()];
const withArticle = (noun: string) => `${/^[aeiou]/i.test(noun) ? 'an' : 'a'} ${noun}`;

const DeepHedgingSection: React.FC<Props> = ({ demo, theme }) => {
  const [pathIndex, setPathIndex] = useState(1);
  const [histTable, setHistTable] = useState(false);
  const [pathTable, setPathTable] = useState(false);
  const narrow = useNarrow();
  const c = chrome(theme);
  const { setup, results } = demo;
  const label = Object.fromEntries(demo.hedgers.map((h) => [h.key, h.label])) as Record<HedgerKey, string>;
  const fairValues = demo.hedgers.map((h) => -meanOf(results[h.key].mean_error) - meanOf(results[h.key].mean_cost));
  const edges = demo.histogram.edges;
  const centers = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
  const total = demo.histogram.bs_delta.counts.reduce((a, b) => a + b, 0);
  const share = (key: HedgerKey) => demo.histogram[key].counts.map((n) => (100 * n) / total);
  const sample = demo.samples[pathIndex];
  const horizon = sample.spot.length - 1;
  const learned = results.lstm_options;

  const lineSeries = (keys: HedgerKey[], pick: (k: HedgerKey) => number[]) =>
    keys.map((k) => ({ name: label[k], x: days(horizon), y: pick(k), color: hedgerColor(theme, k) }));
  const lines = (series: ReturnType<typeof lineSeries>): Data[] => series.map((s) => ({
    type: 'scatter', mode: 'lines', name: s.name, x: s.x, y: s.y,
    line: { color: s.color, width: 2 }, hovertemplate: '%{y:.3f}<extra>' + s.name + '</extra>',
  }));
  const positionSeries = lineSeries(PATH_HEDGERS, (k) => sample.positions[k]);
  const optionSeries = lineSeries(OPTION_HEDGERS, (k) => sample.option_value[k]);

  const curveTraces: Data[] = (['lstm_underlying', 'lstm_options'] as const).flatMap((k) =>
    Object.entries(demo.training_curves[k] ?? {}).map(([seed, curve]) => ({
      type: 'scatter', mode: 'lines', name: `${label[k]}, seed ${seed}`, x: curve.epoch, y: curve.std_error,
      line: { color: hedgerColor(theme, k), width: 1.5 },
      hovertemplate: '%{y:.2f}<extra>' + label[k] + ', seed ' + seed + '</extra>',
    })));
  // one label per method, at the end of its median seed's curve
  const curveLabels = endLabels((['lstm_underlying', 'lstm_options'] as const).map((k) => {
    const seeds = Object.values(demo.training_curves[k] ?? {});
    const ends = seeds.map((s) => s.std_error[s.std_error.length - 1]).sort((a, b) => a - b);
    const all = seeds.flatMap((s) => s.std_error);
    return { name: label[k], x: seeds[0]?.epoch ?? [0], y: [...all, ends[Math.floor(ends.length / 2)]],
      color: hedgerColor(theme, k) };
  }), { logY: true });
  const positionLabels = endLabels(positionSeries);
  const optionLabels = endLabels(optionSeries);
  const pathX = { title: { text: 'Trading day' }, range: [0, horizon] };

  return (
    <section className="lab-panel" aria-label="Deep hedging">
      <div className="lab-titlebar">Deep hedging · a short call in a Heston world</div>
      <div className="lab-panel-body">
        <p className="lab-caption">
          A desk sells {withArticle(setup.liability.replace('short ', ''))} and hedges it daily. The market is Heston
          (κ {setup.market.kappa}, θ {setup.market.theta}, ξ {setup.market.sigma_v}, ρ {setup.market.rho}) with no rates or
          dividends, so the pricing and real-world measures coincide. The hedgers can trade the underlying (cost{' '}
          {(1e4 * setup.costs.underlying).toFixed(0)} bp) and {setup.options} (cost {(100 * setup.costs.options).toFixed(0)}%
          of premium). The LSTM learns its trades by minimizing {setup.objective}, differentiating through the
          simulated market; it sees the spot, the variance, the options&apos; prices, strikes and tenors, and its holdings.
        </p>

        <div className="lab-table-scroll">
          <table className="lab-results learn-results">
            <caption>
              {setup.eval_paths.toLocaleString('en-US')} evaluation paths shared by every hedger; the LSTMs are trained
              from {setup.training.seeds.length} seed{setup.training.seeds.length === 1 ? '' : 's'} ({setup.training.n_epochs} epochs
              of {setup.training.batch_size.toLocaleString('en-US')} paths)
              and shown as mean ± standard deviation across them.
            </caption>
            <thead>
              <tr>
                <th scope="col">Hedger</th>
                <th scope="col">Hedging error, sd</th>
                <th scope="col">Worst 5%, mean</th>
                <th scope="col">Trading cost</th>
                <th scope="col">Options traded</th>
              </tr>
            </thead>
            <tbody>
              {demo.hedgers.map((h) => {
                const r = results[h.key];
                return (
                  <tr key={h.key} className={LEARNED.has(h.key) ? 'learn-learned-row' : undefined}>
                    <th scope="row">
                      <Chip color={hedgerColor(theme, h.key)} />
                      {h.label}
                      <span className="learn-row-note">{h.summary}</span>
                    </th>
                    <Cell value={r.std_error as number | SeedStat} />
                    <Cell value={r.expected_shortfall_95 as number | SeedStat} />
                    <Cell value={r.mean_cost as number | SeedStat} digits={3} />
                    <Cell value={r.option_turnover as number | SeedStat} />
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="lab-caption">
          The worst 5% is the mean shortfall of the worst twentieth of paths, measured from each hedger&apos;s own average.
          Options traded is the premium bought and sold over the year. Before costs, every hedger&apos;s average outcome
          is the same up to sampling noise ({Math.min(...fairValues).toFixed(2)} to {Math.max(...fairValues).toFixed(2)}{' '}
          on these paths, whose average payoff is {setup.mean_payoff.toFixed(2)}): no strategy earns money on average
          here, so all of the difference between hedgers is risk and cost.
        </p>
        <div className="learn-verdict">
          {verdict(demo, label).map((sentence) => <p key={sentence}>{sentence}</p>)}
        </div>

        <div className="lab-section-head">
          <h3 className="learn-subhead">How the errors are spread</h3>
          <button type="button" className="lab-link" onClick={() => setHistTable(!histTable)}>{histTable ? 'Charts' : 'Table'}</button>
        </div>
        {histTable ? (
          <div className="lab-table-scroll lab-table-tall">
            <table className="lab-data-table">
              <thead>
                <tr>
                  <th scope="col">Error from</th>
                  <th scope="col">to</th>
                  {demo.hedgers.map((h) => <th key={h.key} scope="col">{h.label} %</th>)}
                </tr>
              </thead>
              <tbody>
                {centers.map((_, i) => (
                  <tr key={i}>
                    <td>{edges[i].toFixed(2)}</td>
                    <td>{edges[i + 1].toFixed(2)}</td>
                    {demo.hedgers.map((h) => <td key={h.key}>{share(h.key)[i].toFixed(1)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="learn-multiples">
            {demo.hedgers.map((h) => (
              <figure key={h.key} className="lab-chart">
                <figcaption><span className="lab-chart-title"><Chip color={hedgerColor(theme, h.key)} />{h.label}</span></figcaption>
                <Plot
                  data={[{
                    type: 'bar', x: centers, y: share(h.key), marker: { color: hedgerColor(theme, h.key) },
                    hovertemplate: '%{y:.1f}% of paths<extra></extra>',
                  }]}
                  layout={baseLayout(theme, {
                    height: 150, bargap: 0.1, hovermode: 'closest',
                    xaxis: { range: [edges[0], edges[edges.length - 1]], title: { text: 'Error around its mean' } },
                    yaxis: { title: { text: '% of paths' } },
                  })}
                  config={PLOT_CONFIG}
                  useResizeHandler
                  style={{ width: '100%' }}
                />
              </figure>
            ))}
          </div>
        )}

        <div className="lab-section-head">
          <h3 className="learn-subhead">Inside one path</h3>
          <button type="button" className="lab-link" onClick={() => setPathTable(!pathTable)}>{pathTable ? 'Charts' : 'Table'}</button>
        </div>
        <div className="learn-toggle" role="group" aria-label="Sample path">
          {PATH_NAMES.map((name, i) => (
            <button key={name} type="button" className={i === pathIndex ? 'active' : ''} aria-pressed={i === pathIndex}
              onClick={() => setPathIndex(i)}>{name}</button>
          ))}
        </div>
        {pathTable ? (
          <div className="lab-table-scroll lab-table-tall">
            <table className="lab-data-table">
              <thead>
                <tr>
                  <th scope="col">Day</th>
                  <th scope="col">Spot</th>
                  {PATH_HEDGERS.map((k) => <th key={k} scope="col">{label[k]}: shares</th>)}
                  {OPTION_HEDGERS.map((k) => <th key={k} scope="col">{label[k]}: options held</th>)}
                </tr>
              </thead>
              <tbody>
                {days(horizon).map((d) => (
                  <tr key={d}>
                    <td>{d}</td>
                    <td>{sample.spot[d].toFixed(2)}</td>
                    {PATH_HEDGERS.map((k) => <td key={k}>{sample.positions[k][d].toFixed(3)}</td>)}
                    {OPTION_HEDGERS.map((k) => <td key={k}>{sample.option_value[k][d].toFixed(3)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="lab-multiples">
            <figure className="lab-chart">
              <figcaption><span className="lab-chart-title">Spot price</span></figcaption>
              <Plot
                data={[{ type: 'scatter', mode: 'lines', x: [...Array(horizon + 1).keys()], y: sample.spot,
                  line: { color: c.inkSecondary, width: 2 }, hovertemplate: '%{y:.2f}<extra>Spot</extra>' }]}
                layout={baseLayout(theme, { height: 200, xaxis: pathX, yaxis: { title: { text: 'Spot' } } })}
                config={PLOT_CONFIG}
                useResizeHandler
                style={{ width: '100%' }}
              />
            </figure>
            <figure className="lab-chart">
              <figcaption><span className="lab-chart-title">Shares held</span></figcaption>
              {narrow && <SeriesLegend series={positionSeries} />}
              <Plot
                data={lines(positionSeries)}
                layout={baseLayout(theme, {
                  height: 200, ...labeled(narrow, positionLabels, 120),
                  xaxis: pathX, yaxis: { title: { text: 'Shares' } },
                })}
                config={PLOT_CONFIG}
                useResizeHandler
                style={{ width: '100%' }}
              />
            </figure>
            <figure className="lab-chart">
              <figcaption><span className="lab-chart-title">Options held, at their prices</span></figcaption>
              {narrow && <SeriesLegend series={optionSeries} />}
              <Plot
                data={lines(optionSeries)}
                layout={baseLayout(theme, {
                  height: 200, ...labeled(narrow, optionLabels, 120),
                  xaxis: pathX, yaxis: { title: { text: 'Premium held' } },
                })}
                config={PLOT_CONFIG}
                useResizeHandler
                style={{ width: '100%' }}
              />
            </figure>
          </div>
        )}
        <p className="lab-caption">
          The LSTM shown is the seed whose error spread is the median of its {learned.seeds.length} (seed {learned.representative_seed}).
          Options held is the net premium of the option positions: positive when long.
        </p>

        <h3 className="learn-subhead">Training</h3>
        <figure className="lab-chart">
          <figcaption><span className="lab-chart-title">Hedging error sd on each training batch, by epoch</span></figcaption>
          {narrow && <SeriesLegend series={(['lstm_underlying', 'lstm_options'] as const).map((k) => ({ name: label[k], color: hedgerColor(theme, k) }))} />}
          <Plot
            data={curveTraces}
            layout={baseLayout(theme, {
              height: 230, hovermode: 'closest', ...labeled(narrow, curveLabels, 150),
              xaxis: { title: { text: 'Epoch' }, range: curveLabels.xRange },
              yaxis: { title: { text: 'Error sd (log scale)' }, type: 'log', tickvals: [0.5, 1, 2, 5, 10, 20, 50, 100] },
            })}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: '100%' }}
          />
        </figure>

        <details className="learn-details">
          <summary>A hedger that found a free lunch in an earlier version of this simulator</summary>
          <p>
            The first version of this market re-priced every option slot each day as a fresh option struck at the
            current spot, while a position held overnight was marked to that new option. A held &quot;option&quot; was
            therefore not a tradable contract: worth about the spot times a function of the variance, its next move
            could be forecast from today&apos;s variance (a regression t-statistic near 40). The LSTM, which sees the
            variance, learned to trade on that forecast: its mean P&amp;L was +3.0 a year against 0 for delta hedging,
            and its error spread looked like half of delta&apos;s. Learned hedgers optimize the simulator they are given,
            flaws included. Each slot is now a listed contract kept from issue to expiry, whose price is a martingale:
            no option position earns on average, and today&apos;s variance no longer predicts any contract&apos;s next move.
          </p>
        </details>

        <p className="learn-provenance">
          Run {demo.provenance.run} · commit {demo.provenance.gleipnir_commit.slice(0, 8)} · evaluation seed {setup.eval_seed}
        </p>
      </div>
    </section>
  );
};

export default DeepHedgingSection;
