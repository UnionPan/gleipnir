import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import type { Data } from 'plotly.js';
import type { DeepHedgingDemo, HedgerKey, SeedStat } from '../../services/rlApi';
import { baseLayout, PLOT_CONFIG, ThemeName } from '../lab/chartTheme';
import { hedgerColor } from './learningTheme';
import { endLabels, LabeledSeries } from './endLabels';
import { headline, verdict } from './verdict';
import { useNarrow } from './useNarrow';
import Bars from './Bars';
import Disclosure from './Disclosure';
import YearReplay from './YearReplay';

interface Props {
  demo: DeepHedgingDemo;
  theme: ThemeName;
}

const LEARNED = new Set<HedgerKey>(['lstm_underlying', 'lstm_options']);

const isStat = (v: number | SeedStat): v is SeedStat => typeof v === 'object';
const meanOf = (v: number | SeedStat) => (isStat(v) ? v.mean : v);
const sdOf = (v: number | SeedStat) => (isStat(v) ? v.sd : undefined);

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

const withArticle = (noun: string) => `${/^[aeiou]/i.test(noun) ? 'an' : 'a'} ${noun}`;

const Distributions: React.FC<Props> = ({ demo, theme }) => {
  const [asTable, setAsTable] = useState(false);
  const edges = demo.histogram.edges;
  const centers = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
  const total = demo.histogram.bs_delta.counts.reduce((a, b) => a + b, 0);
  const share = (key: HedgerKey) => demo.histogram[key].counts.map((n) => (100 * n) / total);
  return (
    <>
      <div className="lab-section-head">
        <p className="lab-hero-note">Each hedger's error on every evaluation path, around its own average.</p>
        <button type="button" className="lab-link" onClick={() => setAsTable(!asTable)}>{asTable ? 'Charts' : 'Table'}</button>
      </div>
      {asTable ? (
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
    </>
  );
};

const Training: React.FC<Props> = ({ demo, theme }) => {
  const narrow = useNarrow();
  const label = Object.fromEntries(demo.hedgers.map((h) => [h.key, h.label])) as Record<HedgerKey, string>;
  const traces: Data[] = (['lstm_underlying', 'lstm_options'] as const).flatMap((k) =>
    Object.entries(demo.training_curves[k] ?? {}).map(([seed, curve]) => ({
      type: 'scatter', mode: 'lines', name: `${label[k]}, seed ${seed}`, x: curve.epoch, y: curve.std_error,
      line: { color: hedgerColor(theme, k), width: 1.5 },
      hovertemplate: '%{y:.2f}<extra>' + label[k] + ', seed ' + seed + '</extra>',
    })));
  // one label per method, at the end of its median seed's curve
  const labels = endLabels((['lstm_underlying', 'lstm_options'] as const).map((k) => {
    const seeds = Object.values(demo.training_curves[k] ?? {});
    const ends = seeds.map((s) => s.std_error[s.std_error.length - 1]).sort((a, b) => a - b);
    const all = seeds.flatMap((s) => s.std_error);
    return { name: label[k], x: seeds[0]?.epoch ?? [0], y: [...all, ends[Math.floor(ends.length / 2)]],
      color: hedgerColor(theme, k) };
  }), { logY: true });
  return (
    <figure className="lab-chart">
      <figcaption><span className="lab-chart-title">Hedging error on each training batch, by epoch</span></figcaption>
      {narrow && <SeriesLegend series={(['lstm_underlying', 'lstm_options'] as const).map((k) => ({ name: label[k], color: hedgerColor(theme, k) }))} />}
      <Plot
        data={traces}
        layout={baseLayout(theme, {
          height: 230, hovermode: 'closest', ...labeled(narrow, labels, 150),
          xaxis: { title: { text: 'Epoch' }, range: labels.xRange },
          yaxis: { title: { text: 'Error sd (log scale)' }, type: 'log', tickvals: [0.5, 1, 2, 5, 10, 20, 50, 100] },
        })}
        config={PLOT_CONFIG}
        useResizeHandler
        style={{ width: '100%' }}
      />
      <p className="lab-hero-note">
        {demo.setup.training.seeds.length} seeds per hedger, {demo.setup.training.n_epochs} epochs of{' '}
        {demo.setup.training.batch_size.toLocaleString('en-US')} simulated years each.
      </p>
    </figure>
  );
};

const DeepHedgingSection: React.FC<Props> = ({ demo, theme }) => {
  const { setup, results } = demo;
  const label = Object.fromEntries(demo.hedgers.map((h) => [h.key, h.label])) as Record<HedgerKey, string>;
  const fairValues = demo.hedgers.map((h) => -meanOf(results[h.key].mean_error) - meanOf(results[h.key].mean_cost));
  const reasons = verdict(demo, label);
  const bars = (pick: (k: HedgerKey) => number | SeedStat) => demo.hedgers.map((h) => ({
    key: h.key, label: h.label, value: meanOf(pick(h.key)), sd: sdOf(pick(h.key)), color: hedgerColor(theme, h.key),
  }));

  return (
    <div className="learn-tab">
      <div className="lab-verdict">
        <p className="lab-headline">{headline(demo, label)}</p>
        {reasons.length > 0 && (
          <details className="lab-why">
            <summary>Why?</summary>
            {reasons.map((r) => <p key={r}>{r}</p>)}
          </details>
        )}
      </div>

      <section className="lab-panel" aria-label="One simulated year">
        <div className="lab-titlebar">
          One simulated year · short {setup.liability.replace('short ', '')}
        </div>
        <div className="lab-panel-body">
          <YearReplay demo={demo} theme={theme} />
        </div>
      </section>

      <section className="lab-panel" aria-label="Results">
        <div className="lab-titlebar">Over {setup.eval_paths.toLocaleString('en-US')} simulated years</div>
        <div className="lab-panel-body learn-bars-row">
          <Bars title="Spread of outcomes (lower is better)" theme={theme} rows={bars((k) => results[k].std_error)}
            note="Standard deviation of the P&L at expiry. Whiskers: spread across training seeds." />
          <Bars title="Spent on trading" theme={theme} rows={bars((k) => results[k].mean_cost)} digits={2}
            note="Average trading cost over the year, in the units of the option (spot = 100)." />
        </div>
      </section>

      <Disclosure title="The setup" hint="market, instruments, costs, objective">
        <p className="lab-caption">
          A desk sells {withArticle(setup.liability.replace('short ', ''))} and hedges it daily in a Heston market
          (κ {setup.market.kappa}, θ {setup.market.theta}, ξ {setup.market.sigma_v}, ρ {setup.market.rho}), with no rates or
          dividends. It may trade the stock (cost {(1e4 * setup.costs.underlying).toFixed(0)} bp) and {setup.options}{' '}
          (cost {(100 * setup.costs.options).toFixed(0)}% of premium). The LSTM learns its trades by minimizing{' '}
          {setup.objective}, differentiating through the simulated market; it sees the price, the variance, the
          options&apos; prices, strikes and tenors, and its holdings. Before costs every hedger&apos;s average outcome is the
          same ({Math.min(...fairValues).toFixed(2)} to {Math.max(...fairValues).toFixed(2)}): no strategy earns money on
          average here, so the differences are risk and cost.
        </p>
      </Disclosure>
      <Disclosure title="All numbers" hint="every hedger, mean ± sd across seeds">
        <div className="lab-table-scroll">
          <table className="lab-results learn-results">
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
          Worst 5%: the mean shortfall of the worst twentieth of paths, from each hedger&apos;s own average. Options traded:
          the premium bought and sold over the year.
        </p>
      </Disclosure>
      <Disclosure title="How the errors are spread" hint="a histogram per hedger">
        <Distributions demo={demo} theme={theme} />
      </Disclosure>
      <Disclosure title="Training" hint="error by epoch, every seed">
        <Training demo={demo} theme={theme} />
      </Disclosure>
      <Disclosure title="The free lunch an earlier simulator offered">
        <p className="lab-caption">
          The first version of this market re-priced every option slot each day as a fresh option struck at the current
          price, while a position held overnight was marked to that new option. A held &quot;option&quot; was not a
          tradable contract: its next move could be forecast from today&apos;s variance (a t-statistic near 40). The LSTM,
          which sees the variance, learned to trade on that forecast and looked twice as good as delta hedging. Learned
          hedgers optimize the simulator they are given, flaws included. Each slot is now a listed contract kept from issue
          to expiry, whose price is a martingale.
        </p>
      </Disclosure>
      <p className="learn-provenance">
        Run {demo.provenance.run} · commit {demo.provenance.gleipnir_commit.slice(0, 8)} · evaluation seed {setup.eval_seed}
      </p>
    </div>
  );
};

export default DeepHedgingSection;
