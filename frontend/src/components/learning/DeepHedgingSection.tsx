import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { Data } from 'plotly.js';
import type { Bands, DeepHedgingDemo, HedgerKey, SeedStat } from '../../services/rlApi';
import { baseLayout, PLOT_CONFIG, ThemeName } from '../lab/chartTheme';
import { hedgerColor } from './learningTheme';
import { endLabels, LabeledSeries } from './endLabels';
import { headline, verdict } from './verdict';
import { useNarrow } from './useNarrow';
import Bars from './Bars';
import Disclosure from '../common/Disclosure';
import YearReplay from './YearReplay';
import Ridges from './Ridges';
import BandMultiples from './BandMultiples';
import { premiumOf } from './verdict';

interface Props {
  demo: DeepHedgingDemo;
  theme: ThemeName;
}

const shiftBands = (b: Bands, by: number): Bands => ({
  p5: b.p5.map((v) => v + by), p25: b.p25.map((v) => v + by), p50: b.p50.map((v) => v + by),
  p75: b.p75.map((v) => v + by), p95: b.p95.map((v) => v + by),
});

const isStat = (v: number | SeedStat): v is SeedStat => typeof v === 'object';
const meanOf = (v: number | SeedStat) => (isStat(v) ? v.mean : v);
const sdOf = (v: number | SeedStat) => (isStat(v) ? v.sd : undefined);

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
  const premium = premiumOf(demo);
  // the export's books start with no cash: credit the premium so every drawn book starts at 0
  const wealth = useMemo(() => (demo.wealth_bands ? Object.fromEntries(Object.entries(demo.wealth_bands)
    .map(([k, b]) => [k, shiftBands(b, premium)])) as Record<HedgerKey, Bands> : null), [demo, premium]);
  const spread = Object.fromEntries(demo.hedgers.map((h) => [h.key, meanOf(results[h.key].std_error)])) as Record<HedgerKey, number>;
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
        <div className="lab-panel-body">
          {demo.terminal_histogram && (
            <Ridges title="P&L at expiry, every simulated year" theme={theme}
              edges={demo.terminal_histogram.edges.map((e) => e + premium)}
              axisLabel="P&L at expiry (the option's units, spot = 100)"
              rows={demo.hedgers.map((h) => ({
                key: h.key, label: h.label, color: hedgerColor(theme, h.key),
                counts: (demo.terminal_histogram as NonNullable<DeepHedgingDemo['terminal_histogram']>)[h.key].counts,
                end: `±${spread[h.key].toFixed(2)}`,
              }))}
              note={<>Each ridge is how one hedger&apos;s {setup.eval_paths.toLocaleString('en-US')} years end: the
                premium of {premium.toFixed(2)}, plus the hedge&apos;s gains, less its costs and the payoff. A narrow
                ridge is a good hedge; ± is its spread (standard deviation).</>} />
          )}
          {wealth && (
            <figure className="lab-chart">
              <figcaption><span className="lab-chart-title">What the book is worth through the year</span></figcaption>
              <BandMultiples theme={theme} animationKey={demo}
                rows={demo.hedgers.filter((h) => wealth[h.key]).map((h) => ({
                  key: h.key, label: h.label, color: hedgerColor(theme, h.key), bands: wealth[h.key],
                }))}
                note={<>What the book is worth each day: the hedge account (the premium of {premium.toFixed(2)}, plus
                  gains, less costs) less the true value of the call still owed. It starts at 0 and ends at the
                  P&amp;L. The bands hold the middle half and the middle 90% of{' '}
                  {setup.eval_paths.toLocaleString('en-US')} years; one scale for every hedger, so a wider band is a
                  worse hedge.</>} />
            </figure>
          )}
        </div>
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
