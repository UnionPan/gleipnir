import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import type { Data } from 'plotly.js';
import type { HedgeResponse, HedgerName, WorldResult } from '../../services/labApi';
import { HEDGER_LABELS } from './insights';
import { baseLayout, chrome, PLOT_CONFIG, ThemeName, World } from './chartTheme';
import { WorldLegend } from './DistributionPanels';

interface Props {
  result: HedgeResponse;
  theme: ThemeName;
}

type Series = (w: WorldResult) => number[];

const HEDGERS: HedgerName[] = ['no_hedge', 'bs_delta', 'heston_delta'];

const PANELS: { key: string; title: string; yTitle: string; series: Series; days: (n: number) => number[] }[] = [
  { key: 'spot', title: 'Spot price', yTitle: 'Spot', series: (w) => w.sample_path.spots, days: (n) => [...Array(n + 1).keys()] },
  ...(['bs_delta', 'heston_delta'] as HedgerName[]).map((h) => ({
    key: h,
    title: `${HEDGER_LABELS[h]} position`,
    yTitle: 'Shares held',
    series: (w: WorldResult) => w.sample_path.positions[h],
    days: (n: number) => [...Array(n).keys()],
  })),
];

const SamplePathPanels: React.FC<Props> = ({ result, theme }) => {
  const [asTable, setAsTable] = useState(false);
  const c = chrome(theme);
  const n = result.book.maturity_days;
  const worlds: [World, WorldResult][] = [
    ['factual', result.factual],
    ...(result.counterfactual ? [['counterfactual', result.counterfactual] as [World, WorldResult]] : []),
  ];

  return (
    <section className="lab-panel" aria-label="One sample path">
      <div className="lab-titlebar">Inside one path</div>
      <div className="lab-panel-body">
        <div className="lab-section-head">
          <p className="lab-caption">
            The median path of the factual world{result.counterfactual ? ', and the same random draw run through the counterfactual world' : ''}.
            The hedgers rebalance once per trading day.
          </p>
          <button type="button" className="lab-link" onClick={() => setAsTable(!asTable)}>{asTable ? 'Charts' : 'Table'}</button>
        </div>
        <WorldLegend result={result} />
        {asTable ? (
          <div className="lab-table-scroll lab-table-tall">
            <table className="lab-data-table">
              <thead>
                <tr>
                  <th scope="col">Day</th>
                  {worlds.map(([, w]) => PANELS.map((p) => <th key={`${w.label}-${p.key}`} scope="col">{w.label}: {p.title}</th>))}
                </tr>
              </thead>
              <tbody>
                {[...Array(n + 1).keys()].map((d) => (
                  <tr key={d}>
                    <td>{d}</td>
                    {worlds.map(([, w]) => PANELS.map((p) => {
                      const v = p.series(w)[d];
                      return <td key={`${w.label}-${p.key}`}>{v === undefined ? '—' : v.toFixed(p.key === 'spot' ? 2 : 3)}</td>;
                    }))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="lab-multiples">
            {PANELS.map((p) => {
              const traces: Data[] = worlds.map(([key, w]) => ({
                type: 'scatter',
                mode: 'lines',
                name: w.label,
                x: p.days(n),
                y: p.series(w),
                line: { color: c.world[key], width: 2 },
                hovertemplate: (p.key === 'spot' ? '%{y:.2f}' : '%{y:.3f}') + '<extra>' + w.label + '</extra>',
              }));
              return (
                <figure key={p.key} className="lab-chart">
                  <figcaption><span className="lab-chart-title">{p.title}</span></figcaption>
                  <Plot
                    data={traces}
                    layout={baseLayout(theme, {
                      height: 190,
                      xaxis: { title: { text: 'Trading day' } },
                      yaxis: { title: { text: p.yTitle } },
                    })}
                    config={PLOT_CONFIG}
                    useResizeHandler
                    style={{ width: '100%' }}
                  />
                </figure>
              );
            })}
          </div>
        )}
        <table className="lab-results lab-path-pnl">
          <caption>P&L at expiry on this path</caption>
          <thead>
            <tr>
              <th scope="col">World</th>
              {HEDGERS.map((h) => <th key={h} scope="col">{HEDGER_LABELS[h]}</th>)}
            </tr>
          </thead>
          <tbody>
            {worlds.map(([key, w]) => (
              <tr key={key}>
                <td className="lab-world-cell"><span className={`lab-key lab-key-${key}`} aria-hidden />{w.label}</td>
                {HEDGERS.map((h) => <td key={h}>{w.sample_path.pnl[h].toFixed(2)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};

export default SamplePathPanels;
