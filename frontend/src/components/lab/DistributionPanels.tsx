import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import type { Data } from 'plotly.js';
import type { HedgeResponse, HedgerInfo, HedgerName, WorldResult } from '../../services/labApi';
import { baseLayout, chrome, PLOT_CONFIG, ThemeName, wash, World } from './chartTheme';

interface Props {
  result: HedgeResponse;
  theme: ThemeName;
}

const share = (counts: number[], n: number) => counts.map((c) => (100 * c) / n);
function tailNote(ws: WorldResult[], hedger: HedgerName, n: number): string {
  const outside = Math.max(...ws.map((w) => w.hedgers[hedger].below + w.hedgers[hedger].above));
  return `${((100 * outside) / n).toFixed(1)}% of paths fall outside the plotted range.`;
}

const centers = (edges: number[]) => edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);

function worlds(result: HedgeResponse): [World, WorldResult][] {
  return [['factual', result.factual], ...(result.counterfactual ? [['counterfactual', result.counterfactual] as [World, WorldResult]] : [])];
}

const Panel: React.FC<{ info: HedgerInfo; result: HedgeResponse; theme: ThemeName }> = ({ info, result, theme }) => {
  const hedger: HedgerName = info.name;
  const [asTable, setAsTable] = useState(false);
  const c = chrome(theme);
  const edges = result.bin_edges[hedger];
  const x = centers(edges);
  const ws = worlds(result);

  const traces: Data[] = ws.map(([key, w]) => ({
    type: 'scatter',
    mode: 'lines',
    name: w.label,
    x,
    y: share(w.hedgers[hedger].histogram, result.n_paths),
    line: { color: c.world[key], width: 2, shape: 'hvh' },
    fill: 'tozeroy',
    fillcolor: wash(c.world[key]),
    hovertemplate: '%{y:.1f}% of paths<extra>' + w.label + '</extra>',
  }));

  return (
    <figure className="lab-chart">
      <figcaption>
        <span className="lab-chart-title">{info.label}</span>
        <button type="button" className="lab-link" onClick={() => setAsTable(!asTable)}>
          {asTable ? 'Chart' : 'Table'}
        </button>
      </figcaption>
      {asTable ? (
        <div className="lab-table-scroll lab-table-tall">
          <table className="lab-data-table">
            <thead>
              <tr>
                <th scope="col">P&L from</th>
                <th scope="col">to</th>
                {ws.map(([, w]) => <th key={w.label} scope="col">{w.label} %</th>)}
              </tr>
            </thead>
            <tbody>
              {x.map((_, i) => (
                <tr key={i}>
                  <td>{edges[i].toFixed(2)}</td>
                  <td>{edges[i + 1].toFixed(2)}</td>
                  {ws.map(([, w]) => (
                    <td key={w.label}>{share(w.hedgers[hedger].histogram, result.n_paths)[i].toFixed(1)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Plot
          data={traces}
          layout={baseLayout(theme, {
            height: 200,
            xaxis: { title: { text: 'P&L at expiry' }, zeroline: true },
            yaxis: { title: { text: '% of paths' }, rangemode: 'tozero' },
          })}
          config={PLOT_CONFIG}
          useResizeHandler
          style={{ width: '100%' }}
        />
      )}
      <p className="lab-chart-note">
        {tailNote(ws.map(([, w]) => w), hedger, result.n_paths)}
      </p>
      {info.role === 'none' && (
        <p className="lab-chart-note">
          The spike is every path where the option expires worthless: the book simply keeps the premium.
        </p>
      )}
    </figure>
  );
};

const DistributionPanels: React.FC<Props> = ({ result, theme }) => (
  <section className="lab-panel" aria-label="P&L distributions">
    <div className="lab-titlebar">How the outcomes spread</div>
    <div className="lab-panel-body">
      <WorldLegend result={result} />
      <div className="lab-multiples">
        {result.hedgers.map((h) => <Panel key={h.name} info={h} result={result} theme={theme} />)}
      </div>
      <p className="lab-caption">
        Each panel has its own P&L scale: unhedged outcomes spread about ten times wider than hedged ones.
        A narrower curve means a better hedge; a curve shifted left means the book loses money in that world.
      </p>
    </div>
  </section>
);

export const WorldLegend: React.FC<{ result: HedgeResponse }> = ({ result }) => (
  <ul className="lab-legend" aria-label="Legend">
    <li><span className="lab-key lab-key-factual" aria-hidden />{result.factual.label}: {result.factual.description}</li>
    {result.counterfactual && (
      <li><span className="lab-key lab-key-counterfactual" aria-hidden />{result.counterfactual.label}: {result.counterfactual.description}</li>
    )}
  </ul>
);

export default DistributionPanels;
