import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import type { Data } from 'plotly.js';
import './SimulationResults.css';
import { DriftScenario, SimulationResponse } from '../../services/calibrationApi';
import { useTheme } from '../../contexts/ThemeContext';
import { baseLayout, chrome, PLOT_CONFIG, wash } from '../lab/chartTheme';

interface SimulationResultsProps {
  data: SimulationResponse;
}

const DAYS_PER_YEAR = 252;
const SAMPLE_PATHS_SHOWN = 8;
const TABLE_ROWS = 6;

const price = (x: number) => x.toFixed(2);
const pct = (x: number) => `${x < 0 ? '−' : '+'}${Math.abs(x * 100).toFixed(0)}%`;

/** Scenarios come in policy order: positive and negative excess drift, then the historical drift. */
const HISTORICAL = 2;

function legendName(s: DriftScenario, i: number, spotDrift: number | null): string {
  if (i === HISTORICAL) {
    const total = (spotDrift ?? 0) + s.excess_drift;
    return `Historical drift (${(total * 100).toFixed(0)}%/yr), not a forecast`;
  }
  return `${pct(s.excess_drift)}/yr excess drift`;
}

const endLabel = (s: DriftScenario, i: number) => (i === HISTORICAL ? 'historical' : `${pct(s.excess_drift)}/yr`);

/**
 * The simulated spot as a fan: 5-95% and 25-75% bands and the median in the
 * model's one color, a few faint sample paths, and the drift policy's
 * scenarios as dashed neutral lines labelled at their ends.
 */
const SimulationResults: React.FC<SimulationResultsProps> = ({ data }) => {
  const { theme } = useTheme();
  const [showTable, setShowTable] = useState(false);
  const c = chrome(theme);
  const fan = c.world.factual;
  const days = data.time_grid.map((t) => t * DAYS_PER_YEAR);
  const q = data.quantiles;

  const band = (lower: number[], upper: number[], name: string, opacity: string): Data[] => [
    { x: days, y: lower, type: 'scatter', mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false },
    {
      x: days, y: upper, type: 'scatter', mode: 'lines', line: { width: 0 }, fill: 'tonexty',
      fillcolor: `${fan}${opacity}`, name, hoverinfo: 'skip',
    },
  ];

  const traces: Data[] = [
    ...data.sample_paths.slice(0, SAMPLE_PATHS_SHOWN).map((path): Data => ({
      x: days, y: path, type: 'scatter', mode: 'lines', line: { width: 1, color: wash(fan) },
      hoverinfo: 'skip', showlegend: false,
    })),
    ...band(q.p5, q.p95, '5-95% of paths', '22'),
    ...band(q.p25, q.p75, '25-75% of paths', '44'),
    {
      x: days, y: q.p50, type: 'scatter', mode: 'lines', name: 'Median', line: { width: 2, color: fan },
      hovertemplate: 'Day %{x:.0f}: median %{y:.2f}<extra></extra>',
    },
    ...data.scenarios.map((s, i): Data => ({
      x: days, y: s.median, type: 'scatter', mode: 'lines', name: legendName(s, i, data.spot_drift),
      line: { width: 1.5, color: i === 2 ? c.muted : c.inkSecondary, dash: i === 2 ? 'dot' : 'dash' },
      hovertemplate: `${s.label}: %{y:.2f}<extra></extra>`,
    })),
  ];

  const last = days.length - 1;
  const annotations = data.scenarios.map((s, i) => ({
    x: days[last], y: s.median[last], xanchor: 'left' as const, yanchor: 'middle' as const, showarrow: false,
    text: endLabel(s, i),
    font: { size: 11, color: c.inkSecondary },
  }));

  const tableIndex = Array.from({ length: TABLE_ROWS }, (_, i) => Math.round((i * last) / (TABLE_ROWS - 1)));

  return (
    <div className="simulation-results">
      <figure className="simulation-chart">
        <div className="simulation-chart-head">
          <h2>Range of simulated paths</h2>
          <button type="button" className="simulation-table-toggle" onClick={() => setShowTable(!showTable)}>
            {showTable ? 'Chart' : 'Table'}
          </button>
        </div>
        {showTable ? (
          <div className="simulation-table-wrap">
            <table className="simulation-table">
              <thead>
                <tr>
                  <th scope="col">Day</th>
                  <th scope="col">5%</th>
                  <th scope="col">25%</th>
                  <th scope="col">Median</th>
                  <th scope="col">75%</th>
                  <th scope="col">95%</th>
                  {data.scenarios.map((s) => <th scope="col" key={s.label}>{s.label}</th>)}
                </tr>
              </thead>
              <tbody>
                {tableIndex.map((i) => (
                  <tr key={i}>
                    <th scope="row">{days[i].toFixed(0)}</th>
                    <td>{price(q.p5[i])}</td>
                    <td>{price(q.p25[i])}</td>
                    <td>{price(q.p50[i])}</td>
                    <td>{price(q.p75[i])}</td>
                    <td>{price(q.p95[i])}</td>
                    {data.scenarios.map((s) => <td key={s.label}>{price(s.median[i])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Plot
            data={traces}
            layout={baseLayout(theme, {
              height: 380,
              showlegend: true,
              legend: {
                orientation: 'h', x: 0, y: -0.2, yanchor: 'top', traceorder: 'normal',
                font: { size: 11, color: c.inkSecondary },
              },
              margin: { l: 56, r: 72, t: 8, b: 40 },
              annotations,
              xaxis: { title: { text: 'Trading days ahead' } },
              yaxis: { title: { text: 'Spot' } },
            })}
            config={PLOT_CONFIG}
            useResizeHandler
            style={{ width: '100%' }}
          />
        )}
        <figcaption className="simulation-note">
          {data.drift_note} {data.n_paths.toLocaleString()} paths.
        </figcaption>
      </figure>

      <div className="simulation-stats">
        <h3>Spot at the horizon</h3>
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Median</span>
            <span className="stat-value">{price(data.stats.p50)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">5% to 95%</span>
            <span className="stat-value">{price(data.stats.p5)} to {price(data.stats.p95)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Mean</span>
            <span className="stat-value">{price(data.stats.mean)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Std. deviation</span>
            <span className="stat-value">{price(data.stats.std)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Min / Max</span>
            <span className="stat-value">{price(data.stats.min)} / {price(data.stats.max)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimulationResults;
