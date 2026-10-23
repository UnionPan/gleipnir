import React from 'react';
import type { HedgeResponse, WorldResult } from '../../services/labApi';
import { HEDGER_LABELS, HEDGER_NOTES, HEDGER_ORDER, Insight } from './insights';

interface Props {
  result: HedgeResponse;
  insight: Insight;
}

const METRICS = [
  { key: 'mean', label: 'Mean P&L' },
  { key: 'std', label: 'Hedging error' },
  { key: 'cvar_5', label: 'Worst 5%' },
  { key: 'mean_cost', label: 'Trading cost' },
] as const;

const ResultsTable: React.FC<Props> = ({ result, insight }) => {
  const worlds: { key: string; world: WorldResult }[] = [
    { key: 'factual', world: result.factual },
    ...(result.counterfactual ? [{ key: 'counterfactual', world: result.counterfactual }] : []),
  ];
  const showWorld = worlds.length > 1;
  return (
    <section className="lab-panel" aria-label="Results">
      <div className="lab-titlebar">Results</div>
      <div className="lab-panel-body">
        <p className="lab-headline">{insight.headline}</p>
        <p className="lab-detail">{insight.detail}</p>
        <div className="lab-table-scroll">
          <table className="lab-results">
            <thead>
              <tr>
                <th scope="col">Hedger</th>
                {showWorld && <th scope="col" className="lab-world-col">World</th>}
                {METRICS.map((m) => <th key={m.key} scope="col">{m.label}</th>)}
              </tr>
            </thead>
            {HEDGER_ORDER.map((h) => (
              <tbody key={h} className="lab-results-group">
                {worlds.map(({ key, world }, i) => (
                  <tr key={key}>
                    {i === 0 && (
                      <th scope="rowgroup" rowSpan={worlds.length}>
                        {HEDGER_LABELS[h]}
                        <small>{HEDGER_NOTES[h]}</small>
                      </th>
                    )}
                    {showWorld && (
                      <td className="lab-world-cell">
                        <span className={`lab-key lab-key-${key}`} aria-hidden />
                        {world.label}
                      </td>
                    )}
                    {METRICS.map((m) => <td key={m.key}>{world.hedgers[h][m.key].toFixed(2)}</td>)}
                  </tr>
                ))}
              </tbody>
            ))}
          </table>
        </div>
        <p className="lab-caption">
          P&L of one option sold, in the same units as the spot (100). Hedging error is the standard deviation
          across {result.n_paths.toLocaleString()} paths; worst 5% is the average over the worst 5% of paths.
          Every hedger sees the same paths.
        </p>
      </div>
    </section>
  );
};

export default ResultsTable;
