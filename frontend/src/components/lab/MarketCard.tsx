import React from 'react';
import type { AssetInfo, HedgeResponse, LabCatalog } from '../../services/labApi';

const short = (hash: string | null | undefined) => (hash ? hash.slice(0, 8) : '—');
const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;

function halfLife(kappa: number): string {
  const days = (Math.LN2 / kappa) * 252;
  return days < 60 ? `${days.toFixed(0)} trading days` : `${(days / 21).toFixed(1)} months`;
}

interface Row {
  key: string;
  label: string;
  hint: string;
  format: (p: Record<string, number>) => string;
}

const ROWS: Row[] = [
  { key: 'v0', label: 'Volatility today', hint: 'Trailing 21-day realized vol: the starting state', format: (p) => pct(Math.sqrt(p.v0)) },
  { key: 'theta', label: 'Long-run volatility', hint: 'Where volatility reverts to', format: (p) => pct(Math.sqrt(p.theta)) },
  { key: 'kappa', label: 'Mean-reversion half-life', hint: 'How fast a vol shock fades', format: (p) => halfLife(p.kappa) },
  { key: 'sigma_v', label: 'Vol of variance ξ', hint: 'How violently volatility itself moves', format: (p) => p.sigma_v.toFixed(2) },
  { key: 'rho', label: 'Spot-vol correlation ρ', hint: 'Negative means vol rises when the price falls', format: (p) => p.rho.toFixed(2) },
  { key: 'mu', label: 'Spot drift', hint: 'r − q: no excess drift for books up to 3 months', format: (p) => pct(p.mu) },
];

const signedPct = (x: number) => `${x < 0 ? '−' : ''}${pct(Math.abs(x))}`;

interface Props {
  catalog: LabCatalog;
  asset: AssetInfo;
  result: HedgeResponse | null;
}

const MarketCard: React.FC<Props> = ({ catalog, asset, result }) => {
  const factual = result?.factual.params ?? asset.params;
  const shocked = result?.counterfactual?.params;
  return (
    <section className="lab-panel" aria-label="Market and book">
      <div className="lab-titlebar">Market · {asset.ticker}</div>
      <div className="lab-panel-body">
        <p className="lab-caption">
          Heston dynamics fitted to {asset.ticker}'s daily closes over {catalog.window.replace('..', ' to ')} by
          quasi-maximum likelihood. Prices are normalized so today's spot is 100.
        </p>
        <table className="lab-params">
          <thead>
            <tr>
              <th scope="col">Parameter</th>
              <th scope="col"><span className="lab-key lab-key-factual" aria-hidden />Factual</th>
              {shocked && <th scope="col"><span className="lab-key lab-key-counterfactual" aria-hidden />{result?.counterfactual?.label}</th>}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => {
              const before = row.format(factual);
              const after = shocked ? row.format(shocked) : null;
              return (
                <tr key={row.key}>
                  <th scope="row">
                    {row.label}
                    <small>{row.hint}</small>
                  </th>
                  <td>{before}</td>
                  {after !== null && <td className={after !== before ? 'lab-changed' : ''}>{after}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
        {result && (
          <dl className="lab-book">
            <div><dt>Premium received</dt><dd>{result.book.premium.toFixed(2)}</dd></div>
            <div><dt>Implied vol</dt><dd>{pct(result.book.implied_vol)}</dd></div>
            <div><dt>Strike</dt><dd>{result.book.strike.toFixed(0)}</dd></div>
            <div><dt>Expiry</dt><dd>{result.book.maturity_days} trading days</dd></div>
          </dl>
        )}
        <p className="lab-note">
          Historical drift {signedPct(asset.historical_drift.estimate)} a year ± {pct(asset.historical_drift.std_error)}{' '}
          (95% interval {signedPct(asset.historical_drift.ci_low)} to {signedPct(asset.historical_drift.ci_high)}).
          Too uncertain to forecast with, so the lab never simulates it.
        </p>
        <p className="lab-note">
          ρ is not identified by daily closes: the estimator returns about 0 for every stock. Equity option
          markets typically imply about {catalog.rho_assumption.toFixed(1)}, which you can assume instead.
        </p>
        <p className="lab-provenance">
          calibration <code>{short(asset.params_hash)}</code>
          {result && result.book.parent_hash && <> → book <code>{short(result.book.params_hash)}</code></>}
          {result?.counterfactual && <> → world <code>{short(result.counterfactual.params_hash)}</code></>}
        </p>
      </div>
    </section>
  );
};

export default MarketCard;
