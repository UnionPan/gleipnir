import React from 'react';
import type { HedgeResponse, LabCatalog, LabModelName } from '../../services/labApi';

const short = (hash: string | null | undefined) => (hash ? hash.slice(0, 8) : '—');
const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
const signedPct = (x: number) => `${x < 0 ? '−' : ''}${pct(Math.abs(x))}`;
const DAYS_PER_YEAR = 252;

const days = (n: number) => `${n.toFixed(0)} trading day${Math.round(n) === 1 ? '' : 's'}`;

function halfLife(n: number): string {
  return n < 60 ? days(n) : `${(n / 21).toFixed(1)} months`;
}

interface Row {
  key: string;
  label: string;
  hint: string;
  format: (p: Record<string, number>) => string;
}

/** Rows in each world's own parameter names (mjollnir), read for a person. */
function rows(model: LabModelName, params: Record<string, number>): Row[] {
  switch (model) {
    case 'heston':
      return [
        { key: 'v0', label: 'Volatility today', hint: 'Trailing 21-day realized vol: the starting state', format: (p) => pct(Math.sqrt(p.v0)) },
        { key: 'theta', label: 'Long-run volatility', hint: 'Where volatility reverts to', format: (p) => pct(Math.sqrt(p.theta)) },
        { key: 'kappa', label: 'Mean-reversion half-life', hint: 'How fast a vol shock fades', format: (p) => halfLife((Math.LN2 / p.kappa) * DAYS_PER_YEAR) },
        { key: 'sigma_v', label: 'Vol of variance ξ', hint: 'How violently volatility itself moves', format: (p) => p.sigma_v.toFixed(2) },
        { key: 'rho', label: 'Spot-vol correlation ρ', hint: 'Negative means vol rises when the price falls', format: (p) => p.rho.toFixed(2) },
      ];
    case 'gbm':
      return [{ key: 'sigma', label: 'Volatility', hint: 'Constant: the historical volatility of the window', format: (p) => pct(p.sigma) }];
    case 'merton_jump':
      return [
        { key: 'sigma', label: 'Diffusion volatility', hint: 'Between jumps', format: (p) => pct(p.sigma) },
        { key: 'lambda_j', label: 'Jumps a year', hint: 'Average number of jumps', format: (p) => p.lambda_j.toFixed(0) },
        { key: 'mu_j', label: 'Average jump', hint: 'Mean log-size of a jump', format: (p) => signedPct(p.mu_j) },
        { key: 'sigma_j', label: 'Jump size spread', hint: 'Standard deviation of a jump', format: (p) => pct(p.sigma_j) },
      ];
    case 'regime_switching_gbm': {
      const n = Object.keys(params).filter((k) => /^sigma_\d+$/.test(k)).length;
      return [
        ...Array.from({ length: n }, (_, k): Row => ({
          key: `sigma_${k}`,
          label: `Regime ${k} volatility`,
          hint: `Lasts about ${days(1 / Math.max(1 - params[`p_${k}${k}`], 1e-9))} at a time`,
          format: (p) => pct(p[`sigma_${k}`]),
        })),
        { key: 'regime0', label: 'Regime today', hint: 'Where the window ends', format: (p) => `${p.regime0.toFixed(0)}` },
      ];
    }
    case 'garch':
      return [
        { key: 'h0', label: 'Volatility today', hint: 'The GARCH filter at the end of the window', format: (p) => pct(Math.sqrt(p.h0 * DAYS_PER_YEAR)) },
        { key: 'omega', label: 'Long-run volatility', hint: 'sqrt(ω / (1 − α − β)), annualized', format: (p) => pct(Math.sqrt((p.omega / (1 - p.alpha - p.beta)) * DAYS_PER_YEAR)) },
        { key: 'alpha', label: 'Shock reaction α', hint: 'How much a big day raises tomorrow\'s variance', format: (p) => p.alpha.toFixed(3) },
        { key: 'beta', label: 'Shock half-life', hint: 'Persistence α + β', format: (p) => halfLife(Math.log(0.5) / Math.log(p.alpha + p.beta)) },
        ...(params.nu === undefined ? [] : [
          { key: 'nu', label: 'Tail index ν', hint: 'Student-t shocks: lower is fatter-tailed', format: (p: Record<string, number>) => p.nu.toFixed(1) },
        ]),
      ];
    case 'rough_bergomi':
      return [
        { key: 'xi0', label: 'Volatility today', hint: 'Trailing 21-day realized vol: the forward variance', format: (p) => pct(Math.sqrt(p.xi0)) },
        { key: 'eta', label: 'Vol of variance η', hint: 'Scale of log-variance moves', format: (p) => p.eta.toFixed(2) },
        { key: 'H', label: 'Roughness H', hint: 'Below 0.5 is rough; daily closes barely identify it', format: (p) => p.H.toFixed(2) },
        { key: 'rho', label: 'Spot-vol correlation ρ', hint: 'Negative means vol rises when the price falls', format: (p) => p.rho.toFixed(2) },
      ];
  }
}

interface Props {
  catalog: LabCatalog;
  result: HedgeResponse;
}

/** The book that ran, described from the result alone (shipped or visitor calibration). */
const MarketCard: React.FC<Props> = ({ catalog, result }) => {
  const { book } = result;
  const { origin } = book;
  const info = catalog.models.find((m) => m.name === book.model);
  const factual = result.factual.params;
  const shocked = result.counterfactual?.params;
  const drift = book.rate - book.dividend_yield;
  const whose = origin.kind === 'visitor' ? 'your calibration of' : '';
  return (
    <section className="lab-panel" aria-label="Market and book">
      <div className="lab-titlebar">
        Market · {book.asset}
        {origin.kind === 'visitor' && <span className="lab-origin-badge">Your calibration</span>}
      </div>
      <div className="lab-panel-body">
        <p className="lab-caption">
          {info?.label}, {whose} {book.asset}'s daily closes over {origin.window.replace('..', ' to ')}.
          Prices are normalized so today's spot is 100.
        </p>
        <table className="lab-params">
          <thead>
            <tr>
              <th scope="col">Parameter</th>
              <th scope="col"><span className="lab-key lab-key-factual" aria-hidden />Factual</th>
              {shocked && <th scope="col"><span className="lab-key lab-key-counterfactual" aria-hidden />{result.counterfactual?.label}</th>}
            </tr>
          </thead>
          <tbody>
            {rows(book.model, factual).map((row) => {
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
            <tr>
              <th scope="row">
                Spot drift r − q
                <small>No excess drift for books up to 3 months</small>
              </th>
              <td>{pct(drift)}</td>
              {shocked && <td>{pct(drift)}</td>}
            </tr>
          </tbody>
        </table>
        {origin.note && (
          <details className="lab-fit-note">
            <summary>About this fit</summary>
            <p>{origin.method_label}. {origin.note}</p>
          </details>
        )}
        <dl className="lab-book">
          <div><dt>Premium received</dt><dd>{book.premium.toFixed(2)}</dd></div>
          <div><dt>Implied vol</dt><dd>{pct(book.implied_vol)}</dd></div>
          <div><dt>Strike</dt><dd>{book.strike.toFixed(0)}</dd></div>
          <div><dt>Expiry</dt><dd>{book.maturity_days} trading days</dd></div>
        </dl>
        <p className="lab-note">
          Rate r {pct(book.rate, 2)}
          {origin.kind === 'shipped' ? ` (${catalog.rate_source})` : ' (13-week Treasury bill on the day you calibrated)'};
          dividend yield q {pct(book.dividend_yield, 2)}. Cash earns r and the shares held earn q.
        </p>
        {origin.historical_drift && (
          <p className="lab-note">
            Historical drift {signedPct(origin.historical_drift.estimate)} a year ± {pct(origin.historical_drift.std_error)}{' '}
            (95% interval {signedPct(origin.historical_drift.ci_low)} to {signedPct(origin.historical_drift.ci_high)}).
            Too uncertain to forecast with, so the lab never simulates it.
          </p>
        )}
        {info?.has_rho && (
          <p className="lab-note">
            ρ is not identified by daily closes: the estimators return about 0 for every stock. Equity option
            markets typically imply about {catalog.rho_assumption.toFixed(1)}, which you can assume instead.
          </p>
        )}
        <p className="lab-provenance">
          calibration <code>{short(origin.calibration_hash)}</code>
          {book.parent_hash && <> → book <code>{short(book.params_hash)}</code></>}
          {result.counterfactual && <> → world <code>{short(result.counterfactual.params_hash)}</code></>}
        </p>
      </div>
    </section>
  );
};

export default MarketCard;
