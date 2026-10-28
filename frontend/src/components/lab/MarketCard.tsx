import React from 'react';
import type { HedgeResponse, LabCatalog, LabModelName } from '../../services/labApi';
import { chrome, ThemeName } from './chartTheme';
import { linear, useWidth } from './viz';

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
  /** The number the row stands for, drawn on the row's own scale. */
  value: (p: Record<string, number>) => number;
  show: (v: number) => string;
}

const vol = (v: number) => pct(v);
const plain = (digits: number) => (v: number) => v.toFixed(digits);

/** Rows in each world's own parameter names (mjollnir), read for a person. */
function rows(model: LabModelName, params: Record<string, number>): Row[] {
  switch (model) {
    case 'heston':
      return [
        { key: 'v0', label: 'Volatility today', hint: 'Trailing 21-day realized vol: the starting state', value: (p) => Math.sqrt(p.v0), show: vol },
        { key: 'theta', label: 'Long-run volatility', hint: 'Where volatility reverts to', value: (p) => Math.sqrt(p.theta), show: vol },
        { key: 'kappa', label: 'Mean-reversion half-life', hint: 'How fast a vol shock fades', value: (p) => (Math.LN2 / p.kappa) * DAYS_PER_YEAR, show: halfLife },
        { key: 'sigma_v', label: 'Vol of variance ξ', hint: 'How violently volatility itself moves', value: (p) => p.sigma_v, show: plain(2) },
        { key: 'rho', label: 'Spot-vol correlation ρ', hint: 'Negative means vol rises when the price falls', value: (p) => p.rho, show: plain(2) },
      ];
    case 'gbm':
      return [{ key: 'sigma', label: 'Volatility', hint: 'Constant: the historical volatility of the window', value: (p) => p.sigma, show: vol }];
    case 'merton_jump':
      return [
        { key: 'sigma', label: 'Diffusion volatility', hint: 'Between jumps', value: (p) => p.sigma, show: vol },
        { key: 'lambda_j', label: 'Jumps a year', hint: 'Average number of jumps', value: (p) => p.lambda_j, show: plain(0) },
        { key: 'mu_j', label: 'Average jump', hint: 'Mean log-size of a jump', value: (p) => p.mu_j, show: signedPct },
        { key: 'sigma_j', label: 'Jump size spread', hint: 'Standard deviation of a jump', value: (p) => p.sigma_j, show: vol },
      ];
    case 'regime_switching_gbm': {
      const n = Object.keys(params).filter((k) => /^sigma_\d+$/.test(k)).length;
      return Array.from({ length: n }, (_, k): Row => ({
        key: `sigma_${k}`,
        label: `Regime ${k} volatility${params.regime0 === k ? ' (today)' : ''}`,
        hint: `Lasts about ${days(1 / Math.max(1 - params[`p_${k}${k}`], 1e-9))} at a time`,
        value: (p) => p[`sigma_${k}`],
        show: vol,
      }));
    }
    case 'garch':
      return [
        { key: 'h0', label: 'Volatility today', hint: 'The GARCH filter at the end of the window', value: (p) => Math.sqrt(p.h0 * DAYS_PER_YEAR), show: vol },
        { key: 'omega', label: 'Long-run volatility', hint: 'sqrt(ω / (1 − α − β)), annualized', value: (p) => Math.sqrt((p.omega / (1 - p.alpha - p.beta)) * DAYS_PER_YEAR), show: vol },
        { key: 'alpha', label: 'Shock reaction α', hint: 'How much a big day raises tomorrow\'s variance', value: (p) => p.alpha, show: plain(3) },
        { key: 'beta', label: 'Shock half-life', hint: 'Persistence α + β', value: (p) => Math.log(0.5) / Math.log(p.alpha + p.beta), show: halfLife },
        ...(params.nu === undefined ? [] : [
          { key: 'nu', label: 'Tail index ν', hint: 'Student-t shocks: lower is fatter-tailed', value: (p: Record<string, number>) => p.nu, show: plain(1) },
        ]),
      ];
    case 'rough_bergomi':
      return [
        { key: 'xi0', label: 'Volatility today', hint: 'Trailing 21-day realized vol: the forward variance', value: (p) => Math.sqrt(p.xi0), show: vol },
        { key: 'eta', label: 'Vol of variance η', hint: 'Scale of log-variance moves', value: (p) => p.eta, show: plain(2) },
        { key: 'H', label: 'Roughness H', hint: 'Below 0.5 is rough; daily closes barely identify it', value: (p) => p.H, show: plain(2) },
        { key: 'rho', label: 'Spot-vol correlation ρ', hint: 'Negative means vol rises when the price falls', value: (p) => p.rho, show: plain(2) },
      ];
  }
}

const ROW = 46;
const LABEL = 280;

/** Each parameter of the world as a dot, the what-if's beside it: what the counterfactual moved, and how far. */
const WorldShift: React.FC<{ rows: Row[]; factual: Record<string, number>; shocked?: Record<string, number>;
  theme: ThemeName }> = ({ rows: list, factual, shocked, theme }) => {
  const c = chrome(theme);
  const [ref, width] = useWidth<HTMLDivElement>(600);
  // on a phone the label sits above its track
  const narrow = width < 520;
  const label = narrow ? 0 : LABEL;
  const row = narrow ? 58 : ROW;
  return (
    <div ref={ref}>
      <svg width={width} height={list.length * row} role="img" aria-label="The world's parameters, factual and what-if">
        {list.map((r, i) => {
          const before = r.value(factual);
          const after = shocked ? r.value(shocked) : null;
          const lo = Math.min(0, before, after ?? before);
          const hi = Math.max(0, before, after ?? before);
          const x = linear([lo - 0.08 * (hi - lo || 1), hi + 0.15 * (hi - lo || 1)], [label + 10, width - 60]);
          const y = narrow ? i * row + 38 : i * row + 26;
          const moved = after !== null && r.show(after) !== r.show(before);
          return (
            <g key={r.key}>
              {narrow
                ? <text x={0} y={y - 25} fill={c.ink} fontSize={12} fontWeight={moved ? 700 : 400}>{r.label}</text>
                : (
                  <>
                    <text x={label} y={y - 4} textAnchor="end" fill={c.ink} fontSize={12} fontWeight={moved ? 700 : 400}>
                      {r.label}
                    </text>
                    <text x={label} y={y + 10} textAnchor="end" fill={c.muted} fontSize={10}>{r.hint}</text>
                  </>
                )}
              <line x1={label + 10} x2={width - 60} y1={y} y2={y} stroke={c.grid} />
              {lo < 0 && hi > 0 && <line x1={x(0)} x2={x(0)} y1={y - 6} y2={y + 6} stroke={c.baseline} />}
              {moved && <line x1={x(before)} x2={x(after as number)} y1={y} y2={y} stroke={c.inkSecondary} strokeWidth={2} />}
              <circle cx={x(before)} cy={y} r={5} fill={c.world.factual} stroke={c.surface} strokeWidth={2} />
              <text x={x(before)} y={y - 9} textAnchor="middle" fill={c.inkSecondary} fontSize={11}>{r.show(before)}</text>
              {moved && (
                <>
                  <circle cx={x(after as number)} cy={y} r={5} fill={c.world.counterfactual} stroke={c.surface} strokeWidth={2} />
                  <text x={x(after as number)} y={y + 18} textAnchor="middle" fill={c.ink} fontSize={11} fontWeight={700}>
                    {r.show(after as number)}
                  </text>
                </>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
};

interface Props {
  catalog: LabCatalog;
  result: HedgeResponse;
  theme: ThemeName;
}

/** The book that ran, described from the result alone (shipped or visitor calibration): the
 * model's parameters in each world, the premium, rates and provenance. */
const MarketCard: React.FC<Props> = ({ catalog, result, theme }) => {
  const { book } = result;
  const { origin } = book;
  const info = catalog.models.find((m) => m.name === book.model);
  const factual = result.factual.params;
  const shocked = result.counterfactual?.params;
  const drift = book.rate - book.dividend_yield;
  const whose = origin.kind === 'visitor' ? 'your calibration of' : '';
  return (
    <div className="lab-market">
        <p className="lab-caption">
          {origin.kind === 'visitor' && <span className="lab-origin-badge">Your calibration</span>}{' '}
          {info?.label}, {whose} {book.asset}'s daily closes over {origin.window.replace('..', ' to ')}.
          Prices are normalized so today's spot is 100.
        </p>
        <WorldShift rows={rows(book.model, factual)} factual={factual} shocked={shocked} theme={theme} />
        <p className="lab-note">
          <span className="lab-key lab-key-factual" aria-hidden /> as calibrated
          {shocked && <>, <span className="lab-key lab-key-counterfactual" aria-hidden /> {result.counterfactual?.label}</>}.
          {' '}The spot drifts at r − q = {pct(drift)} a year in every world: no excess drift for books up to 3 months.
        </p>
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
  );
};

export default MarketCard;
