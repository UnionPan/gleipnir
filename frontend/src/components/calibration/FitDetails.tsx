import React from 'react';
import type { CalibrationResponse } from '../../services/calibrationApi';
import { chrome, ThemeName } from '../lab/chartTheme';
import { extent, linear, useWidth } from '../lab/viz';
import { driftSentence, formatSig, identificationLabel, parameterDisplay } from './receiptFormat';

interface Props {
  result: CalibrationResponse;
  theme: ThemeName;
}

// A classical estimate's band: ±1.645 standard errors, the same 90% as the posterior's interval.
const Z90 = 1.645;
const ROW = 40;
const LABEL = 190;

interface Estimate {
  key: string;
  label: string;
  value: number;
  low: number | null;
  high: number | null;
}

function estimates(result: CalibrationResponse): Estimate[] {
  const posterior = result.diagnostics.posterior;
  return Object.entries(result.parameters)
    // the drift is reported in words below: simulations never use it
    .filter(([key]) => key !== 'mu' && !key.startsWith('mu_regime'))
    .map(([key, value]) => {
      const info = parameterDisplay(result.model, key);
      const summary = posterior?.parameters[key];
      const se = result.diagnostics.std_errors?.[key];
      const [low, high] = summary ? [summary.q05, summary.q95]
        : typeof se === 'number' && Number.isFinite(se) ? [value - Z90 * se, value + Z90 * se] : [null, null];
      return { key, label: `${info.name} ${info.symbol}`, value, low, high };
    });
}

/** Each estimate on its own scale: the dot, and the band its 90% interval spans. */
const EstimateStrips: React.FC<{ rows: Estimate[]; theme: ThemeName }> = ({ rows, theme }) => {
  const c = chrome(theme);
  const [ref, width] = useWidth<HTMLDivElement>(640);
  const label = width < 520 ? 130 : LABEL;
  return (
    <div ref={ref}>
      <svg width={width} height={rows.length * ROW} role="img" aria-label="Each parameter's estimate and its 90% interval">
        {rows.map((r, i) => {
          const y = i * ROW + 16;
          const x = linear(extent([r.value, r.low ?? r.value, r.high ?? r.value], 0.25), [label + 12, width - 16]);
          return (
            <g key={r.key}>
              <text x={label} y={y + 4} textAnchor="end" fill={c.ink} fontSize={12}>{r.label}</text>
              <line x1={label + 12} x2={width - 16} y1={y} y2={y} stroke={c.grid} />
              {r.low !== null && r.high !== null && (
                <>
                  <rect x={x(r.low)} width={Math.max(x(r.high) - x(r.low), 2)} y={y - 5} height={10} rx={5}
                    fill={c.world.factual} fillOpacity={0.3} />
                  <text x={x(r.low)} y={y + 20} textAnchor="middle" fill={c.muted} fontSize={10}>{formatSig(r.low, 3)}</text>
                  <text x={x(r.high)} y={y + 20} textAnchor="middle" fill={c.muted} fontSize={10}>{formatSig(r.high, 3)}</text>
                </>
              )}
              <circle cx={x(r.value)} cy={y} r={5} fill={c.world.factual} stroke={c.surface} strokeWidth={2} />
              <text x={x(r.value)} y={y - 9} textAnchor="middle" fill={c.ink} fontSize={11} fontWeight={700}
                stroke={c.surface} strokeWidth={3} paintOrder="stroke">{formatSig(r.value, 3)}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};

/** How much the history narrowed each quantity the network infers: a full bar pins it exactly. */
const Narrowing: React.FC<{ result: CalibrationResponse; theme: ThemeName }> = ({ result, theme }) => {
  const c = chrome(theme);
  const [ref, width] = useWidth<HTMLDivElement>(640);
  const label = width < 520 ? 130 : LABEL;
  const rows = Object.entries(result.diagnostics.posterior?.identification ?? {});
  const x = linear([0, 1], [label + 12, Math.max(width - 130, label + 60)]);
  return (
    <div ref={ref}>
      <svg width={width} height={rows.length * 28} role="img" aria-label="How much the history narrowed each quantity">
        {rows.map(([key, row], i) => {
          const y = i * 28 + 14;
          const share = Math.min(Math.max(row.contraction ?? 0, 0), 1);
          const info = parameterDisplay(result.model, key);
          return (
            <g key={key}>
              <text x={label} y={y + 4} textAnchor="end" fill={c.ink} fontSize={12}>{info.name} {info.symbol}</text>
              <rect x={x(0)} width={x(1) - x(0)} y={y - 5} height={10} rx={5} fill={c.grid} />
              <rect x={x(0)} width={Math.max(x(share) - x(0), 0)} y={y - 5} height={10} rx={5} fill={c.world.factual} />
              <text x={x(1) + 8} y={y + 4} fill={c.inkSecondary} fontSize={11}>{identificationLabel(row.contraction)}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};

/** The fit behind the headline, drawn: the estimates with their uncertainty, and what the data pin down. */
const FitDetails: React.FC<Props> = ({ result, theme }) => {
  const d = result.diagnostics;
  const posterior = d.posterior;
  const rows = estimates(result);
  const narrowed = posterior && Object.keys(posterior.identification).length > 0;
  return (
    <div className="calib-details">
      {d.converged === false && (
        <p className="lab-error-inline" role="alert">
          The optimizer stopped without converging: treat these values as unreliable.
        </p>
      )}
      <figure className="lab-chart">
        <figcaption><span className="lab-chart-title">The estimates</span></figcaption>
        <EstimateStrips rows={rows} theme={theme} />
        <p className="lab-hero-note">
          The dot is the {posterior ? 'posterior mean' : 'estimate'}; the band spans its 90%{' '}
          {posterior ? 'credible interval' : 'confidence interval'}. Each row has its own scale.{' '}
          {d.method_label}.
        </p>
      </figure>
      {narrowed && (
        <figure className="lab-chart">
          <figcaption><span className="lab-chart-title">What the history pins down</span></figcaption>
          <Narrowing result={result} theme={theme} />
          <p className="lab-hero-note">
            How much reading {result.ticker}&apos;s history narrowed each quantity beyond the prior ({posterior?.prior}).
            An empty bar leaves the prior&apos;s spread; a full one would pin the quantity exactly.
          </p>
        </figure>
      )}
      {(d.drift || d.note) && (
        <p className="lab-hero-note">
          {d.drift && <>{driftSentence(d.drift)} Too uncertain to forecast with, so the futures drift at r − q. </>}
          {d.note}
        </p>
      )}
    </div>
  );
};

export default FitDetails;
