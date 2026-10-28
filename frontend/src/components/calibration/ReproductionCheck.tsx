import React from 'react';
import type { CheckedAspect, PredictiveCheck } from '../../services/calibrationApi';
import { chrome, ThemeName } from '../lab/chartTheme';
import { extent, linear, useWidth } from '../lab/viz';
import { checkValue, intervalText } from './receiptFormat';

interface Props {
  check: PredictiveCheck;
  ticker: string;
  theme: ThemeName;
}

const VERDICTS: Record<CheckedAspect['verdict'], { icon: string; text: string }> = {
  fits: { icon: '✓', text: 'Reproduced' },
  more: { icon: '▲', text: 'More than every simulated history' },
  less: { icon: '▼', text: 'Less than every simulated history' },
};

const STRIP_HEIGHT = 42;
const NARROW = 620;

const Strip: React.FC<{ row: CheckedAspect; width: number; theme: ThemeName }> = ({ row, width, theme }) => {
  const c = chrome(theme);
  const color = c.world.factual;
  const x = linear(extent([row.q05, row.q95, row.observed], 0.18), [10, width - 10]);
  const mid = 20;
  const labelX = Math.min(Math.max(x(row.observed), 24), width - 24);
  return (
    <svg width={width} height={STRIP_HEIGHT} role="img"
      aria-label={`${row.aspect}: ${checkValue(row.observed)}; simulated histories ${intervalText(row.q05, row.q95)}`}>
      <line x1={10} x2={width - 10} y1={mid} y2={mid} stroke={c.grid} />
      <rect x={x(row.q05)} width={Math.max(x(row.q95) - x(row.q05), 2)} y={mid - 5} height={10} rx={5}
        fill={color} fillOpacity={0.3} />
      <line x1={x(row.q50)} x2={x(row.q50)} y1={mid - 7} y2={mid + 7} stroke={color} strokeWidth={2} />
      <text x={x(row.q05)} y={STRIP_HEIGHT - 2} textAnchor="middle" fill={c.muted} fontSize={10}>{checkValue(row.q05)}</text>
      <text x={x(row.q95)} y={STRIP_HEIGHT - 2} textAnchor="middle" fill={c.muted} fontSize={10}>{checkValue(row.q95)}</text>
      <circle cx={x(row.observed)} cy={mid} r={5.5} fill={c.ink} stroke={c.surface} strokeWidth={2} />
      <text x={labelX} y={mid - 9} textAnchor="middle" fill={c.ink} fontSize={11} fontWeight={700}
        stroke={c.surface} strokeWidth={3} paintOrder="stroke">{checkValue(row.observed)}</text>
    </svg>
  );
};

/** Each feature of the stock's history against the same feature in histories simulated from the fit. */
const ReproductionCheck: React.FC<Props> = ({ check, ticker, theme }) => {
  const c = chrome(theme);
  const [ref, width] = useWidth<HTMLDivElement>(720);
  const narrow = width < NARROW;
  const stripWidth = Math.max(narrow ? width : width - 400, 160);

  return (
    <div className="calib-check" ref={ref}>
      <div className="calib-check-legend" aria-hidden>
        <span><svg width={12} height={12}><circle cx={6} cy={6} r={5} fill={c.ink} /></svg>{ticker}&apos;s history</span>
        <span>
          <svg width={22} height={12}><rect x={1} y={1} width={20} height={10} rx={5} fill={c.world.factual} fillOpacity={0.3} /></svg>
          middle 90% of {check.n_replicates} simulated histories
        </span>
        <span><svg width={6} height={14}><line x1={3} x2={3} y1={0} y2={14} stroke={c.world.factual} strokeWidth={2} /></svg>their median</span>
      </div>
      <div className="calib-check-rows">
        {check.aspects.map((row) => {
          const verdict = VERDICTS[row.verdict];
          return (
            <div key={row.aspect} className={`calib-check-row ${narrow ? 'narrow' : ''}`}>
              <div className="calib-check-name">
                <strong>{row.aspect}</strong>
                <span>{row.summary}</span>
              </div>
              <Strip row={row} width={stripWidth} theme={theme} />
              <div className={`calib-verdict ${row.verdict === 'fits' ? 'fits' : 'misfit'}`}>
                <span aria-hidden>{verdict.icon}</span> {verdict.text}
              </div>
            </div>
          );
        })}
      </div>
      <p className="lab-hero-note">
        Each feature is measured on {ticker}&apos;s history and on {check.n_replicates} histories of the same length,
        each simulated from a draw of the fitted parameters. Every row has its own scale. A history with more or less of
        a feature than every simulated one shows something the model cannot produce, however well its parameters are
        pinned down.
      </p>
    </div>
  );
};

export default ReproductionCheck;
