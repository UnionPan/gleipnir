import React from 'react';
import { chrome, ThemeName } from '../lab/chartTheme';
import { linear, useWidth } from '../lab/viz';

export interface BarRow {
  key: string;
  label: string;
  value: number;
  /** Spread across seeds: drawn as a whisker of ± sd. */
  sd?: number;
  color: string;
}

interface Props {
  title: string;
  rows: BarRow[];
  theme: ThemeName;
  digits?: number;
  note?: string;
}

const ROW = 26;
const LABEL = 150;
const VALUE = 74;

/** Horizontal bars, one per row, labelled on both sides; whiskers show the spread across seeds. */
const Bars: React.FC<Props> = ({ title, rows, theme, digits = 2, note }) => {
  const c = chrome(theme);
  const [ref, width] = useWidth<HTMLDivElement>(360);
  const top = Math.max(...rows.map((r) => r.value + (r.sd ?? 0)), 1e-9);
  const x = linear([0, top], [LABEL, Math.max(width - VALUE, LABEL + 40)]);
  return (
    <figure className="learn-bars" ref={ref}>
      <figcaption><span className="lab-chart-title">{title}</span></figcaption>
      <svg width={width} height={rows.length * ROW} role="img" aria-label={title}>
        {rows.map((r, i) => {
          const y = i * ROW;
          return (
            <g key={r.key}>
              <text x={LABEL - 8} y={y + ROW / 2 + 4} textAnchor="end" fill={c.ink} fontSize={12}>{r.label}</text>
              <rect x={LABEL} y={y + 6} width={Math.max(x(r.value) - LABEL, 1)} height={ROW - 12} rx={2} fill={r.color} />
              {r.sd !== undefined && r.sd > 0 && (
                <g stroke={c.ink} strokeWidth={1.25}>
                  <line x1={x(r.value - r.sd)} x2={x(r.value + r.sd)} y1={y + ROW / 2} y2={y + ROW / 2} />
                  <line x1={x(r.value - r.sd)} x2={x(r.value - r.sd)} y1={y + 8} y2={y + ROW - 8} />
                  <line x1={x(r.value + r.sd)} x2={x(r.value + r.sd)} y1={y + 8} y2={y + ROW - 8} />
                </g>
              )}
              <text x={x(r.value + (r.sd ?? 0)) + 6} y={y + ROW / 2 + 4} fill={c.inkSecondary} fontSize={12}>
                {r.value.toFixed(digits)}{r.sd !== undefined && r.sd > 0 ? ` ± ${r.sd.toFixed(digits)}` : ''}
              </text>
              <title>{r.label}: {r.value.toFixed(digits)}{r.sd ? ` ± ${r.sd.toFixed(digits)} across seeds` : ''}</title>
            </g>
          );
        })}
      </svg>
      {note && <p className="lab-hero-note">{note}</p>}
    </figure>
  );
};

export default Bars;
