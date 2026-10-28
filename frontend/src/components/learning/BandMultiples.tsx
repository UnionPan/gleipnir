import React from 'react';
import type { Bands } from '../../services/rlApi';
import { chrome, ThemeName } from '../lab/chartTheme';
import { extent, linear, Scale, ticks, useAnimationClock, useWidth } from '../lab/viz';

export interface BandRow {
  key: string;
  label: string;
  color: string;
  bands: Bands;
}

interface Props {
  rows: BandRow[];
  theme: ThemeName;
  /** Restarts the sweep when it changes. */
  animationKey: unknown;
  /** x-axis value of each point (defaults to its index). */
  xs?: number[];
  /** The narrowest a panel gets before the row wraps (default 210 px). */
  minPanel?: number;
  note: React.ReactNode;
}

const HEIGHT = 160;
const PAD = { left: 36, right: 8, top: 8, bottom: 22 };
const SWEEP = 2200;

function band(lower: number[], upper: number[], x: Scale, y: Scale, upTo: number): string {
  const last = Math.max(1, Math.min(Math.ceil(upTo), upper.length - 1));
  let d = '';
  for (let i = 0; i <= last; i++) d += `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(upper[i]).toFixed(1)}`;
  for (let i = last; i >= 0; i--) d += `L${x(i).toFixed(1)},${y(lower[i]).toFixed(1)}`;
  return `${d}Z`;
}

const Panel: React.FC<{ row: BandRow; domain: [number, number]; theme: ThemeName; upTo: number; xs?: number[] }> = ({
  row, domain, theme, upTo, xs,
}) => {
  const c = chrome(theme);
  const [ref, width] = useWidth<HTMLDivElement>(240);
  const n = row.bands.p50.length - 1;
  const index = linear([0, n], [PAD.left, width - PAD.right]);
  const x = xs ? linear([xs[0], xs[n]], [PAD.left, width - PAD.right]) : index;
  const y = linear(domain, [HEIGHT - PAD.bottom, PAD.top]);
  const median = row.bands.p50.slice(0, Math.ceil(upTo) + 1)
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${index(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
  return (
    <figure className="lab-chart" ref={ref}>
      <figcaption>
        <span className="lab-chart-title"><span className="learn-chip" style={{ background: row.color }} aria-hidden />{row.label}</span>
      </figcaption>
      <svg width={width} height={HEIGHT} role="img" aria-label={`${row.label}: middle half and middle 90% of paths over time`}>
        {ticks(domain[0], domain[1], 4).map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={width - PAD.right} y1={y(v)} y2={y(v)} stroke={v === 0 ? c.baseline : c.grid} />
            <text x={PAD.left - 5} y={y(v) + 4} textAnchor="end" fill={c.muted} fontSize={10}>{v}</text>
          </g>
        ))}
        {ticks(xs ? xs[0] : 0, xs ? xs[n] : n, 3).map((d) => (
          <text key={d} x={x(d)} y={HEIGHT - 6} textAnchor="middle" fill={c.muted} fontSize={10}>{d}</text>
        ))}
        <path d={band(row.bands.p5, row.bands.p95, index, y, upTo)} fill={row.color} fillOpacity={0.16} />
        <path d={band(row.bands.p25, row.bands.p75, index, y, upTo)} fill={row.color} fillOpacity={0.3} />
        <path d={median} fill="none" stroke={row.color} strokeWidth={1.8} />
      </svg>
    </figure>
  );
};

/** One band chart per row on one shared scale, growing from the start: a wider band is a wider spread. */
const BandMultiples: React.FC<Props> = ({ rows, theme, animationKey, xs, minPanel = 210, note }) => {
  const clock = useAnimationClock(SWEEP, animationKey);
  if (rows.length === 0) return null;
  const domain = extent(rows.flatMap((r) => [...r.bands.p5, ...r.bands.p95]), 0.05);
  const n = rows[0].bands.p50.length - 1;
  return (
    <div>
      <div className="learn-multiples" style={{ gridTemplateColumns: `repeat(auto-fit, minmax(${minPanel}px, 1fr))` }}>
        {rows.map((r) => <Panel key={r.key} row={r} domain={domain} theme={theme} upTo={(clock.t / SWEEP) * n} xs={xs} />)}
      </div>
      <p className="lab-hero-note">{note}</p>
    </div>
  );
};

export default BandMultiples;
