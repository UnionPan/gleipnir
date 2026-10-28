import React from 'react';
import { chrome, ThemeName } from '../lab/chartTheme';
import { linear, ticks, useWidth } from '../lab/viz';

export interface RidgeRow {
  key: string;
  label: string;
  color: string;
  /** Counts on the shared bins. */
  counts: number[];
  /** Written at the end of the row (e.g. the spread). */
  end?: string;
}

interface Props {
  title: string;
  rows: RidgeRow[];
  /** Bin edges, shared by every row. */
  edges: number[];
  axisLabel: string;
  theme: ThemeName;
  note: React.ReactNode;
}

const ROW = 42;
const LABEL = 160;
// the ridges rise above their rows (a ridgeline), so the first needs headroom
const PAD = { top: 26, bottom: 28, right: 64 };

/** One distribution per row on one shared axis: a narrow ridge is a tight outcome. */
const Ridges: React.FC<Props> = ({ title, rows, edges, axisLabel, theme, note }) => {
  const c = chrome(theme);
  const [ref, width] = useWidth<HTMLDivElement>(640);
  const label = width < 520 ? 110 : LABEL;
  const x = linear([edges[0], edges[edges.length - 1]], [label, width - PAD.right]);
  const height = PAD.top + rows.length * ROW + PAD.bottom;
  return (
    <figure className="lab-chart" ref={ref}>
      <figcaption><span className="lab-chart-title">{title}</span></figcaption>
      <svg width={width} height={height} role="img" aria-label={title}>
        {ticks(edges[0], edges[edges.length - 1], 6).map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={PAD.top} y2={height - PAD.bottom} stroke={v === 0 ? c.baseline : c.grid} />
            <text x={x(v)} y={height - PAD.bottom + 14} textAnchor="middle" fill={c.muted} fontSize={10}>{v}</text>
          </g>
        ))}
        <text x={(label + width - PAD.right) / 2} y={height - 2} textAnchor="middle" fill={c.inkSecondary} fontSize={11}>
          {axisLabel}
        </text>
        {rows.map((r, i) => {
          const base = PAD.top + (i + 1) * ROW;
          // each ridge on its own height: its width is the message, and one hedger's point mass
          // (a premium kept whole) would otherwise flatten every other row
          const top = Math.max(...r.counts, 1);
          const y = (n: number) => base - (n / top) * (ROW * 1.5);
          let d = `M${x(edges[0]).toFixed(1)},${base}`;
          r.counts.forEach((n, k) => {
            d += `L${x(edges[k]).toFixed(1)},${y(n).toFixed(1)}L${x(edges[k + 1]).toFixed(1)},${y(n).toFixed(1)}`;
          });
          d += `L${x(edges[edges.length - 1]).toFixed(1)},${base}Z`;
          return (
            <g key={r.key}>
              <path d={d} fill={r.color} fillOpacity={0.3} stroke={r.color} strokeWidth={1.4} />
              <text x={label - 8} y={base - 4} textAnchor="end" fill={c.ink} fontSize={12}>{r.label}</text>
              {r.end && <text x={width - PAD.right + 8} y={base - 4} fill={c.inkSecondary} fontSize={11}>{r.end}</text>}
            </g>
          );
        })}
      </svg>
      <p className="lab-hero-note">{note}</p>
    </figure>
  );
};

export default Ridges;
