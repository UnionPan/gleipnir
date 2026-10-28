import React from 'react';
import type { DeepHedgingDemo, HedgerKey } from '../../services/rlApi';
import { chrome, ThemeName } from '../lab/chartTheme';
import { linear, ticks, useWidth } from '../lab/viz';
import { hedgerColor } from './learningTheme';
import { premiumOf } from './verdict';

interface Props {
  demo: DeepHedgingDemo;
  theme: ThemeName;
  /** Each hedger's spread of outcomes, written at the end of its row. */
  spread: Record<HedgerKey, number>;
}

const ROW = 42;
const LABEL = 150;
// the ridges rise above their rows (a ridgeline), so the first needs headroom
const PAD = { top: 26, bottom: 28, right: 64 };

/**
 * The P&L at expiry over every evaluation path, one row per hedger on one axis:
 * a narrow ridge is a good hedge.
 */
const TerminalRidges: React.FC<Props> = ({ demo, theme, spread }) => {
  const c = chrome(theme);
  const [ref, width] = useWidth<HTMLDivElement>(640);
  const hist = demo.terminal_histogram;
  if (!hist) return null;
  // the export's P&L starts from no cash: credit the premium the desk sold the call for
  const premium = premiumOf(demo);
  const edges = hist.edges.map((e) => e + premium);
  const label = width < 520 ? 104 : LABEL;
  const x = linear([edges[0], edges[edges.length - 1]], [label, width - PAD.right]);
  const height = PAD.top + demo.hedgers.length * ROW + PAD.bottom;
  const top = Math.max(...demo.hedgers.flatMap((h) => hist[h.key].counts));
  return (
    <figure className="lab-chart" ref={ref}>
      <figcaption><span className="lab-chart-title">P&amp;L at expiry, every simulated year</span></figcaption>
      <svg width={width} height={height} role="img" aria-label="Distribution of each hedger's P&L at expiry">
        {ticks(edges[0], edges[edges.length - 1], 6).map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={PAD.top} y2={height - PAD.bottom} stroke={v === 0 ? c.baseline : c.grid} />
            <text x={x(v)} y={height - PAD.bottom + 14} textAnchor="middle" fill={c.muted} fontSize={10}>{v}</text>
          </g>
        ))}
        <text x={(label + width - PAD.right) / 2} y={height - 2} textAnchor="middle" fill={c.inkSecondary} fontSize={11}>
          P&amp;L at expiry (the option&apos;s units, spot = 100)
        </text>
        {demo.hedgers.map((h, i) => {
          const base = PAD.top + (i + 1) * ROW;
          const y = (n: number) => base - (n / top) * (ROW * 1.5);
          const counts = hist[h.key].counts;
          let d = `M${x(edges[0]).toFixed(1)},${base}`;
          counts.forEach((n, k) => {
            d += `L${x(edges[k]).toFixed(1)},${y(n).toFixed(1)}L${x(edges[k + 1]).toFixed(1)},${y(n).toFixed(1)}`;
          });
          d += `L${x(edges[edges.length - 1]).toFixed(1)},${base}Z`;
          const color = hedgerColor(theme, h.key);
          return (
            <g key={h.key}>
              <path d={d} fill={color} fillOpacity={0.3} stroke={color} strokeWidth={1.4} />
              <text x={label - 8} y={base - 4} textAnchor="end" fill={c.ink} fontSize={12}>{h.label}</text>
              <text x={width - PAD.right + 8} y={base - 4} fill={c.inkSecondary} fontSize={11}>
                ±{spread[h.key].toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="lab-hero-note">
        Each ridge is how one hedger&apos;s {demo.setup.eval_paths.toLocaleString('en-US')} years end: the premium of{' '}
        {premium.toFixed(2)}, plus the hedge&apos;s gains, less its costs and the payoff. A narrow ridge is a good hedge;
        ± is its spread (standard deviation).
      </p>
    </figure>
  );
};

export default TerminalRidges;
