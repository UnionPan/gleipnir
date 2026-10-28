import React, { useId, useMemo } from 'react';
import type { OptionSlot } from '../../services/rlApi';
import { chrome, ThemeName } from '../lab/chartTheme';
import { linear, ticks, useWidth } from '../lab/viz';

interface Props {
  title: string;
  /** Holdings of each slot (rows) after each day's trade (columns). */
  holdings: number[][];
  slots: OptionSlot[];
  /** Days drawn so far (the replay's cursor). */
  upTo: number;
  theme: ThemeName;
}

// Long and short: a diverging pair with a neutral midpoint, kept apart from the hedgers' colors.
const POLES: Record<ThemeName, { long: [number, number, number]; short: [number, number, number] }> = {
  light: { long: [106, 79, 214], short: [184, 125, 0] },
  dark: { long: [150, 125, 240], short: [214, 150, 20] },
};

const ROW = 14;
const LABEL = 96;
const PAD = { top: 4, bottom: 22, right: 10 };

const hex = (c: string) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));

export const slotLabel = (s: OptionSlot) => `${s.tenor_days}d ${s.kind} ${Math.round(100 * s.moneyness)}%`;

/** Each listed contract the hedger may trade, day by day: how many it holds, long or short. */
const OptionGrid: React.FC<Props> = ({ title, holdings, slots, upTo, theme }) => {
  const c = chrome(theme);
  const clipId = useId();
  const [ref, width] = useWidth<HTMLDivElement>(700);
  const days = holdings[0]?.length ?? 0;
  const height = PAD.top + slots.length * ROW + PAD.bottom;
  const x = linear([0, days], [LABEL, width - PAD.right]);
  // a robust scale: the 98th percentile of the sizes, so one large trade does not wash out the rest
  const scale = useMemo(() => {
    const sizes = holdings.flat().map(Math.abs).filter((v) => v > 1e-9).sort((a, b) => a - b);
    return sizes.length ? sizes[Math.floor(0.98 * (sizes.length - 1))] : 1;
  }, [holdings]);
  // every cell, built once per data and width: the replay only moves the clip over them
  const cells = useMemo(() => {
    const surface = hex(c.surface);
    const fill = (v: number) => {
      const t = Math.min(Math.abs(v) / scale, 1);
      const pole = v >= 0 ? POLES[theme].long : POLES[theme].short;
      return `rgb(${surface.map((s, i) => Math.round(s + (pole[i] - s) * t)).join(',')})`;
    };
    const at = linear([0, days], [LABEL, width - PAD.right]);
    const cell = Math.max(at(1) - at(0), 0.5);
    return holdings.map((row, r) => row.map((v, d) => (Math.abs(v) > 1e-6 ? (
      <rect key={`${r}-${d}`} x={at(d)} y={PAD.top + r * ROW + 1} width={cell + 0.4} height={ROW - 2} fill={fill(v)} />
    ) : null)));
  }, [holdings, scale, days, width, theme, c.surface]);
  return (
    <figure className="lab-chart learn-grid" ref={ref}>
      <figcaption>
        <span className="lab-chart-title">{title}</span>
        <span className="learn-grid-key" aria-hidden>
          short
          <svg width={84} height={10}>
            <defs>
              <linearGradient id={`${clipId}-key`}>
                <stop offset="0" stopColor={`rgb(${POLES[theme].short.join(',')})`} />
                <stop offset="0.5" stopColor={c.surface} />
                <stop offset="1" stopColor={`rgb(${POLES[theme].long.join(',')})`} />
              </linearGradient>
            </defs>
            <rect width={84} height={10} fill={`url(#${clipId}-key)`} stroke={c.baseline} />
          </svg>
          long
        </span>
      </figcaption>
      <svg width={width} height={height} role="img" aria-label={`${title}: holdings of each listed option, day by day`}>
        <defs>
          <clipPath id={clipId}>
            <rect x={LABEL} y={0} width={Math.max(x(Math.min(upTo, days)) - LABEL, 0)} height={height} />
          </clipPath>
        </defs>
        {slots.map((s, r) => (
          <text key={slotLabel(s)} x={LABEL - 6} y={PAD.top + r * ROW + ROW - 4} textAnchor="end" fill={c.inkSecondary}
            fontSize={10}>{slotLabel(s)}</text>
        ))}
        <rect x={LABEL} y={PAD.top} width={Math.max(width - PAD.right - LABEL, 0)} height={slots.length * ROW}
          fill={c.surface} stroke={c.grid} />
        <g clipPath={`url(#${clipId})`}>{cells}</g>
        {slots.map((s, r) => r > 0 && slots[r - 1].tenor_days !== s.tenor_days && (
          <line key={`sep-${r}`} x1={LABEL} x2={width - PAD.right} y1={PAD.top + r * ROW} y2={PAD.top + r * ROW}
            stroke={c.baseline} />
        ))}
        {ticks(0, days, 4).map((d) => (
          <text key={d} x={x(d)} y={height - 6} textAnchor="middle" fill={c.muted} fontSize={10}>{d}</text>
        ))}
        {upTo < days && (
          <line x1={x(upTo)} x2={x(upTo)} y1={PAD.top} y2={PAD.top + slots.length * ROW} stroke={c.inkSecondary} />
        )}
      </svg>
    </figure>
  );
};

export default OptionGrid;
