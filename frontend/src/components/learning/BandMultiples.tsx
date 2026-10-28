import React, { useMemo } from 'react';
import type { Bands, DeepHedgingDemo, HedgerKey } from '../../services/rlApi';
import { chrome, ThemeName } from '../lab/chartTheme';
import { extent, linear, Scale, ticks, useAnimationClock, useWidth } from '../lab/viz';
import { hedgerColor } from './learningTheme';
import { premiumOf } from './verdict';

interface Props {
  demo: DeepHedgingDemo;
  theme: ThemeName;
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

const Panel: React.FC<{ label: string; color: string; bands: Bands; domain: [number, number]; theme: ThemeName;
  upTo: number }> = ({ label, color, bands, domain, theme, upTo }) => {
  const c = chrome(theme);
  const [ref, width] = useWidth<HTMLDivElement>(240);
  const days = bands.p50.length - 1;
  const x = linear([0, days], [PAD.left, width - PAD.right]);
  const y = linear(domain, [HEIGHT - PAD.bottom, PAD.top]);
  const median = bands.p50.slice(0, Math.ceil(upTo) + 1)
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
  return (
    <figure className="lab-chart" ref={ref}>
      <figcaption><span className="lab-chart-title"><span className="learn-chip" style={{ background: color }} aria-hidden />{label}</span></figcaption>
      <svg width={width} height={HEIGHT} role="img" aria-label={`${label}: the book's value over the year, across all paths`}>
        {ticks(domain[0], domain[1], 4).map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={width - PAD.right} y1={y(v)} y2={y(v)} stroke={v === 0 ? c.baseline : c.grid} />
            <text x={PAD.left - 5} y={y(v) + 4} textAnchor="end" fill={c.muted} fontSize={10}>{v}</text>
          </g>
        ))}
        {ticks(0, days, 3).map((d) => (
          <text key={d} x={x(d)} y={HEIGHT - 6} textAnchor="middle" fill={c.muted} fontSize={10}>{d}</text>
        ))}
        <path d={band(bands.p5, bands.p95, x, y, upTo)} fill={color} fillOpacity={0.16} />
        <path d={band(bands.p25, bands.p75, x, y, upTo)} fill={color} fillOpacity={0.3} />
        <path d={median} fill="none" stroke={color} strokeWidth={1.8} />
      </svg>
    </figure>
  );
};

/** Each hedger's book marked to market through the year, across every evaluation path, on one scale. */
const HedgerBands: React.FC<Props> = ({ demo, theme }) => {
  const clock = useAnimationClock(SWEEP, demo);
  const bands = useMemo(() => {
    if (!demo.wealth_bands) return null;
    const premium = premiumOf(demo);
    const shift = (b: Bands): Bands => ({
      p5: b.p5.map((v) => v + premium), p25: b.p25.map((v) => v + premium), p50: b.p50.map((v) => v + premium),
      p75: b.p75.map((v) => v + premium), p95: b.p95.map((v) => v + premium),
    });
    return Object.fromEntries(Object.entries(demo.wealth_bands).map(([k, b]) => [k, shift(b)])) as Record<HedgerKey, Bands>;
  }, [demo]);
  if (!bands) return null;
  const keys = demo.hedgers.map((h) => h.key).filter((k): k is HedgerKey => !!bands[k]);
  const domain = extent(keys.flatMap((k) => [...bands[k].p5, ...bands[k].p95]), 0.05);
  const days = bands[keys[0]].p50.length - 1;
  return (
    <div>
      <div className="learn-multiples learn-bands">
        {demo.hedgers.filter((h) => bands[h.key]).map((h) => (
          <Panel key={h.key} label={h.label} color={hedgerColor(theme, h.key)} bands={bands[h.key]} domain={domain}
            theme={theme} upTo={(clock.t / SWEEP) * days} />
        ))}
      </div>
      <p className="lab-hero-note">
        What the book is worth each day: the hedge account (the premium of {premiumOf(demo).toFixed(2)}, plus gains,
        less costs) less the true value of the call still owed. It starts at 0 and ends at the P&amp;L. The bands hold
        the middle half and the middle 90% of {demo.setup.eval_paths.toLocaleString('en-US')} years; one scale for every
        hedger, so a wider band is a worse hedge.
      </p>
    </div>
  );
};

export default HedgerBands;
