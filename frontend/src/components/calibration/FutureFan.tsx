import React, { useId, useMemo, useState } from 'react';
import type { DriftScenario, SimulationResponse } from '../../services/calibrationApi';
import { chrome, ThemeName } from '../lab/chartTheme';
import { extent, linear, linePath, Scale, ticks, useAnimationClock, useWidth } from '../lab/viz';

export const HORIZONS = [
  { days: 21, label: '1 mo' },
  { days: 63, label: '3 mo' },
  { days: 126, label: '6 mo' },
  { days: 252, label: '1 yr' },
  { days: 504, label: '2 yr' },
];

interface Props {
  ticker: string;
  model: string;
  /** The stock's real history, oldest first. */
  dates: string[];
  closes: number[];
  simulation: SimulationResponse;
  theme: ThemeName;
  horizon: number;
  onHorizon: (days: number) => void;
  simulating: boolean;
}

// The animation: the real history draws first, then path k starts at PAST + k * STAGGER and
// takes DRAW to reach the horizon; the bands and the median fade in once every path is out.
const PAST = 650;
const STAGGER = 45;
const DRAW = 800;
const REVEAL = 650;

const HEIGHT = 340;
const AXIS = { left: 52, right: 104, top: 24, bottom: 30 };
// On a phone the end labels drop their names (the readout below has them) to leave the plot room.
const NARROW = 560;
const NARROW_RIGHT = 62;
const LABEL_GAP = 13;

const clamp01 = (v: number) => Math.min(Math.max(v, 0), 1);

/** 21 → "1 mo", 252 → "1 yr", 10 → "2 wk" (trading days). */
export function spanLabel(days: number): string {
  if (days % 252 === 0) return `${days / 252} yr`;
  if (days % 21 === 0) return `${days / 21} mo`;
  return `${Math.round(days / 5)} wk`;
}

/** Day offsets for the x axis: today and round spans either side of it. */
export function dayTicks(past: number, ahead: number, maxTicks: number): number[] {
  const step = [5, 10, 21, 42, 63, 126, 252].find((s) => (past + ahead) / s <= maxTicks) ?? 252;
  const out: number[] = [];
  for (let d = -Math.floor(past / step) * step; d <= ahead; d += step) out.push(d);
  return out;
}

const dayLabel = (d: number) => (d === 0 ? 'today' : d < 0 ? `${spanLabel(-d)} ago` : `in ${spanLabel(d)}`);

/** Spread labels so none sits closer than ``gap`` px to the next, keeping them inside [lo, hi]. */
export function spreadLabels<T extends { y: number }>(labels: T[], gap: number, lo: number, hi: number): T[] {
  const out = [...labels].sort((a, b) => a.y - b.y).map((l) => ({ ...l }));
  for (let i = 1; i < out.length; i++) out[i].y = Math.max(out[i].y, out[i - 1].y + gap);
  if (out.length > 0 && out[out.length - 1].y > hi) {
    out[out.length - 1].y = hi;
    for (let i = out.length - 2; i >= 0; i--) out[i].y = Math.min(out[i].y, out[i + 1].y - gap);
  }
  if (out.length > 0 && out[0].y < lo) {
    out[0].y = lo;
    for (let i = 1; i < out.length; i++) out[i].y = Math.max(out[i].y, out[i - 1].y + gap);
  }
  return out;
}

/** A price as wide as it needs: 104,321 / 182.40 / 0.0412. */
export function price(v: number): string {
  const a = Math.abs(v);
  if (a >= 10000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 });
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

const axisPrice = (v: number) => (Math.abs(v) >= 10000 ? `${Number((v / 1000).toFixed(1))}k` : `${v}`);

const shortDate = (iso: string) => new Date(`${iso.slice(0, 10)}T12:00:00Z`)
  .toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });

const driftLabel = (s: DriftScenario) => (s.kind === 'historical'
  ? 'historical drift'
  : `${s.excess_drift > 0 ? '+' : '−'}${Math.abs(100 * s.excess_drift).toFixed(0)}% a year`);

function bandPath(lower: number[], upper: number[], x: Scale, y: Scale): string {
  let d = '';
  for (let i = 0; i < upper.length; i++) d += `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(upper[i]).toFixed(1)}`;
  for (let i = lower.length - 1; i >= 0; i--) d += `L${x(i).toFixed(1)},${y(lower[i]).toFixed(1)}`;
  return `${d}Z`;
}

/**
 * The stock's real prices up to today, then the futures the fitted model draws
 * from today's price: one line per simulated future, and the bands that hold the
 * middle half and the middle 90% of all of them.
 */
const FutureFan: React.FC<Props> = ({ ticker, model, dates, closes, simulation, theme, horizon, onHorizon, simulating }) => {
  const c = chrome(theme);
  const color = c.world.factual;
  const clipId = useId();
  const [ref, width] = useWidth<HTMLDivElement>(720);
  const [hover, setHover] = useState<number | null>(null);
  const [showDrifts, setShowDrifts] = useState(false);

  const narrow = width < NARROW;
  const right = narrow ? NARROW_RIGHT : AXIS.right;
  const paths = simulation.sample_paths;
  const q = simulation.quantiles;
  const ahead = paths[0].length - 1;
  const pastDays = Math.min(ahead, closes.length - 1);
  const past = useMemo(() => closes.slice(closes.length - 1 - pastDays), [closes, pastDays]);
  const pastDates = useMemo(() => dates.slice(dates.length - 1 - pastDays), [dates, pastDays]);
  const s0 = paths[0][0];
  const drifts = useMemo(() => (showDrifts ? simulation.scenarios : []), [showDrifts, simulation]);

  const total = PAST + (paths.length - 1) * STAGGER + DRAW + REVEAL;
  const clock = useAnimationClock(total, simulation);
  const pastProgress = clamp01(clock.t / PAST);
  const progress = (k: number) => clamp01((clock.t - PAST - k * STAGGER) / DRAW);
  const reveal = clamp01((clock.t - PAST - (paths.length - 1) * STAGGER - DRAW) / REVEAL);
  const drawn = paths.filter((_, k) => progress(k) > 0).length;

  const [lo, hi] = useMemo(() => {
    const [a, b] = extent([...past, ...q.p5, ...q.p95, ...drifts.flatMap((s) => s.median)], 0);
    const pad = 0.12 * (b - a || b);
    return [Math.max(0, a - pad), b + pad];
  }, [past, q, drifts]);
  // Scales and every finished path's outline only change with the data and the width,
  // not with the animation frame.
  const geometry = useMemo(() => {
    const x = linear([-pastDays, ahead], [AXIS.left, width - right]);
    const y = linear([lo, hi], [HEIGHT - AXIS.bottom, AXIS.top]);
    const xFuture = linear([0, ahead], [x(0), x(ahead)]);
    const xPast = linear([0, pastDays], [x(-pastDays), x(0)]);
    return {
      x, y, xFuture, xPast,
      outlines: paths.map((p) => linePath(p, xFuture, y)),
      outer: bandPath(q.p5, q.p95, xFuture, y),
      inner: bandPath(q.p25, q.p75, xFuture, y),
      median: linePath(q.p50, xFuture, y),
    };
  }, [width, right, lo, hi, pastDays, ahead, paths, q]);
  const { x, y, xFuture, xPast } = geometry;

  const endLabels = spreadLabels([
    { key: 'p95', y: y(q.p95[ahead]), name: '95th pct', value: q.p95[ahead] },
    { key: 'p50', y: y(q.p50[ahead]), name: 'median', value: q.p50[ahead] },
    { key: 'p5', y: y(q.p5[ahead]), name: '5th pct', value: q.p5[ahead] },
    ...drifts.map((s) => ({ key: s.kind, y: y(s.median[ahead]), name: driftLabel(s), value: null })),
  ], LABEL_GAP, AXIS.top, HEIGHT - AXIS.bottom);

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const share = (e.clientX - box.left) / box.width;
    setHover(Math.min(Math.max(Math.round(-pastDays + share * (pastDays + ahead)), -pastDays), ahead));
  };

  const readout = () => {
    if (hover !== null && hover <= 0) {
      const i = pastDays + hover;
      return <><strong>{shortDate(pastDates[i])}</strong><span>{ticker} closed at {price(past[i])}</span></>;
    }
    const d = hover ?? ahead;
    return (
      <>
        <strong>{d === ahead ? `In ${spanLabel(ahead)}` : `${d} trading days ahead`}</strong>
        <span>median {price(q.p50[d])}</span>
        <span>half of the futures between {price(q.p25[d])} and {price(q.p75[d])}</span>
        <span>90% between {price(q.p5[d])} and {price(q.p95[d])}</span>
      </>
    );
  };

  const plotWidth = Math.max(width - AXIS.left - right, 0);
  return (
    <div className="calib-fan" ref={ref}>
      <div className="calib-fan-head">
        <span className="calib-fan-label">How far ahead</span>
        <div className="lab-segmented" role="radiogroup" aria-label="How far ahead">
          {HORIZONS.map((h) => (
            <button key={h.days} type="button" role="radio" aria-checked={h.days === horizon} disabled={simulating}
              className={h.days === horizon ? 'active' : ''} onClick={() => onHorizon(h.days)}>{h.label}</button>
          ))}
        </div>
        {simulation.scenarios.length > 0 && (
          <button type="button" className={`lab-chip ${showDrifts ? 'active' : ''}`} aria-pressed={showDrifts}
            title="Median paths under other drifts: the simulation itself drifts at r − q"
            onClick={() => setShowDrifts(!showDrifts)}>Other drifts</button>
        )}
        <span className="lab-hero-count calib-fan-count" aria-live="off">
          {clock.done
            ? `${paths.length} of ${simulation.n_paths.toLocaleString('en-US')} simulated futures shown`
            : `drawing future ${Math.max(drawn, 1)} of ${paths.length}`}
        </span>
      </div>

      <svg width={width} height={HEIGHT} role="img"
        aria-label={`${ticker}'s last ${pastDays} closing prices, then ${paths.length} futures simulated by ${model} over ${ahead} trading days`}>
        <defs>
          <clipPath id={clipId}>
            <rect x={AXIS.left} y={AXIS.top - 4} width={plotWidth} height={HEIGHT - AXIS.top - AXIS.bottom + 8} />
          </clipPath>
        </defs>
        {ticks(lo, hi, 5).map((v) => (
          <g key={v}>
            <line x1={AXIS.left} x2={width - right} y1={y(v)} y2={y(v)} stroke={c.grid} />
            <text x={AXIS.left - 6} y={y(v) + 4} textAnchor="end" fill={c.muted} fontSize={11}>{axisPrice(v)}</text>
          </g>
        ))}
        {dayTicks(pastDays, ahead, width < 560 ? 3 : 6).map((d) => (
          <text key={d} x={x(d)} y={HEIGHT - AXIS.bottom + 17} textAnchor="middle" fill={d === 0 ? c.inkSecondary : c.muted}
            fontSize={11} fontWeight={d === 0 ? 700 : 400}>{dayLabel(d)}</text>
        ))}
        <line x1={x(0)} x2={x(0)} y1={AXIS.top - 10} y2={HEIGHT - AXIS.bottom} stroke={c.baseline} strokeDasharray="3 3" />
        {x(0) - AXIS.left > 110 && (
          <text x={x(0) - 8} y={AXIS.top - 12} textAnchor="end" fill={c.inkSecondary} fontSize={11}>what happened</text>
        )}
        <text x={x(0) + 8} y={AXIS.top - 12} fill={c.inkSecondary} fontSize={11}>what could happen</text>

        <g clipPath={`url(#${clipId})`}>
          {reveal > 0 && (
            <g opacity={reveal}>
              <path d={geometry.outer} fill={color} fillOpacity={0.13} />
              <path d={geometry.inner} fill={color} fillOpacity={0.2} />
            </g>
          )}
          {paths.map((p, k) => {
            const t = progress(k);
            if (t <= 0) return null;
            return t < 1
              ? <path key={k} d={linePath(p, xFuture, y, t * ahead)} fill="none" stroke={color} strokeWidth={1.6} />
              : <path key={k} d={geometry.outlines[k]} fill="none" stroke={color} strokeWidth={1}
                strokeOpacity={0.34 - 0.16 * reveal} />;
          })}
          {reveal > 0 && <path d={geometry.median} fill="none" stroke={color} strokeWidth={2.5} opacity={reveal} />}
          {drifts.map((s) => (
            <path key={s.kind} d={linePath(s.median, xFuture, y)} fill="none" stroke={c.inkSecondary} strokeWidth={1.4}
              strokeDasharray={s.kind === 'historical' ? '2 3' : '6 4'} />
          ))}
          <path d={linePath(past, xPast, y, pastProgress * pastDays)} fill="none" stroke={c.ink} strokeWidth={1.6} />
        </g>
        <circle cx={x(0)} cy={y(s0)} r={4} fill={c.ink} stroke={c.surface} strokeWidth={2} />

        {reveal > 0 && endLabels.map((l) => (
          <text key={l.key} x={width - right + 6} y={l.y + 4} fontSize={narrow ? 10 : 11} opacity={reveal}
            fill={l.key === 'p50' ? c.ink : c.inkSecondary} fontWeight={l.key === 'p50' ? 700 : 400}>
            {l.value === null ? l.name
              : narrow ? price(l.value) : <><tspan fill={c.muted} fontWeight={400}>{l.name}</tspan> {price(l.value)}</>}
          </text>
        ))}

        {hover !== null && (
          <g pointerEvents="none">
            <line x1={x(hover)} x2={x(hover)} y1={AXIS.top} y2={HEIGHT - AXIS.bottom} stroke={c.inkSecondary} />
            {hover <= 0
              ? <circle cx={x(hover)} cy={y(past[pastDays + hover])} r={3.5} fill={c.ink} />
              : <circle cx={x(hover)} cy={y(q.p50[hover])} r={3.5} fill={color} stroke={c.surface} strokeWidth={1.5} />}
          </g>
        )}
        <rect x={AXIS.left} y={AXIS.top} width={plotWidth} height={HEIGHT - AXIS.top - AXIS.bottom} fill="transparent"
          onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>

      <p className="lab-readout" aria-live="off">{readout()}</p>
      <div className="lab-hero-controls">
        {clock.playing
          ? <button type="button" className="lab-secondary" onClick={clock.pause}>Pause</button>
          : <button type="button" className="lab-secondary" onClick={clock.done ? clock.replay : clock.resume}>
            {clock.done ? 'Replay' : 'Play'}
          </button>}
        {!clock.done && <button type="button" className="lab-link" onClick={clock.skip}>Skip to the end</button>}
      </div>
      <p className="lab-hero-note">
        Left of today: {ticker}&apos;s real closing prices. Right: each thin line is one future the fitted {model} model
        considers possible, all starting from today&apos;s {price(s0)}. The shaded bands hold the middle half and the
        middle 90% of {simulation.n_paths.toLocaleString('en-US')} simulated futures; the thick line is their
        median. {simulation.drift_note}
      </p>
    </div>
  );
};

export default FutureFan;
