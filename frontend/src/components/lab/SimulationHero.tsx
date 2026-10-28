import React, { useMemo, useState } from 'react';
import type { HedgeResponse, HedgerInfo, SimulatedPaths, WorldResult } from '../../services/labApi';
import { chrome, ThemeName, World } from './chartTheme';
import { extent, linear, linePath, Scale, ticks, useAnimationClock, useWidth } from './viz';

export type WorldView = 'factual' | 'counterfactual' | 'both';

interface Props {
  result: HedgeResponse;
  theme: ThemeName;
  view: WorldView;
  selected: number;
  onSelect: (path: number) => void;
}

// The animation: path k starts at k * STAGGER and takes DRAW to reach expiry, then its
// outcomes drop into the hedger rows; the full distribution fades in at the end.
const STAGGER = 85;
const DRAW = 650;
const DROP = 260;
const REVEAL = 700;

const FAN_HEIGHT = 300;
const ROW_HEIGHT = 50;
const AXIS = { left: 44, right: 12, top: 10, bottom: 38 };

function shownWorlds(result: HedgeResponse, view: WorldView): [World, WorldResult][] {
  const all: [World, WorldResult][] = [['factual', result.factual],
    ...(result.counterfactual ? [['counterfactual', result.counterfactual] as [World, WorldResult]] : [])];
  return all.filter(([w]) => view === 'both' || w === view || !result.counterfactual);
}

const clamp01 = (x: number) => Math.min(Math.max(x, 0), 1);
const easeOut = (x: number) => 1 - (1 - x) ** 3;

/** Deterministic vertical offset of path k's dot within a row, so dots do not pile up. */
const jitter = (k: number) => (((k * 37) % 11) - 5) / 5;

const SimulationHero: React.FC<Props> = ({ result, theme, view, selected, onSelect }) => {
  const c = chrome(theme);
  const worlds = shownWorlds(result, view);
  const first = result.factual.paths as SimulatedPaths;
  const nPaths = first.spots.length;
  const days = first.spots[0].length - 1;
  const total = (nPaths - 1) * STAGGER + DRAW + DROP + REVEAL;
  const clock = useAnimationClock(total, result);
  const [hover, setHover] = useState<number | null>(null);
  const [fanRef, fanWidth] = useWidth<HTMLDivElement>(640);
  const [rowsRef, rowsWidth] = useWidth<HTMLDivElement>(420);

  const progress = (k: number) => clamp01((clock.t - k * STAGGER) / DRAW);
  const dropped = (k: number) => clamp01((clock.t - k * STAGGER - DRAW) / DROP);
  const reveal = clamp01((clock.t - (nPaths - 1) * STAGGER - DRAW - DROP) / REVEAL);
  const drawing = Math.min(nPaths - 1, Math.max(0, Math.floor(clock.t / STAGGER)));
  const landed = Array.from({ length: nPaths }, (_, k) => dropped(k) > 0).filter(Boolean).length;

  // Fan scales
  const x = linear([0, days], [AXIS.left, fanWidth - AXIS.right]);
  const yDomain = useMemo(() => extent([
    ...shownWorlds(result, view).flatMap(([, w]) => (w.paths as SimulatedPaths).spots.flat()), result.book.strike,
  ], 0.04), [result, view]);
  const y = linear(yDomain, [FAN_HEIGHT - AXIS.bottom, AXIS.top]);
  const focus = hover ?? selected;

  // Outcome rows: hedged rows share a P&L axis; no hedge, about ten times wider, has its own
  const sharedDomain = useMemo(() => extent(result.hedgers.filter((h) => h.role !== 'none').flatMap((h) => {
    const e = result.bin_edges[h.name];
    return [e[0], e[e.length - 1]];
  }), 0), [result]);
  const rowScale = (h: HedgerInfo): Scale => {
    const e = result.bin_edges[h.name];
    const domain: [number, number] = h.role === 'none' ? [e[0], e[e.length - 1]] : sharedDomain;
    return linear(domain, [12, rowsWidth - 12]);
  };

  return (
    <div className="lab-hero">
      <figure className="lab-hero-fan" ref={fanRef}>
        <figcaption>
          <span className="lab-chart-title">Possible futures of {result.book.asset}</span>
          <span className="lab-hero-count" aria-live="off">
            {clock.done ? `${nPaths} of ${result.n_paths.toLocaleString()} simulated paths` : `simulating path ${drawing + 1}`}
          </span>
        </figcaption>
        <svg width={fanWidth} height={FAN_HEIGHT} role="img"
          aria-label={`${nPaths} simulated price paths over ${days} trading days`}>
          {ticks(yDomain[0], yDomain[1], 4).map((v) => (
            <g key={v}>
              <line x1={AXIS.left} x2={fanWidth - AXIS.right} y1={y(v)} y2={y(v)} stroke={c.grid} />
              <text x={AXIS.left - 6} y={y(v) + 4} textAnchor="end" fill={c.muted} fontSize={11}>{v}</text>
            </g>
          ))}
          {ticks(0, days, 4).map((d) => (
            <text key={d} x={x(d)} y={FAN_HEIGHT - AXIS.bottom + 15} textAnchor="middle" fill={c.muted} fontSize={11}>
              {d}
            </text>
          ))}
          <text x={(AXIS.left + fanWidth - AXIS.right) / 2} y={FAN_HEIGHT - 2} textAnchor="middle" fill={c.inkSecondary}
            fontSize={11}>Trading day</text>
          <line x1={AXIS.left} x2={fanWidth - AXIS.right} y1={y(result.book.strike)} y2={y(result.book.strike)}
            stroke={c.inkSecondary} strokeDasharray="5 4" />

          {worlds.map(([key, w]) => (w.paths as SimulatedPaths).spots.map((spots, k) => {
            const p = progress(k);
            if (p <= 0 || k === focus) return null;
            const active = p < 1;
            return (
              <path key={`${key}-${k}`} d={linePath(spots, x, y, p * days)} fill="none" stroke={c.world[key]}
                strokeWidth={active ? 1.6 : 1.1} strokeOpacity={active ? 0.95 : 0.3} />
            );
          }))}
          {worlds.map(([key, w]) => {
            const p = progress(focus);
            return p > 0 && (
              <path key={`${key}-focus`} d={linePath((w.paths as SimulatedPaths).spots[focus], x, y, p * days)}
                fill="none" stroke={c.world[key]} strokeWidth={2.6} />
            );
          })}
          {/* hit areas: the visible paths are too thin to click */}
          {clock.done && worlds.slice(0, 1).map(([key, w]) => (w.paths as SimulatedPaths).spots.map((spots, k) => (
            <path key={`hit-${key}-${k}`} d={linePath(spots, x, y)} fill="none" stroke="transparent" strokeWidth={9}
              style={{ cursor: 'pointer' }} onMouseEnter={() => setHover(k)} onMouseLeave={() => setHover(null)}
              onClick={() => onSelect(k)}>
              <title>Path {k + 1}: ends at {spots[days].toFixed(2)}</title>
            </path>
          )))}
          <circle cx={x(0)} cy={y(first.spots[0][0])} r={4} fill={c.ink} />
          <text x={fanWidth - AXIS.right - 2} y={y(result.book.strike) - 6} textAnchor="end" fill={c.inkSecondary}
            fontSize={11} stroke={c.surface} strokeWidth={4} paintOrder="stroke">strike {result.book.strike.toFixed(0)}</text>
        </svg>
        <p className="lab-hero-note">
          Each line is one future the {result.book.model === 'gbm' ? 'model' : 'calibrated model'} considers possible,
          day by day until expiry; the dashed line is the strike. Click a line to follow it below.
        </p>
      </figure>

      <figure className="lab-hero-outcomes" ref={rowsRef}>
        <figcaption>
          <span className="lab-chart-title">What the desk keeps at expiry</span>
          <span className="lab-hero-count">{landed} of {nPaths}</span>
        </figcaption>
        {result.hedgers.map((h) => {
          const s = rowScale(h);
          const e = result.bin_edges[h.name];
          return (
            <div key={h.name} className="lab-hero-row">
              <div className="lab-hero-row-head">
                <span className="lab-hero-row-name">{h.label}</span>
                <span className="lab-hero-row-stat" title="Hedging error: the spread of P&L across all simulated paths">
                  {worlds.map(([key, w]) => (
                    <span key={key}><span className={`lab-key lab-key-${key}`} aria-hidden />±{w.hedgers[h.name].std.toFixed(2)}</span>
                  ))}
                </span>
              </div>
              <svg width={rowsWidth} height={ROW_HEIGHT} role="img"
                aria-label={`${h.label}: P&L at expiry on each simulated path`}>
                {worlds.map(([key, w]) => {
                  const counts = w.hedgers[h.name].histogram;
                  const top = Math.max(...counts, 1);
                  return counts.map((n, i) => {
                    const x0 = s(e[i]);
                    const x1 = s(e[i + 1]);
                    const height = (ROW_HEIGHT - 16) * (n / top) * reveal;
                    return height > 0 && x1 > s.range[0] && x0 < s.range[1] ? (
                      <rect key={`${key}-${i}`} x={x0} width={Math.max(x1 - x0 - 0.5, 0.5)} y={ROW_HEIGHT - 12 - height}
                        height={height} fill={c.world[key]} fillOpacity={0.22} />
                    ) : null;
                  });
                })}
                {s.domain[0] < 0 && s.domain[1] > 0 && (
                  <line x1={s(0)} x2={s(0)} y1={2} y2={ROW_HEIGHT - 12} stroke={c.baseline} />
                )}
                {worlds.map(([key, w]) => (w.paths as SimulatedPaths).pnl[h.name].map((pnl, k) => {
                  const d = dropped(k);
                  if (d <= 0) return null;
                  const cx = Math.min(Math.max(s(pnl), s.range[0]), s.range[1]);
                  const cy = (ROW_HEIGHT - 12) / 2 + jitter(k) * 9 - (1 - easeOut(d)) * 18;
                  const isFocus = k === focus;
                  return (
                    <circle key={`${key}-${k}`} cx={cx} cy={cy} r={isFocus ? 4.5 : 3} fill={c.world[key]}
                      fillOpacity={isFocus ? 1 : 0.75 * d} stroke={isFocus ? c.ink : 'none'} strokeWidth={1.5}
                      style={{ cursor: 'pointer' }} onClick={() => onSelect(k)}
                      onMouseEnter={() => setHover(k)} onMouseLeave={() => setHover(null)}>
                      <title>Path {k + 1}: {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}</title>
                    </circle>
                  );
                }))}
                {ticks(s.domain[0], s.domain[1], 4).map((v) => (
                  <text key={v} x={s(v)} y={ROW_HEIGHT - 1} textAnchor="middle" fill={c.muted} fontSize={10}>{v}</text>
                ))}
              </svg>
              {h.role === 'none' && (
                <small className="lab-hero-row-note">Own scale: unhedged outcomes spread about ten times wider.</small>
              )}
            </div>
          );
        })}
        <p className="lab-hero-note">
          Each dot is one path: premium kept, plus the hedge's gains, minus the payoff and trading costs. A good hedge
          keeps the dots tight around 0; ± is the spread over all {result.n_paths.toLocaleString()} paths.
        </p>
      </figure>

      <div className="lab-hero-controls">
        {clock.playing
          ? <button type="button" className="lab-secondary" onClick={clock.pause}>Pause</button>
          : <button type="button" className="lab-secondary" onClick={clock.done ? clock.replay : clock.resume}>
            {clock.done ? 'Replay' : 'Play'}
          </button>}
        {!clock.done && <button type="button" className="lab-link" onClick={clock.skip}>Skip to the end</button>}
      </div>
    </div>
  );
};

export default SimulationHero;
