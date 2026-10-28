import React, { useMemo } from 'react';
import type { Bands, HedgeResponse, HedgerInfo } from '../../services/labApi';
import { chrome, ThemeName, World } from './chartTheme';
import { extent, linear, Scale, ticks, useAnimationClock, useWidth } from './viz';
import type { WorldView } from './SimulationHero';

interface Props {
  result: HedgeResponse;
  theme: ThemeName;
  view: WorldView;
}

const HEIGHT = 170;
const PAD = { left: 40, right: 10, top: 10, bottom: 22 };
const SWEEP = 1400; // the bands grow from sale to expiry

function bandPath(lower: number[], upper: number[], x: Scale, y: Scale, upTo: number): string {
  const last = Math.max(1, Math.min(Math.ceil(upTo), upper.length - 1));
  let d = '';
  for (let i = 0; i <= last; i++) d += `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(upper[i]).toFixed(1)}`;
  for (let i = last; i >= 0; i--) d += `L${x(i).toFixed(1)},${y(lower[i]).toFixed(1)}`;
  return `${d}Z`;
}

const Panel: React.FC<{
  hedger: HedgerInfo; worlds: [World, Bands][]; domain: [number, number]; theme: ThemeName; upTo: number;
  ownScale: boolean;
}> = ({ hedger, worlds, domain, theme, upTo, ownScale }) => {
  const c = chrome(theme);
  const [ref, width] = useWidth<HTMLDivElement>(260);
  const days = worlds[0][1].p50.length - 1;
  const x = linear([0, days], [PAD.left, width - PAD.right]);
  const y = linear(domain, [HEIGHT - PAD.bottom, PAD.top]);
  const median = (b: Bands) => b.p50.slice(0, Math.ceil(upTo) + 1)
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
  return (
    <figure className="lab-chart" ref={ref}>
      <figcaption>
        <span className="lab-chart-title">{hedger.label}</span>
        {ownScale && <span className="lab-hero-count">own scale</span>}
      </figcaption>
      <svg width={width} height={HEIGHT} role="img"
        aria-label={`${hedger.label}: the book's value over time, middle half and middle 90% of paths`}>
        {ticks(domain[0], domain[1], 3).map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={width - PAD.right} y1={y(v)} y2={y(v)} stroke={v === 0 ? c.baseline : c.grid} />
            <text x={PAD.left - 5} y={y(v) + 4} textAnchor="end" fill={c.muted} fontSize={10}>{v}</text>
          </g>
        ))}
        {ticks(0, days, 3).map((d) => (
          <text key={d} x={x(d)} y={HEIGHT - 6} textAnchor="middle" fill={c.muted} fontSize={10}>{d}</text>
        ))}
        {worlds.map(([w, b]) => (
          <g key={w}>
            <path d={bandPath(b.p5, b.p95, x, y, upTo)} fill={c.world[w]} fillOpacity={0.14} />
            <path d={bandPath(b.p25, b.p75, x, y, upTo)} fill={c.world[w]} fillOpacity={0.24} />
            <path d={median(b)} fill="none" stroke={c.world[w]} strokeWidth={1.8} />
          </g>
        ))}
      </svg>
    </figure>
  );
};

/** Every hedger's book, marked to market each day, across all simulated paths of each world. */
const WealthBands: React.FC<Props> = ({ result, theme, view }) => {
  const worlds = useMemo(() => {
    const all: [World, Record<string, Bands> | undefined][] = [['factual', result.factual.mtm_bands],
      ...(result.counterfactual ? [['counterfactual', result.counterfactual.mtm_bands] as [World, Record<string, Bands> | undefined]] : [])];
    return all.filter(([w]) => view === 'both' || w === view || !result.counterfactual);
  }, [result, view]);
  const clock = useAnimationClock(SWEEP, result);
  const bands = worlds[0][1];
  // hedged books share one scale, so their spreads compare; no hedge swings far wider and has its own
  const hedged = useMemo(() => extent(result.hedgers.filter((h) => h.role !== 'none')
    .flatMap((h) => worlds.flatMap(([, b]) => (b ? [...b[h.name].p5, ...b[h.name].p95] : []))), 0.06), [result, worlds]);
  if (!bands) return null;
  const days = bands[result.hedgers[0].name].p50.length - 1;
  const upTo = (clock.t / SWEEP) * days;
  return (
    <section className="lab-panel" aria-label="The book's value over time">
      <div className="lab-titlebar">The book&apos;s value, day by day, on all {result.n_paths.toLocaleString('en-US')} paths</div>
      <div className="lab-panel-body">
        <div className="lab-multiples">
          {result.hedgers.map((h) => {
            const own = h.role === 'none';
            const domain = own
              ? extent(worlds.flatMap(([, b]) => (b ? [...b[h.name].p5, ...b[h.name].p95] : [])), 0.06)
              : hedged;
            return (
              <Panel key={h.name} hedger={h} theme={theme} upTo={upTo} ownScale={own} domain={domain}
                worlds={worlds.filter((e): e is [World, Record<string, Bands>] => !!e[1]).map(([w, b]) => [w, b[h.name]])} />
            );
          })}
        </div>
        <p className="lab-hero-note">
          Each day the book is marked to market: the hedge account (premium, plus the hedge&apos;s gains, less costs)
          minus the option sold, marked at its implied vol at sale. It starts at 0 and ends at the P&amp;L at expiry.
          The bands hold the middle half and the middle 90% of paths; the line is the median. A good hedge keeps the
          band thin all the way.
        </p>
      </div>
    </section>
  );
};

export default WealthBands;
