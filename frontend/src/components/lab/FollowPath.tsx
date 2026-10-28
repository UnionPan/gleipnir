import React, { useState } from 'react';
import type { HedgeResponse, SimulatedPaths } from '../../services/labApi';
import { chrome, ThemeName, World } from './chartTheme';
import { extent, linear, linePath, ticks, useWidth } from './viz';
import type { WorldView } from './SimulationHero';

interface Props {
  result: HedgeResponse;
  theme: ThemeName;
  view: WorldView;
  path: number;
  onPath: (path: number) => void;
}

const HEIGHT = 150;
const PAD = { left: 40, right: 10, top: 8, bottom: 22 };

interface Series {
  world: World;
  values: number[];
}

/** One small line chart with the shared day cursor. */
const Mini: React.FC<{
  title: string; series: Series[]; days: number; theme: ThemeName; cursor: number | null;
  onCursor: (day: number | null) => void; reference?: number; digits: number;
}> = ({ title, series, days, theme, cursor, onCursor, reference, digits }) => {
  const c = chrome(theme);
  const [ref, width] = useWidth<HTMLDivElement>(300);
  const x = linear([0, days], [PAD.left, width - PAD.right]);
  const domain = extent([...series.flatMap((s) => s.values), ...(reference === undefined ? [] : [reference])], 0.08);
  const y = linear(domain, [HEIGHT - PAD.bottom, PAD.top]);
  const move = (e: React.MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const day = Math.round(((e.clientX - box.left - PAD.left) / (width - PAD.left - PAD.right)) * days);
    onCursor(Math.min(Math.max(day, 0), days));
  };
  return (
    <figure className="lab-chart" ref={ref}>
      <figcaption><span className="lab-chart-title">{title}</span></figcaption>
      <svg width={width} height={HEIGHT} onMouseMove={move} onMouseLeave={() => onCursor(null)} role="img" aria-label={title}>
        {ticks(domain[0], domain[1], 3).map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={width - PAD.right} y1={y(v)} y2={y(v)} stroke={c.grid} />
            <text x={PAD.left - 5} y={y(v) + 4} textAnchor="end" fill={c.muted} fontSize={10}>{Number(v.toFixed(digits))}</text>
          </g>
        ))}
        {ticks(0, days, 4).map((d) => (
          <text key={d} x={x(d)} y={HEIGHT - 6} textAnchor="middle" fill={c.muted} fontSize={10}>{d}</text>
        ))}
        {reference !== undefined && (
          <line x1={PAD.left} x2={width - PAD.right} y1={y(reference)} y2={y(reference)} stroke={c.inkSecondary}
            strokeDasharray="5 4" />
        )}
        {series.map((s) => (
          <path key={s.world} d={linePath(s.values, x, y)} fill="none" stroke={c.world[s.world]} strokeWidth={2} />
        ))}
        {cursor !== null && (
          <g>
            <line x1={x(cursor)} x2={x(cursor)} y1={PAD.top} y2={HEIGHT - PAD.bottom} stroke={c.baseline} />
            {series.map((s) => s.values[cursor] !== undefined && (
              <circle key={s.world} cx={x(cursor)} cy={y(s.values[cursor])} r={3.5} fill={c.world[s.world]} />
            ))}
          </g>
        )}
      </svg>
    </figure>
  );
};

const FollowPath: React.FC<Props> = ({ result, theme, view, path, onPath }) => {
  const worlds: [World, SimulatedPaths][] = [['factual', result.factual.paths as SimulatedPaths],
    ...(result.counterfactual?.paths ? [['counterfactual', result.counterfactual.paths] as [World, SimulatedPaths]] : [])];
  const shown = worlds.filter(([w]) => view === 'both' || w === view || worlds.length === 1);
  const traders = result.hedgers.filter((h) => h.role !== 'none');
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [cursor, setCursor] = useState<number | null>(null);
  const nPaths = worlds[0][1].spots.length;
  const days = worlds[0][1].spots[0].length - 1;
  const at = cursor ?? days;
  const label = (w: World) => (w === 'factual' ? result.factual.label : result.counterfactual?.label ?? '');
  const toggle = (name: string) => setHidden((prev) => {
    const next = new Set(prev);
    if (next.has(name)) next.delete(name); else next.add(name);
    return next;
  });

  return (
    <section className="lab-panel" aria-label="Follow one path">
      <div className="lab-titlebar">Follow one path</div>
      <div className="lab-panel-body">
        <div className="lab-follow-head">
          <div className="lab-stepper" role="group" aria-label="Path">
            <button type="button" className="lab-secondary" aria-label="Previous path"
              onClick={() => onPath((path + nPaths - 1) % nPaths)}>‹</button>
            <span>Path {path + 1} of {nPaths}</span>
            <button type="button" className="lab-secondary" aria-label="Next path"
              onClick={() => onPath((path + 1) % nPaths)}>›</button>
          </div>
          <div className="lab-chips" role="group" aria-label="Hedgers shown">
            {traders.map((h) => (
              <button key={h.name} type="button" aria-pressed={!hidden.has(h.name)}
                className={`lab-chip ${hidden.has(h.name) ? '' : 'active'}`} onClick={() => toggle(h.name)}>
                {h.label}
              </button>
            ))}
          </div>
        </div>
        <p className="lab-readout" aria-live="polite">
          <strong>{cursor === null ? `At expiry (day ${days})` : `Day ${at}`}</strong>
          {shown.map(([w, p]) => (
            <span key={w}>
              <span className={`lab-key lab-key-${w}`} aria-hidden />
              {shown.length > 1 ? `${label(w)}: ` : ''}spot {p.spots[path][at].toFixed(2)}
              {cursor === null
                ? traders.filter((h) => !hidden.has(h.name)).map((h) => ` · ${h.label} ${p.pnl[h.name][path] >= 0 ? '+' : ''}${p.pnl[h.name][path].toFixed(2)}`).join('')
                : traders.filter((h) => !hidden.has(h.name)).map((h) => ` · ${h.label} holds ${(p.positions[h.name][path][Math.min(at, days - 1)] ?? 0).toFixed(2)}`).join('')}
            </span>
          ))}
        </p>
        <div className="lab-multiples">
          <Mini title="Price" series={shown.map(([w, p]) => ({ world: w, values: p.spots[path] }))} days={days}
            theme={theme} cursor={cursor} onCursor={setCursor} reference={result.book.strike} digits={0} />
          {traders.filter((h) => !hidden.has(h.name)).map((h) => (
            <Mini key={h.name} title={`${h.label}: shares held`} days={days} theme={theme} cursor={cursor}
              onCursor={setCursor} digits={2}
              series={shown.map(([w, p]) => ({ world: w, values: p.positions[h.name][path] }))} />
          ))}
        </div>
        <p className="lab-hero-note">
          Shares each hedger holds against the option sold, after each day's trade. Hover a chart to read any day;
          without a cursor the line above shows each hedger's result at expiry.
        </p>
      </div>
    </section>
  );
};

export default FollowPath;
