import React, { useState } from 'react';
import type { DeepHedgingDemo, HedgerKey } from '../../services/rlApi';
import { chrome, ThemeName } from '../lab/chartTheme';
import { extent, linear, linePath, ticks, useAnimationClock, useWidth } from '../lab/viz';
import { hedgerColor } from './learningTheme';

interface Props {
  demo: DeepHedgingDemo;
  theme: ThemeName;
}

const DURATION = 9000; // a simulated year in nine seconds
const PATH_NAMES = ['Price ends low', 'Price ends near the middle', 'Price ends high'];
const DEFAULT_SHOWN: HedgerKey[] = ['bs_delta', 'delta_gamma', 'lstm_options'];
const TRADES_OPTIONS = new Set<HedgerKey>(['delta_gamma', 'delta_gamma_vega', 'lstm_options']);
const PAD = { left: 42, right: 12, top: 8, bottom: 22 };

interface Line {
  key: string;
  values: number[];
  color: string;
  width?: number;
}

const ReplayChart: React.FC<{
  title: string; lines: Line[]; days: number; upTo: number; theme: ThemeName; height: number; reference?: number;
  digits: number;
}> = ({ title, lines, days, upTo, theme, height, reference, digits }) => {
  const c = chrome(theme);
  const [ref, width] = useWidth<HTMLDivElement>(500);
  const x = linear([0, days], [PAD.left, width - PAD.right]);
  const domain = extent([...lines.flatMap((l) => l.values), ...(reference === undefined ? [] : [reference])], 0.08);
  const y = linear(domain, [height - PAD.bottom, PAD.top]);
  return (
    <figure className="lab-chart" ref={ref}>
      <figcaption><span className="lab-chart-title">{title}</span></figcaption>
      <svg width={width} height={height} role="img" aria-label={title}>
        {ticks(domain[0], domain[1], 3).map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={width - PAD.right} y1={y(v)} y2={y(v)} stroke={c.grid} />
            <text x={PAD.left - 5} y={y(v) + 4} textAnchor="end" fill={c.muted} fontSize={10}>{Number(v.toFixed(digits))}</text>
          </g>
        ))}
        {ticks(0, days, 4).map((d) => (
          <text key={d} x={x(d)} y={height - 6} textAnchor="middle" fill={c.muted} fontSize={10}>{d}</text>
        ))}
        {reference !== undefined && (
          <line x1={PAD.left} x2={width - PAD.right} y1={y(reference)} y2={y(reference)} stroke={c.inkSecondary}
            strokeDasharray="5 4" />
        )}
        {lines.map((l) => {
          const last = Math.min(upTo, l.values.length - 1);
          const i = Math.floor(last);
          const head = l.values[i] + ((l.values[i + 1] ?? l.values[i]) - l.values[i]) * (last - i);
          return (
            <g key={l.key}>
              <path d={linePath(l.values, x, y, last)} fill="none" stroke={l.color} strokeWidth={l.width ?? 2} />
              {Number.isFinite(head) && <circle cx={x(last)} cy={y(head)} r={3.5} fill={l.color} />}
            </g>
          );
        })}
      </svg>
    </figure>
  );
};

/** One simulated year, day by day: the price, and what each hedger holds against the option sold. */
const YearReplay: React.FC<Props> = ({ demo, theme }) => {
  const c = chrome(theme);
  const [pathIndex, setPathIndex] = useState(1);
  const [shown, setShown] = useState<Set<HedgerKey>>(new Set(DEFAULT_SHOWN));
  const clock = useAnimationClock(DURATION, pathIndex);
  const sample = demo.samples[pathIndex];
  const days = sample.spot.length - 1;
  const upTo = (clock.t / DURATION) * days;
  const day = Math.min(Math.round(upTo), days - 1);
  const hedgers = demo.hedgers.filter((h) => shown.has(h.key));
  const withOptions = hedgers.filter((h) => TRADES_OPTIONS.has(h.key));
  const toggle = (key: HedgerKey) => setShown((prev) => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  return (
    <div className="learn-replay">
      <div className="learn-replay-head">
        <div className="learn-toggle" role="radiogroup" aria-label="Simulated year">
          {PATH_NAMES.map((name, i) => (
            <button key={name} type="button" role="radio" aria-checked={i === pathIndex}
              className={i === pathIndex ? 'active' : ''} onClick={() => setPathIndex(i)}>{name}</button>
          ))}
        </div>
        <div className="lab-chips" role="group" aria-label="Hedgers shown">
          {demo.hedgers.map((h) => (
            <button key={h.key} type="button" aria-pressed={shown.has(h.key)}
              className={`lab-chip ${shown.has(h.key) ? 'active' : ''}`} onClick={() => toggle(h.key)}
              style={shown.has(h.key) ? { borderColor: hedgerColor(theme, h.key) } : undefined}>
              <span className="learn-chip" style={{ background: hedgerColor(theme, h.key) }} aria-hidden />{h.label}
            </button>
          ))}
        </div>
      </div>

      <div className="learn-replay-charts">
        <ReplayChart title="Price of the stock" days={days} upTo={upTo} theme={theme} height={170} digits={0}
          reference={demo.setup.initial_spot} lines={[{ key: 'spot', values: sample.spot, color: c.ink, width: 2 }]} />
        <ReplayChart title="Shares held against the option" days={days} upTo={upTo} theme={theme} height={170} digits={2}
          lines={hedgers.map((h) => ({ key: h.key, values: sample.positions[h.key], color: hedgerColor(theme, h.key) }))} />
        {withOptions.length > 0 && (
          <ReplayChart title="Listed options held (premium)" days={days} upTo={upTo} theme={theme} height={150} digits={1}
            lines={withOptions.map((h) => ({ key: h.key, values: sample.option_value[h.key], color: hedgerColor(theme, h.key) }))} />
        )}
      </div>

      <div className="learn-replay-controls">
        {clock.playing
          ? <button type="button" className="lab-secondary" onClick={clock.pause}>Pause</button>
          : <button type="button" className="lab-secondary" onClick={clock.done ? clock.replay : clock.resume}>
            {clock.done ? 'Replay' : 'Play'}
          </button>}
        <input type="range" min={0} max={days} step={1} value={Math.round(upTo)} aria-label="Day"
          onChange={(e) => clock.seek((Number(e.target.value) / days) * DURATION)} />
        <span className="lab-hero-count">Day {Math.round(upTo)} of {days}</span>
      </div>
      <p className="lab-readout" aria-live="off">
        <strong>Price {sample.spot[Math.round(upTo)].toFixed(2)}</strong>
        {hedgers.map((h) => (
          <span key={h.key}>
            <span className="learn-chip" style={{ background: hedgerColor(theme, h.key) }} aria-hidden />
            {h.label} {sample.positions[h.key][day].toFixed(2)} shares
            {TRADES_OPTIONS.has(h.key) ? `, ${sample.option_value[h.key][day].toFixed(2)} in options` : ''}
          </span>
        ))}
      </p>
      <p className="lab-hero-note">
        One of {demo.setup.eval_paths.toLocaleString('en-US')} simulated years. Delta hedgers track the price smoothly;
        hedgers that may trade options also buy listed calls and puts, rolling them as they expire.
      </p>
    </div>
  );
};

export default YearReplay;
