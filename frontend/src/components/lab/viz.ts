/** Small helpers for the lab's hand-drawn SVG charts: scales, ticks, path strings. */
import { useEffect, useRef, useState } from 'react';

export interface Scale {
  (v: number): number;
  domain: [number, number];
  range: [number, number];
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const f = ((v: number) => r0 + (v - d0) * k) as Scale;
  f.domain = domain;
  f.range = range;
  return f;
}

/** About ``count`` round tick values covering [lo, hi]. */
export function ticks(lo: number, hi: number, count = 4): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

/** [min, max] of the values, padded by ``pad`` of the span on each side. */
export function extent(values: Iterable<number>, pad = 0.05): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (Number.isFinite(v)) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  if (!Number.isFinite(lo)) return [0, 1];
  const span = hi - lo || Math.abs(hi) || 1;
  return [lo - pad * span, hi + pad * span];
}

/** An SVG path through the first ``upTo`` (fractional) points of ``ys`` at x = index. */
export function linePath(ys: number[], x: Scale, y: Scale, upTo = ys.length - 1): string {
  const last = Math.min(Math.max(upTo, 0), ys.length - 1);
  const whole = Math.floor(last);
  let d = '';
  for (let i = 0; i <= whole; i++) d += `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(ys[i]).toFixed(1)}`;
  const frac = last - whole;
  if (frac > 0 && whole + 1 < ys.length) {
    const v = ys[whole] + (ys[whole + 1] - ys[whole]) * frac;
    d += `L${x(whole + frac).toFixed(1)},${y(v).toFixed(1)}`;
  }
  return d;
}

/** The width of an element, kept current as it resizes. */
export function useWidth<T extends HTMLElement>(fallback = 600): [React.RefObject<T | null>, number] {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    setWidth(el.clientWidth || fallback);
    const observer = new ResizeObserver((entries) => setWidth(Math.round(entries[0].contentRect.width) || fallback));
    observer.observe(el);
    return () => observer.disconnect();
  }, [fallback]);
  return [ref, width];
}

const reducedMotion = () => typeof window !== 'undefined'
  && typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * A clock for one animation of ``duration`` ms, restarted whenever ``key`` changes.
 * With reduced motion it starts at the end.
 */
export function useAnimationClock(duration: number, key: unknown) {
  const [t, setT] = useState(() => (reducedMotion() ? duration : 0));
  const [playing, setPlaying] = useState(() => !reducedMotion());
  const start = useRef(0);
  const offset = useRef(0);

  useEffect(() => {
    offset.current = 0;
    if (reducedMotion()) {
      setT(duration);
      setPlaying(false);
    } else {
      setT(0);
      setPlaying(true);
    }
  }, [key, duration]);

  useEffect(() => {
    if (!playing) return undefined;
    let frame = 0;
    start.current = performance.now() - offset.current;
    const tick = (now: number) => {
      // a frame's timestamp can precede the performance.now() the clock started from
      const elapsed = Math.max(now - start.current, 0);
      if (elapsed >= duration) {
        setT(duration);
        offset.current = duration;
        setPlaying(false);
        return;
      }
      setT(elapsed);
      offset.current = elapsed;
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, duration]);

  return {
    t,
    playing,
    done: t >= duration,
    replay: () => { offset.current = 0; setT(0); setPlaying(true); },
    pause: () => setPlaying(false),
    resume: () => { if (t < duration) setPlaying(true); },
    skip: () => { offset.current = duration; setT(duration); setPlaying(false); },
    /** Jump to ``ms`` and pause there (a scrubber). */
    seek: (ms: number) => {
      const v = Math.min(Math.max(ms, 0), duration);
      offset.current = v;
      setT(v);
      setPlaying(false);
    },
  };
}
