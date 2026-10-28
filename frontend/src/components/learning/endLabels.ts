import type { Annotations } from 'plotly.js';

export interface LabeledSeries {
  name: string;
  x: number[];
  y: number[];
  color: string;
}

const MIN_GAP = 0.08; // of the y range: about one label height on a 200-250 px chart

/**
 * Direct labels at the right end of each line, moved apart where they would
 * overlap, and the x-range that keeps the axis on the data (Plotly's autorange
 * would otherwise stretch to make room for the labels; they sit in the right
 * margin instead). On a log axis, annotations take log10 coordinates.
 */
export function endLabels(series: LabeledSeries[], { logY = false } = {}): {
  annotations: Partial<Annotations>[];
  xRange: [number, number];
} {
  const t = (v: number) => (logY ? Math.log10(v) : v);
  const all = series.flatMap((s) => s.y.filter(Number.isFinite).map(t));
  const gap = MIN_GAP * Math.max(Math.max(...all) - Math.min(...all), 1e-9);
  const ends = series
    .map((s) => ({ s, y: t(s.y[s.y.length - 1]) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) {
    if (ends[i].y - ends[i - 1].y < gap) ends[i].y = ends[i - 1].y + gap;
  }
  const xs = series.flatMap((s) => s.x);
  return {
    annotations: ends.map(({ s, y }) => ({
      x: s.x[s.x.length - 1], y, text: s.name, showarrow: false, xanchor: 'left', xshift: 6,
      font: { color: s.color, size: 11 },
    })),
    xRange: [Math.min(...xs), Math.max(...xs)],
  };
}
