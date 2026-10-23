/**
 * Chart colors and chrome for the Hedging Lab.
 *
 * Color carries exactly one meaning on every lab chart: which world a mark
 * belongs to. The pair is categorical slots 1-2 of the dataviz reference
 * palette, validated (all-pairs CVD and contrast) against the site's chart
 * surface, the "well" background in index.css, in both modes. Plotly cannot
 * read CSS variables, so the surface and ink values mirror index.css.
 */
import type { Layout } from 'plotly.js';

export type ThemeName = 'light' | 'dark';
export type World = 'factual' | 'counterfactual';

interface Chrome {
  surface: string;
  ink: string;
  inkSecondary: string;
  muted: string;
  grid: string;
  baseline: string;
  world: Record<World, string>;
}

const CHROME: Record<ThemeName, Chrome> = {
  light: {
    surface: '#ffffff',
    ink: '#000000',
    inkSecondary: '#333333',
    muted: '#666666',
    grid: '#e1e0d9',
    baseline: '#a0a0a0',
    world: { factual: '#2a78d6', counterfactual: '#eb6834' },
  },
  dark: {
    surface: '#16162a',
    ink: '#e0e0e8',
    inkSecondary: '#a0a0b4',
    muted: '#6a6a82',
    grid: '#26263c',
    baseline: '#3a3a50',
    world: { factual: '#3987e5', counterfactual: '#d95926' },
  },
};

export function chrome(theme: ThemeName): Chrome {
  return CHROME[theme];
}

/** A hex color at ~10% opacity, for area washes. */
export function wash(hex: string): string {
  return `${hex}1a`;
}

export function baseLayout(theme: ThemeName, overrides: Partial<Layout> = {}): Partial<Layout> {
  const c = CHROME[theme];
  const axis = {
    gridcolor: c.grid,
    linecolor: c.baseline,
    zerolinecolor: c.baseline,
    tickfont: { color: c.muted, size: 11 },
    title: { font: { color: c.inkSecondary, size: 11 } },
    automargin: true,
  };
  return {
    paper_bgcolor: c.surface,
    plot_bgcolor: c.surface,
    font: { family: 'Inter, system-ui, sans-serif', color: c.ink, size: 12 },
    margin: { l: 48, r: 12, t: 8, b: 40 },
    showlegend: false,
    hovermode: 'x unified',
    hoverlabel: { bgcolor: c.surface, bordercolor: c.baseline, font: { color: c.ink, size: 12 } },
    ...overrides,
    xaxis: { ...axis, ...(overrides.xaxis ?? {}) },
    yaxis: { ...axis, ...(overrides.yaxis ?? {}) },
  };
}

export const PLOT_CONFIG = { displayModeBar: false, responsive: true } as const;
