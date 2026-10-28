import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { Data } from 'plotly.js';
import type { OptionChainData, VolSurfaceResponse } from '../../services/calibrationApi';
import { chrome, ThemeName } from '../lab/chartTheme';

interface Props {
  surface: VolSurfaceResponse;
  chain: OptionChainData;
  theme: ThemeName;
}

// One hue, light to dark (dark to bright on the dark surface): the height is a magnitude.
const RAMP: Record<ThemeName, [number, string][]> = {
  light: [[0, '#dbe9fa'], [0.5, '#5b9be6'], [1, '#0d3a73']],
  dark: [[0, '#1b2b4a'], [0.5, '#3987e5'], [1, '#bcd9fb']],
};

const YEAR_MS = 365 * 24 * 3600 * 1000;

/** The quotes a fit would keep: liquid, out of the money, near the money, with a sane implied vol. */
export function fittedQuotes(chain: OptionChainData, maxYears: number) {
  const today = new Date(`${chain.reference_date}T00:00:00Z`).getTime();
  return chain.options
    .map((o) => ({ ...o, years: (new Date(`${o.expiry}T00:00:00Z`).getTime() - today) / YEAR_MS }))
    .filter((o) => typeof o.implied_volatility === 'number' && o.implied_volatility > 0.01 && o.implied_volatility < 3
      && o.volume >= 10 && o.open_interest >= 50 && o.mid > 0 && (o.ask - o.bid) <= 0.5 * o.mid
      && o.moneyness >= 0.8 && o.moneyness <= 1.2 && o.years > 0 && o.years <= maxYears
      && (o.option_type === 'call' ? o.moneyness >= 1 : o.moneyness <= 1));
}

/** The fitted model's volatility over strike and expiry, with the market's own quotes as dots. */
const VolatilitySurface3D: React.FC<Props> = ({ surface, chain, theme }) => {
  const c = chrome(theme);
  const implied = surface.surface_type === 'implied_vol';
  const maxYears = Math.max(...surface.maturities);
  const quotes = useMemo(() => fittedQuotes(chain, maxYears), [chain, maxYears]);
  const axis = (title: string, extra: object = {}) => ({
    title: { text: title, font: { color: c.inkSecondary, size: 11 } },
    gridcolor: c.grid, zerolinecolor: c.baseline, tickfont: { color: c.muted, size: 10 }, backgroundcolor: c.surface,
    ...extra,
  });
  const data: Data[] = [
    {
      type: 'surface', x: surface.strikes.map((k) => k / chain.spot_price), y: surface.maturities, z: surface.vols,
      colorscale: RAMP[theme], opacity: 0.92, showscale: false,
      hovertemplate: 'strike %{x:.0%} of spot<br>%{y:.2f} years<br>model vol %{z:.1%}<extra></extra>',
    } as Data,
    ...(implied && quotes.length > 0 ? [{
      type: 'scatter3d', mode: 'markers', x: quotes.map((o) => o.moneyness), y: quotes.map((o) => o.years),
      z: quotes.map((o) => o.implied_volatility as number), marker: { size: 2.5, color: c.ink },
      hovertemplate: 'market quote<br>strike %{x:.0%} of spot<br>%{y:.2f} years<br>implied vol %{z:.1%}<extra></extra>',
    } as Data] : []),
  ];
  return (
    <figure className="lab-chart calib-surface">
      <figcaption>
        <span className="lab-chart-title">{implied ? 'Implied volatility' : 'Local volatility'} by strike and expiry</span>
        <span className="lab-hero-count">drag to turn</span>
      </figcaption>
      <Plot
        data={data}
        layout={{
          height: 460, margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: c.surface, showlegend: false,
          font: { family: 'Inter, system-ui, sans-serif', color: c.ink, size: 11 },
          hoverlabel: { bgcolor: c.surface, bordercolor: c.baseline, font: { color: c.ink, size: 12 } },
          scene: {
            xaxis: axis('Strike / spot', { tickformat: '.0%' }),
            yaxis: axis('Years to expiry'),
            zaxis: axis(implied ? 'Implied vol' : 'Local vol', { tickformat: '.0%' }),
            camera: { eye: { x: 1.25, y: -1.3, z: 0.62 } },
            bgcolor: c.surface,
          },
        }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: '100%' }}
      />
      <p className="lab-hero-note">
        {implied
          ? <>The surface is the volatility the fitted model implies for each strike and expiry; the dots are{' '}
            {quotes.length.toLocaleString('en-US')} liquid out-of-the-money market quotes. Where the dots leave the
            surface, the model misprices.</>
          : <>The local volatility the option prices imply at each strike and time: the one volatility function under
            which a diffusion reprices every quote.</>}
      </p>
    </figure>
  );
};

export default VolatilitySurface3D;
