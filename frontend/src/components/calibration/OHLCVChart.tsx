import React from 'react';
import Plot from 'react-plotly.js';
import type { Data } from 'plotly.js';
import type { OHLCVData } from '../../services/calibrationApi';
import { baseLayout, chrome, PLOT_CONFIG, ThemeName } from '../lab/chartTheme';

interface Props {
  data: OHLCVData;
  theme: ThemeName;
}

/**
 * The window the model was fitted to, as daily candles over volume. Candles are
 * monochrome (hollow up, filled down): blue on this page means the model.
 */
const OHLCVChart: React.FC<Props> = ({ data, theme }) => {
  const c = chrome(theme);
  const traces: Data[] = [
    {
      type: 'candlestick', x: data.dates, open: data.open, high: data.high, low: data.low, close: data.close,
      name: data.ticker, yaxis: 'y',
      increasing: { line: { color: c.inkSecondary, width: 1 }, fillcolor: c.surface },
      decreasing: { line: { color: c.inkSecondary, width: 1 }, fillcolor: c.inkSecondary },
    } as Data,
    {
      type: 'bar', x: data.dates, y: data.volume, name: 'Volume', yaxis: 'y2', marker: { color: c.baseline },
      hovertemplate: '%{y:,.0f} shares<extra></extra>',
    },
  ];
  return (
    <Plot
      data={traces}
      layout={baseLayout(theme, {
        height: 380, showlegend: false, hovermode: 'x unified', margin: { l: 56, r: 12, t: 8, b: 32 },
        xaxis: { type: 'date', rangeslider: { visible: false } },
        yaxis: { title: { text: 'Price' }, domain: [0.28, 1] },
        yaxis2: { title: { text: 'Volume' }, domain: [0, 0.2] },
      })}
      config={PLOT_CONFIG}
      useResizeHandler
      style={{ width: '100%' }}
    />
  );
};

export default OHLCVChart;
