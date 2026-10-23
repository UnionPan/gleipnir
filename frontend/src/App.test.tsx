import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';
import { ThemeProvider } from './contexts/ThemeContext';

jest.mock('react-plotly.js', () => () => null);
jest.mock('./services/marketApi', () => ({
  marketApi: {
    getMarketOverview: () => new Promise(() => {}),
    getIndexCharts: () => new Promise(() => {}),
    getOHLCData: () => new Promise(() => {}),
  },
}));

beforeAll(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
    }),
  });
});

test('renders the Gleipnir navigation with the hedging lab', () => {
  render(
    <ThemeProvider>
      <App />
    </ThemeProvider>,
  );
  expect(screen.getByRole('button', { name: 'Gleipnir home' })).toHaveTextContent('Gleipnir');
  for (const tab of ['Market', 'Calibration', 'Hedging Lab']) {
    expect(screen.getByRole('button', { name: tab })).toBeInTheDocument();
  }
  expect(screen.queryByText(/backtesting/i)).not.toBeInTheDocument();
});
