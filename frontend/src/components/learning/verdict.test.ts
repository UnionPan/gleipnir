import { headline, verdict } from './verdict';
import type { DeepHedgingDemo, HedgerKey } from '../../services/rlApi';

const stat = (mean: number, sd = 0.05) => ({ mean, sd, values: [mean] });
const classical = (std: number, cost: number) => ({
  mean_error: -7.7, std_error: std, expected_shortfall_95: 2 * std, mean_cost: cost, option_turnover: 0,
});
const learned = (std: number, cost: number) => ({
  mean_error: stat(-7.7), std_error: stat(std), expected_shortfall_95: stat(2 * std), mean_cost: stat(cost),
  option_turnover: stat(0), seeds: [1], representative_seed: 1,
});
const label = {
  bs_delta: 'Black-Scholes delta', delta_gamma: 'Delta-gamma', delta_gamma_vega: 'Delta-gamma-vega',
  lstm_underlying: 'LSTM, underlying only', lstm_options: 'LSTM with options',
} as Record<HedgerKey, string>;

function demo(lstmOptions: number, vega: number): DeepHedgingDemo {
  return {
    results: {
      bs_delta: classical(1.7, 0.06), delta_gamma: classical(1.62, 0.7), delta_gamma_vega: classical(vega, 5),
      lstm_underlying: learned(2.1, 0.03), lstm_options: learned(lstmOptions, 0.8),
    },
  } as unknown as DeepHedgingDemo;
}

test('the verdict says no learned hedger wins when the best classical hedge is clearly better', () => {
  const text = verdict(demo(1.9, 2.7), label).join(' ');
  expect(text).toContain('The best classical hedge, Delta-gamma, ends at 1.62: no learned hedger beats it here.');
  expect(text).toContain('Delta-gamma-vega ends worse than delta alone (2.70)');
});

test('the verdict credits the learned hedger only when it is clearly better', () => {
  expect(verdict(demo(1.2, 1.5), label).join(' ')).toContain('That beats every classical hedge here');
  expect(verdict(demo(1.65, 2.7), label).join(' ')).toContain("within the seeds' spread");
});

test('the headline names the best classical hedge and whether the learner beats it', () => {
  expect(headline(demo(1.9, 2.7), label)).toBe('No learned hedger beats the best classical hedge, Delta-gamma.');
  expect(headline(demo(1.2, 2.7), label)).toBe('With listed options, the learned hedger beats every classical hedge.');
  expect(headline(demo(1.65, 2.7), label)).toContain('matches the best classical hedge, Delta-gamma');
});
