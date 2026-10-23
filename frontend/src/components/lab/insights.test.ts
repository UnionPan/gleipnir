import type { HedgeResponse, HedgerStats, WorldResult } from '../../services/labApi';
import { readResult } from './insights';

const stats = (std: number, mean = 0): HedgerStats => ({
  mean, std, cvar_5: mean - 2 * std, mean_cost: 0.1, histogram: [], below: 0, above: 0,
});

const world = (label: string, s: [number, number, number], mean = 0): WorldResult => ({
  label,
  description: '',
  params: {},
  params_hash: label,
  hedgers: { no_hedge: stats(s[0], mean), bs_delta: stats(s[1], mean), heston_delta: stats(s[2], mean) },
  sample_path: { spots: [], positions: { no_hedge: [], bs_delta: [], heston_delta: [] }, pnl: { no_hedge: 0, bs_delta: 0, heston_delta: 0 } },
});

const response = (counterfactual: WorldResult | null, maturityDays = 21): HedgeResponse => ({
  book: {
    asset: 'TEST', params: {}, params_hash: 'b', parent_hash: null, rho_source: 'assumed',
    strike: 100, maturity_days: maturityDays, is_call: true, premium: 4, implied_vol: 0.3,
  },
  bin_edges: { no_hedge: [], bs_delta: [], heston_delta: [] },
  factual: world('Factual', [7, 1, 1]),
  counterfactual,
  n_paths: 4096, seed: 7, cost_bps: 5, elapsed_ms: 500, mjollnir_version: 'test',
});

test('factual-only run reports how much risk delta hedging removes', () => {
  const insight = readResult(response(null), null);
  expect(insight.headline).toBe('Delta hedging removes 86% of the unhedged risk.');
});

test('a damaging world reports each hedger’s growth and the average loss', () => {
  const insight = readResult(response(world('Volatility spike', [14, 3, 1.8], -4)), 'vol_spike');
  expect(insight.headline).toBe(
    "In a volatility spike, Black-Scholes delta's hedging error grows 3.0x; Heston delta's grows 1.8x.");
  expect(insight.detail).toContain('Heston delta ends up with the smaller error (1.80 vs 3.00)');
  expect(insight.detail).toContain('loses about 4.00 on average');
});

test('a harmless world suggests a longer expiry only when one exists', () => {
  const quiet = world('Regime shift', [7, 1.05, 1.02]);
  expect(readResult(response(quiet, 21), 'regime_shift').detail).toContain('Try a longer expiry');
  const longest = readResult(response(quiet, 63), 'regime_shift');
  expect(longest.headline).toBe('A regime shift barely moves this 63-day book.');
  expect(longest.detail).not.toContain('longer expiry');
});
