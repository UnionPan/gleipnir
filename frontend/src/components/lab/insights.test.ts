import type { HedgeResponse, HedgerStats, LabModelName, WorldResult } from '../../services/labApi';
import { readResult } from './insights';

const stats = (std: number, mean = 0): HedgerStats => ({
  mean, std, cvar_5: mean - 2 * std, mean_cost: 0.1, histogram: [], below: 0, above: 0,
});

const MODEL_HEDGER: Record<string, [string, string]> = {
  heston: ['heston_delta', 'Heston delta'],
  gbm: ['gbm_delta', 'Black-Scholes at historical vol'],
};

const world = (label: string, s: [number, number, number], mean = 0, model = 'heston'): WorldResult => {
  const m = MODEL_HEDGER[model][0];
  return {
    label,
    description: '',
    params: {},
    params_hash: label,
    hedgers: { no_hedge: stats(s[0], mean), bs_delta: stats(s[1], mean), [m]: stats(s[2], mean) },
    sample_path: { spots: [], positions: { no_hedge: [], bs_delta: [], [m]: [] }, pnl: { no_hedge: 0, bs_delta: 0, [m]: 0 } },
  };
};

const response = (counterfactual: WorldResult | null, maturityDays = 21, model: LabModelName = 'heston',
  factual: WorldResult = world('Factual', [7, 1, 1])): HedgeResponse => ({
  book: {
    asset: 'TEST', model, params: {}, params_hash: 'b', parent_hash: null, rho_source: 'assumed',
    origin: {
      kind: 'shipped', window: '2021..2026', as_of: '2026-01-02', vol_today: 0.3, historical_drift: null,
      method_label: '', note: null, calibration_hash: 'c',
    },
    strike: 100, maturity_days: maturityDays, is_call: true, premium: 4, implied_vol: 0.3, rate: 0.04,
    dividend_yield: 0.01,
  },
  hedgers: [
    { name: 'no_hedge', role: 'none', label: 'No hedge', note: '' },
    { name: 'bs_delta', role: 'bs', label: 'Black-Scholes delta', note: '' },
    { name: MODEL_HEDGER[model][0], role: 'model', label: MODEL_HEDGER[model][1], note: '',
      same_as: model === 'gbm' ? 'bs_delta' : null },
  ],
  bin_edges: {},
  factual,
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

test('under Black-Scholes the model hedge and Black-Scholes delta are one hedge', () => {
  const calm = world('Factual', [7, 1, 1], 0, 'gbm');
  const spiked = world('Volatility spike', [14, 3, 3], 0, 'gbm');
  const insight = readResult(response(spiked, 21, 'gbm', calm), 'vol_spike');
  expect(insight.headline).toBe("In a volatility spike, delta hedging's error grows 3.0x.");
  expect(insight.detail).toContain('are the same hedge');
  expect(readResult(response(null, 21, 'gbm', calm), null).detail).toContain("the model's own delta is the same hedge");
});

test('near-equal errors read as a tie', () => {
  const insight = readResult(response(world('Volatility spike', [14, 3, 3.05], 0)), 'vol_spike');
  expect(insight.detail).toContain('the two hedges end up with about the same error (3.00)');
});

test('the learned band reports its cost saving and its error against the model delta', () => {
  const r = response(null);
  r.hedgers.push({ name: 'learned_band', role: 'learned', label: 'Learned band', note: '' });
  r.factual.hedgers.learned_band = { ...stats(1.1), mean_cost: 0.06 };
  r.factual.hedgers.heston_delta = { ...stats(1), mean_cost: 0.1 };
  expect(readResult(r, null).detail).toContain(
    'The learned band spends 0.060 on trading against 0.100 (40% less), for an error 10% wider than the model\'s delta.');
  expect(readResult(response(null), null).detail).not.toContain('learned band');
});
