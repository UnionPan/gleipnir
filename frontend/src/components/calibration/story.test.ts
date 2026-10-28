import type { CalibrationResponse, CheckedAspect } from '../../services/calibrationApi';
import { headline, joinAnd, keyNumbers } from './story';

const aspect = (name: string, verdict: CheckedAspect['verdict']): CheckedAspect => ({
  aspect: name, verdict, summary: '', observed: 0, q05: 0, q50: 0, q95: 0, p_value: 0.5,
});

const calibrated = (overrides: Partial<CalibrationResponse> = {}): CalibrationResponse => ({
  ticker: 'XOM',
  model: 'heston',
  parameters: { kappa: 13.4, theta: 0.0747, xi: 0.81, rho: -0.18, v0: 0.047, mu: 0.24 },
  diagnostics: {},
  timestamp: '2026-10-28T13:10:00Z',
  ...overrides,
});

test('lists read as a sentence', () => {
  expect(joinAnd([])).toBe('');
  expect(joinAnd(['fat tails'])).toBe('fat tails');
  expect(joinAnd(['fat tails', 'roughness'])).toBe('fat tails and roughness');
  expect(joinAnd(['skewness', 'fat tails', 'roughness'])).toBe('skewness, fat tails and roughness');
});

test('the headline is the predictive check when there is one', () => {
  const all = calibrated({ diagnostics: { predictive_check: { n_replicates: 64, aspects: [aspect('Skewness', 'fits'), aspect('Fat tails', 'fits')] } } });
  expect(headline(all)).toBe('Heston reproduces all 2 checked features of XOM\'s history.');
  const some = calibrated({
    diagnostics: {
      predictive_check: {
        n_replicates: 64,
        aspects: [aspect('Volatility level', 'fits'), aspect('Fat tails', 'more'), aspect('Roughness', 'less')],
      },
    },
  });
  expect(headline(some)).toBe('Heston reproduces 1 of 3 features of XOM\'s history, missing its fat tails and roughness.');
});

test('an option-price fit leads with its pricing error', () => {
  const q = calibrated({ ticker: 'AAPL', measure: 'Q-measure', diagnostics: { rmse_iv: 0.0132, n_quotes: 412 } });
  expect(headline(q)).toBe('Heston prices AAPL\'s 412 option quotes to within 1.3 vol points.');
  expect(headline(calibrated({ model: 'ou' }))).toBe('Ornstein-Uhlenbeck fitted to XOM\'s daily prices.');
  const sabr = calibrated({ ticker: 'AAPL', model: 'sabr', measure: 'Q-measure', diagnostics: { rmse_iv: 0.007 } });
  expect(headline(sabr)).toBe('SABR prices AAPL\'s option quotes to within 0.7 vol points.');
});

test('key numbers prefer readable quantities and never show the drift', () => {
  const heston = calibrated({ diagnostics: { derived: { long_run_vol: 0.273, vol_today: 0.217, half_life_days: 13.1, feller_ratio: 3.03 } } });
  expect(keyNumbers(heston).map((k) => k.value)).toEqual(['27.3%', '21.7%', '13.1 trading days']);
  const gbm = calibrated({ model: 'gbm', parameters: { sigma: 0.1715, mu: 0.12 }, diagnostics: { derived: null } });
  expect(keyNumbers(gbm)).toEqual([expect.objectContaining({ key: 'sigma', value: '0.1715' })]);
});

test('an option-price fit reads as volatilities, not raw parameters', () => {
  const q = calibrated({ measure: 'Q-measure', parameters: { kappa: 2.0, theta: 0.0625, xi: 0.5, rho: -0.4, v0: 0.04 } });
  expect(keyNumbers(q).map((k) => [k.key, k.value])).toEqual([
    ['vol_today', '20.0%'], ['long_run_vol', '25.0%'], ['half_life_days', '87.3 trading days'],
  ]);
  const dupire = calibrated({ model: 'dupire', parameters: { min_local_vol: 0.113, atm_local_vol: 0.2246 } });
  expect(keyNumbers(dupire).map((k) => k.value)).toEqual(['11.3%', '22.5%']);
});
