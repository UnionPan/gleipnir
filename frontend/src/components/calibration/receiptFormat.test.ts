import { derivedRow, driftSentence, formatSig, identificationLabel, intervalText, parameterDisplay } from './receiptFormat';

test('significant figures keep tiny and large parameters readable', () => {
  expect(formatSig(1.338e-5)).toBe('1.338e-5');
  expect(formatSig(52.1284)).toBe('52.13');
  expect(formatSig(0.06951)).toBe('0.06951');
  expect(formatSig(-0.5211)).toBe('-0.5211');
  expect(formatSig(0)).toBe('0');
  expect(formatSig(null)).toBe('—');
  expect(formatSig(Number.POSITIVE_INFINITY)).toBe('—');
});

test('derived quantities carry their units', () => {
  expect(derivedRow('long_run_vol', 0.2763).value).toBe('27.6%');
  expect(derivedRow('half_life_days', 13.506).value).toBe('13.5 trading days');
  expect(derivedRow('jump_variance_share', 0.73).value).toBe('73%');
  expect(derivedRow('mystery_number', 2).label).toBe('mystery number');
});

test('the drift reads as an estimate with its interval', () => {
  expect(driftSentence({ estimate: 0.188, std_error: 0.124, ci_low: -0.055, ci_high: 0.431 }))
    .toBe('Historical drift 18.8% a year ± 12.4% (95% interval −5.5% to 43.1%).');
});

test('regime and GARCH parameters get readable names', () => {
  expect(parameterDisplay('regime_switching_gbm', 'sigma_regime1')).toMatchObject({ name: 'Volatility, regime 1', symbol: 'σ₁' });
  expect(parameterDisplay('regime_switching_gbm', 'p_00').name).toBe('Stay in regime 0');
  expect(parameterDisplay('regime_switching_gbm', 'p_01')).toMatchObject({ name: 'Regime 0 → 1', symbol: 'p₀₁' });
  expect(parameterDisplay('garch', 'nu').name).toBe('Tail Index');
  expect(parameterDisplay('sabr', 'nu').name).toBe('Vol of Vol');
});

test('identification reads in words', () => {
  expect(identificationLabel(0.91)).toBe('Well identified');
  expect(identificationLabel(0.45)).toBe('Partly identified');
  expect(identificationLabel(0.1)).toBe('Mostly the prior');
  expect(identificationLabel(null)).toBe('—');
  expect(intervalText(0.0912, 1.314)).toBe('0.0912 to 1.31');
});
