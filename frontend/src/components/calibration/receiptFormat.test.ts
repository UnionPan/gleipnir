import { derivedRow, driftSentence, formatSig } from './receiptFormat';

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
