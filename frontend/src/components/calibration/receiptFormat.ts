/** Formatting for the calibration receipt. */

/**
 * A number to ``digits`` significant figures: GARCH's omega (1.3e-5) and a
 * mean-reversion speed (52) both stay readable; null and non-finite read as —.
 */
export function formatSig(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e-3 && magnitude < 1e6) {
    const decimals = Math.max(0, digits - 1 - Math.floor(Math.log10(magnitude)));
    return value.toFixed(Math.min(decimals, 8));
  }
  return value.toExponential(digits - 1);
}

export const percent = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;

const signed = (value: number, digits = 1) => `${value < 0 ? '−' : ''}${percent(Math.abs(value), digits)}`;

interface Derived {
  label: string;
  format: (value: number) => string;
  hint: string;
}

const DERIVED: Record<string, Derived> = {
  long_run_vol: { label: 'Long-run volatility', format: (v) => percent(v), hint: 'Annualized, where volatility reverts to' },
  vol_today: { label: 'Volatility today', format: (v) => percent(v), hint: 'Trailing 21-day realized, annualized' },
  total_vol: { label: 'Total volatility', format: (v) => percent(v), hint: 'Diffusion and jumps together, annualized' },
  half_life_days: { label: 'Half-life', format: (v) => `${formatSig(v, 3)} trading days`, hint: 'Time for a shock to halve' },
  persistence: { label: 'Persistence α + β', format: (v) => v.toFixed(3), hint: 'Share of a variance shock that survives a day' },
  jump_variance_share: { label: 'Jump share of variance', format: (v) => percent(v, 0), hint: 'Better identified than jump rate and size separately' },
  long_run_level: { label: 'Long-run price level', format: (v) => formatSig(v, 5), hint: 'Where the log price reverts to' },
  stationarity_pvalue: { label: 'Stationarity p-value', format: (v) => v.toFixed(3), hint: 'Augmented Dickey-Fuller; small means mean reversion is real' },
  feller_ratio: { label: 'Feller ratio 2κθ/ξ²', format: (v) => v.toFixed(2), hint: 'Above 1 keeps variance off zero' },
  variance_proxy_r2: { label: 'Variance proxy R²', format: (v) => v.toFixed(3), hint: 'How well the AR(1) explains the realized-variance proxy' },
  noise_floor: { label: 'Variogram noise floor', format: (v) => v.toFixed(2), hint: 'π²/4 ≈ 2.47 for Gaussian daily shocks' },
  regime_today: { label: 'Regime today', format: (v) => `${v.toFixed(0)}`, hint: 'The regime the window ends in (numbered as in the parameters)' },
};

export function derivedRow(key: string, value: number): { label: string; value: string; hint: string } {
  const spec = DERIVED[key];
  return spec
    ? { label: spec.label, value: spec.format(value), hint: spec.hint }
    : { label: key.replace(/_/g, ' '), value: formatSig(value), hint: '' };
}

/** "Historical drift 18.8% a year ± 12.4% (95% interval −5.5% to 43.1%)" */
export function driftSentence(drift: { estimate: number; std_error: number; ci_low: number; ci_high: number }): string {
  return `Historical drift ${signed(drift.estimate)} a year ± ${percent(drift.std_error)} `
    + `(95% interval ${signed(drift.ci_low)} to ${signed(drift.ci_high)}).`;
}
