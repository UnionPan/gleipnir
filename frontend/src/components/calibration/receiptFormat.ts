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

export interface ParameterDisplay {
  name: string;
  symbol: string;
  description: string;
}

const PARAMETERS: Record<string, ParameterDisplay> = {
  mu: { name: 'Drift', symbol: 'μ', description: 'Historical drift: educational, not simulated' },
  sigma: { name: 'Volatility', symbol: 'σ', description: 'Standard deviation of returns' },
  kappa: { name: 'Mean Reversion', symbol: 'κ', description: 'Speed of mean reversion' },
  theta: { name: 'Long-run Level', symbol: 'θ', description: 'Where the process reverts to' },
  xi: { name: 'Vol of Vol', symbol: 'ξ', description: 'Volatility of volatility' },
  sigma_v: { name: 'Vol of Vol', symbol: 'σᵥ', description: 'Volatility of variance' },
  rho: { name: 'Correlation', symbol: 'ρ', description: 'Correlation between processes' },
  v0: { name: 'Variance Today', symbol: 'v₀', description: 'Trailing 21-day realized variance' },
  lambda: { name: 'Jump Intensity', symbol: 'λ', description: 'Average jumps per unit time' },
  mu_j: { name: 'Jump Mean', symbol: 'μⱼ', description: 'Average jump size' },
  sigma_j: { name: 'Jump Volatility', symbol: 'σⱼ', description: 'Jump size volatility' },
  omega: { name: 'GARCH Constant', symbol: 'ω', description: 'Long-term variance level' },
  alpha: { name: 'ARCH Coefficient', symbol: 'α', description: 'Reaction to past shocks' },
  beta: { name: 'GARCH Coefficient', symbol: 'β', description: 'Persistence of volatility' },
  nu: { name: 'Vol of Vol', symbol: 'ν', description: 'Volatility of volatility' },
  xi0: { name: 'Forward Variance', symbol: 'ξ₀', description: 'Variance level today (trailing 21 days)' },
  eta: { name: 'Vol of Vol', symbol: 'η', description: 'Volatility of volatility' },
  H: { name: 'Hurst Parameter', symbol: 'H', description: 'Roughness parameter' },
  // what a Bayesian fit's network infers, where it differs from the model's own parameters
  long_run_vol: { name: 'Long-run Volatility', symbol: 'σ∞', description: 'Annualized, where volatility reverts to' },
  persistence: { name: 'Persistence', symbol: 'α+β', description: 'Share of a variance shock that survives a day' },
  alpha_share: { name: 'ARCH Share', symbol: 'α/(α+β)', description: 'How much of the persistence reacts to news' },
  window_variance: { name: 'Variance Level', symbol: 'ξ̄', description: "The window's average variance" },
};

const MODEL_PARAMETERS: Record<string, Record<string, ParameterDisplay>> = {
  garch: {
    nu: { name: 'Tail Index', symbol: 'ν', description: 'Student-t degrees of freedom of the shocks; lower is fatter-tailed' },
  },
};

const SUBSCRIPT = '₀₁₂₃₄₅₆₇₈₉';
const sub = (digits: string) => digits.replace(/\d/g, (d) => SUBSCRIPT[Number(d)]);

/** Name, symbol and description of a parameter as the receipt shows it. */
export function parameterDisplay(model: string, key: string): ParameterDisplay {
  const known = MODEL_PARAMETERS[model]?.[key] ?? PARAMETERS[key];
  if (known) return known;
  let m = key.match(/^sigma_regime(\d)$/);
  if (m) return { name: `Volatility, regime ${m[1]}`, symbol: `σ${sub(m[1])}`, description: 'Annualized volatility in this regime' };
  m = key.match(/^mu_regime(\d)$/);
  if (m) return { name: `Drift, regime ${m[1]}`, symbol: `μ${sub(m[1])}`, description: 'Educational, not simulated' };
  m = key.match(/^p_(\d)(\d)$/);
  if (m) {
    const stay = m[1] === m[2];
    return {
      name: stay ? `Stay in regime ${m[1]}` : `Regime ${m[1]} → ${m[2]}`,
      symbol: `p${sub(m[1] + m[2])}`,
      description: stay ? 'Daily probability of staying' : 'Daily probability of switching',
    };
  }
  return { name: key, symbol: key, description: '' };
}

/** How much the history narrowed a parameter beyond its prior. */
export function identificationLabel(contraction: number | null | undefined): string {
  if (contraction === null || contraction === undefined || !Number.isFinite(contraction)) return '—';
  if (contraction >= 0.7) return 'Well identified';
  if (contraction >= 0.3) return 'Partly identified';
  return 'Mostly the prior';
}

/** "0.912 to 1.31" */
export const intervalText = (low: number, high: number) => `${formatSig(low, 3)} to ${formatSig(high, 3)}`;

/** A summary statistic of the predictive check, readable at its scale. */
export const checkValue = (value: number) => formatSig(value, 3);

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
  turbulent_probability_today: { label: 'Turbulent today', format: (v) => percent(v, 0), hint: 'Filtered probability of the turbulent regime on the last day' },
  calm_duration_days: { label: 'Calm spells last', format: (v) => `${formatSig(v, 3)} trading days`, hint: 'Expected length of a calm regime' },
  turbulent_duration_days: { label: 'Turbulent spells last', format: (v) => `${formatSig(v, 3)} trading days`, hint: 'Expected length of a turbulent regime' },
  window_vol: { label: 'Volatility level', format: (v) => percent(v), hint: "The window's average, annualized" },
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
