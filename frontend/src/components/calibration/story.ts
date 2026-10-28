/** What a calibration says, in words: the headline and the few numbers worth reading first. */
import type { CalibrationResponse } from '../../services/calibrationApi';
import { derivedRow, formatSig, hasDerived, parameterDisplay } from './receiptFormat';

const MODEL_LABELS: Record<string, string> = {
  gbm: 'Black-Scholes (GBM)',
  ou: 'Ornstein-Uhlenbeck',
  heston: 'Heston',
  rough_bergomi: 'Rough Bergomi',
  regime_switching_gbm: 'Regime-switching GBM',
  merton_jump: 'Merton jump-diffusion',
  garch: 'GARCH(1,1)',
  bates: 'Bates',
  sabr: 'SABR',
  dupire: 'Dupire local volatility',
};

export const modelLabel = (model: string) => MODEL_LABELS[model] ?? model.toUpperCase();

/** "a", "a and b", "a, b and c" */
export function joinAnd(items: string[]): string {
  if (items.length <= 1) return items.join('');
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

/** The features of the history that no simulated history matched, lower-cased for a sentence. */
export function missedFeatures(result: CalibrationResponse): string[] {
  return (result.diagnostics.predictive_check?.aspects ?? [])
    .filter((a) => a.verdict !== 'fits')
    .map((a) => a.aspect.toLowerCase());
}

/** The fit's verdict in one sentence. */
export function headline(result: CalibrationResponse): string {
  const model = modelLabel(result.model);
  const { ticker } = result;
  const aspects = result.diagnostics.predictive_check?.aspects ?? [];
  if (aspects.length > 0) {
    const missed = missedFeatures(result);
    if (missed.length === 0) return `${model} reproduces all ${aspects.length} checked features of ${ticker}'s history.`;
    return `${model} reproduces ${aspects.length - missed.length} of ${aspects.length} features of ${ticker}'s history, `
      + `missing its ${joinAnd(missed)}.`;
  }
  const { rmse_iv: rmse, n_quotes: quotes } = result.diagnostics;
  if (typeof rmse === 'number') {
    const count = typeof quotes === 'number' ? `${quotes.toLocaleString('en-US')} ` : '';
    return `${model} prices ${ticker}'s ${count}option quotes to within ${(100 * rmse).toFixed(1)} vol points.`;
  }
  return `${model} fitted to ${ticker}'s ${result.measure === 'Q-measure' ? 'option prices' : 'daily prices'}.`;
}

export interface KeyNumber {
  key: string;
  label: string;
  value: string;
  hint: string;
}

const MAX_KEY_NUMBERS = 4;
// Read-outs that need the parameter table beside them to mean anything.
const TECHNICAL = new Set(['persistence', 'feller_ratio', 'stationarity_pvalue', 'variance_proxy_r2', 'noise_floor', 'regime_today']);

const TRADING_DAYS = 252;

/** Readable quantities a fit without read-outs still implies: a Heston fit's volatilities and half-life. */
function readable(parameters: Record<string, number>): Record<string, number> | null {
  const { kappa, theta, v0 } = parameters;
  if ([kappa, theta, v0].every((v) => typeof v === 'number' && v > 0)) {
    return { vol_today: Math.sqrt(v0), long_run_vol: Math.sqrt(theta), half_life_days: (Math.LN2 / kappa) * TRADING_DAYS };
  }
  return null;
}

/**
 * The quantities a visitor can read without the parameter chart: volatility
 * levels, half-lives, spell lengths. Models without such read-outs show their
 * parameters; the historical drift is never one of them (it is not simulated).
 */
export function keyNumbers(result: CalibrationResponse): KeyNumber[] {
  const derived = Object.entries(result.diagnostics.derived ?? readable(result.parameters) ?? {})
    .filter((e): e is [string, number] => typeof e[1] === 'number' && Number.isFinite(e[1]) && !TECHNICAL.has(e[0]))
    .map(([key, value]) => ({ key, ...derivedRow(key, value) }));
  if (derived.length > 0) return derived.slice(0, MAX_KEY_NUMBERS);
  return Object.entries(result.parameters)
    .filter(([key]) => key !== 'mu' && !key.startsWith('mu_regime'))
    .slice(0, MAX_KEY_NUMBERS)
    .map(([key, value]) => {
      if (hasDerived(key)) return { key, ...derivedRow(key, value) };
      const info = parameterDisplay(result.model, key);
      return { key, label: `${info.name} ${info.symbol}`, value: formatSig(value), hint: info.description };
    });
}
