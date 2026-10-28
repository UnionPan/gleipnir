import type { DeepHedgingDemo, HedgerKey } from '../../services/rlApi';

const CLASSICAL = ['bs_delta', 'delta_gamma', 'delta_gamma_vega'] as const;

/** The one-line verdict: how the learned hedger with options compares with the best classical hedge. */
export function headline(demo: DeepHedgingDemo, label: Record<HedgerKey, string>): string {
  const r = demo.results;
  const best = CLASSICAL.reduce((a, k) => (r[k].std_error < r[a].std_error ? k : a), CLASSICAL[0]);
  const options = r.lstm_options.std_error;
  if (options.mean + 2 * options.sd < r[best].std_error) return 'With listed options, the learned hedger beats every classical hedge.';
  if (options.mean - 2 * options.sd > r[best].std_error) return `No learned hedger beats the best classical hedge, ${label[best]}.`;
  return `With listed options, the learned hedger matches the best classical hedge, ${label[best]}, within its seeds' spread.`;
}

/** What these results say, in sentences computed from them (they stay true for any run). */
export function verdict(demo: DeepHedgingDemo, label: Record<HedgerKey, string>): string[] {
  const r = demo.results;
  const f = (x: number) => x.toFixed(2);
  const best = CLASSICAL.reduce((a, k) => (r[k].std_error < r[a].std_error ? k : a), CLASSICAL[0]);
  const bestStd = r[best].std_error;
  const under = r.lstm_underlying.std_error;
  const options = r.lstm_options.std_error;
  const out = [
    `Trading only the underlying, the LSTM ends with an error spread of ${f(under.mean)} ± ${f(under.sd)} across seeds, `
    + `against ${f(r.bs_delta.std_error)} for Black-Scholes delta. With the listed options it reaches ${f(options.mean)} `
    + `± ${f(options.sd)}, spending ${f(r.lstm_options.mean_cost.mean)} on trading against delta's ${f(r.bs_delta.mean_cost)}.`,
  ];
  if (options.mean + 2 * options.sd < bestStd) {
    out.push(`That beats every classical hedge here; the best, ${label[best]}, ends at ${f(bestStd)}.`);
  } else if (options.mean - 2 * options.sd > bestStd) {
    out.push(`The best classical hedge, ${label[best]}, ends at ${f(bestStd)}: no learned hedger beats it here.`);
  } else {
    out.push(`The best classical hedge, ${label[best]}, ends at ${f(bestStd)}, within the seeds' spread of the learned one.`);
  }
  if (r.delta_gamma_vega.std_error > r.bs_delta.std_error) {
    out.push(`Delta-gamma-vega ends worse than delta alone (${f(r.delta_gamma_vega.std_error)}): offsetting a long-dated `
      + `option's vega with short-dated ones takes positions whose risk and costs (${f(r.delta_gamma_vega.mean_cost)}) `
      + 'outweigh what they hedge.');
  }
  return out;
}

/**
 * What the desk is paid for the call: its true-model value on day 0. The demo's
 * hedgers start with no cash, so every book the page draws is credited this
 * premium and starts at 0 (0 for an export without it).
 */
export function premiumOf(demo: DeepHedgingDemo): number {
  return (demo.setup.fair_premium ?? 0) - (demo.setup.initial_cash ?? 0);
}
