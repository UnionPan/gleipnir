/**
 * Plain-language readings of a lab result. Pure functions of the response so
 * the narrative stays consistent with the numbers on the page.
 */
import type { HedgeResponse, HedgerName, ScenarioName } from '../../services/labApi';

export const HEDGER_LABELS: Record<HedgerName, string> = {
  no_hedge: 'No hedge',
  bs_delta: 'Black-Scholes delta',
  heston_delta: 'Heston delta',
};

export const HEDGER_NOTES: Record<HedgerName, string> = {
  no_hedge: 'Keeps the premium and pays the payoff.',
  bs_delta: 'Delta at the implied vol of the premium, fixed for the life of the option.',
  heston_delta: 'Model delta at the current variance, so it adapts as volatility moves.',
};

export const HEDGER_ORDER: HedgerName[] = ['no_hedge', 'bs_delta', 'heston_delta'];

/** How each counterfactual acts, and the horizon it needs to matter. */
const MECHANISM: Record<ScenarioName, { how: string; horizon: string }> = {
  vol_spike: {
    how: 'it raises volatility immediately',
    horizon: 'shows up at every maturity',
  },
  correlation_breakdown: {
    how: 'it changes how volatility co-moves with the price, which bends the smile',
    horizon: 'matters most for out-of-the-money options and longer maturities',
  },
  regime_shift: {
    how: 'it slows mean reversion and lifts long-run variance',
    horizon: 'needs time to accumulate, so it matters most for longer maturities',
  },
};

const LONGEST_MATURITY_DAYS = 63;

const ratio = (a: number, b: number) => (b > 0 ? a / b : Number.NaN);
const fmt = (x: number, digits = 2) => x.toFixed(digits);

export interface Insight {
  headline: string;
  detail: string;
}

export function readResult(result: HedgeResponse, scenario: ScenarioName | null): Insight {
  const f = result.factual.hedgers;
  const reduction = 1 - ratio(f.bs_delta.std, f.no_hedge.std);
  if (!result.counterfactual || !scenario) {
    return {
      headline: `Delta hedging removes ${fmt(reduction * 100, 0)}% of the unhedged risk.`,
      detail:
        `Unhedged, the book's P&L has a standard deviation of ${fmt(f.no_hedge.std)}. ` +
        `Black-Scholes delta brings it to ${fmt(f.bs_delta.std)} and Heston delta to ${fmt(f.heston_delta.std)}. ` +
        'What remains comes from rebalancing once a day and from volatility moving, which no delta hedge removes.',
    };
  }

  const c = result.counterfactual.hedgers;
  const bsGrowth = ratio(c.bs_delta.std, f.bs_delta.std);
  const hestonGrowth = ratio(c.heston_delta.std, f.heston_delta.std);
  const meanShift = c.heston_delta.mean - f.heston_delta.mean;
  const world = result.counterfactual.label.toLowerCase();
  const { how, horizon } = MECHANISM[scenario];

  if (Math.max(bsGrowth, hestonGrowth) < 1.15 && Math.abs(meanShift) < 0.5 * f.heston_delta.std) {
    return {
      headline: `A ${world} barely moves this ${result.book.maturity_days}-day book.`,
      detail:
        `Hedging error changes by ${fmt((bsGrowth - 1) * 100, 0)}% for Black-Scholes delta and ` +
        `${fmt((hestonGrowth - 1) * 100, 0)}% for Heston delta. In this world ${how}; that ${horizon}. ` +
        (result.book.maturity_days < LONGEST_MATURITY_DAYS
          ? 'Try a longer expiry or an out-of-the-money strike.'
          : 'Try an out-of-the-money strike or a higher severity.'),
    };
  }

  const loss = meanShift < -0.25 * f.heston_delta.std
    ? ` Every hedger also loses about ${fmt(-meanShift)} on average: the premium was set in the factual world and is too cheap for this one.`
    : '';
  const better = c.heston_delta.std < c.bs_delta.std ? 'Heston delta' : 'Black-Scholes delta';
  return {
    headline: `In a ${world}, Black-Scholes delta's hedging error grows ${fmt(bsGrowth, 1)}x; Heston delta's grows ${fmt(hestonGrowth, 1)}x.`,
    detail:
      `Black-Scholes delta keeps hedging at the vol it was set up with, while Heston delta tracks the variance as it moves. ` +
      `In this world ${how}, and ${better} ends up with the smaller error ` +
      `(${fmt(Math.min(c.bs_delta.std, c.heston_delta.std))} vs ${fmt(Math.max(c.bs_delta.std, c.heston_delta.std))}).` +
      loss,
  };
}
