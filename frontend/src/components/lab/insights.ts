/**
 * Plain-language readings of a lab result. Pure functions of the response so
 * the narrative stays consistent with the numbers on the page.
 */
import { hedgerByRole } from '../../services/labApi';
import type { HedgeResponse, LabModelName, ScenarioName } from '../../services/labApi';

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
    how: 'it makes volatility shocks last longer and lifts long-run variance',
    horizon: 'needs time to accumulate, so it matters most for longer maturities',
  },
  jump_cascade: {
    how: 'jumps arrive more often and hit harder, and no delta hedge can follow a jump',
    horizon: 'shows up at every maturity, most for out-of-the-money options',
  },
};

/** What the model's own hedger does differently from Black-Scholes delta. */
const MODEL_HEDGE: Record<LabModelName, string> = {
  heston: 'tracks the variance as it moves',
  gbm: 'hedges at the historical volatility',
  merton_jump: 'prices the jumps into its delta',
  regime_switching_gbm: 'switches its volatility with the regime',
  garch: 'follows the GARCH forecast of volatility to expiry',
  rough_bergomi: 'follows the instantaneous variance, noise included',
};

const LONGEST_MATURITY_DAYS = 63;
const TIE = 0.03; // errors within 3% read as a tie, not a win

const ratio = (a: number, b: number) => (b > 0 ? a / b : Number.NaN);
const fmt = (x: number, digits = 2) => x.toFixed(digits);

export interface Insight {
  headline: string;
  detail: string;
}

export function readResult(result: HedgeResponse, scenario: ScenarioName | null): Insight {
  const bs = hedgerByRole(result, 'bs');
  const model = hedgerByRole(result, 'model');
  const f = result.factual.hedgers;
  const reduction = 1 - ratio(f[bs.name].std, f.no_hedge.std);
  // Under Black-Scholes the model's delta is Black-Scholes delta at the premium's own vol.
  const same = model.same_as === bs.name;
  const modelPhrase = MODEL_HEDGE[result.book.model];

  if (!result.counterfactual || !scenario) {
    return {
      headline: `Delta hedging removes ${fmt(reduction * 100, 0)}% of the unhedged risk.`,
      detail:
        `Unhedged, the book's P&L has a standard deviation of ${fmt(f.no_hedge.std)}. ` +
        (same
          ? `Black-Scholes delta brings it to ${fmt(f[bs.name].std)}; here the model's own delta is the same hedge. `
          : `${bs.label} brings it to ${fmt(f[bs.name].std)} and ${model.label} to ${fmt(f[model.name].std)}. `) +
        'What remains comes from rebalancing once a day and from what the model lets move besides the price, ' +
        'which no delta hedge removes.',
    };
  }

  const c = result.counterfactual.hedgers;
  const bsGrowth = ratio(c[bs.name].std, f[bs.name].std);
  const modelGrowth = ratio(c[model.name].std, f[model.name].std);
  const meanShift = c[model.name].mean - f[model.name].mean;
  const world = result.counterfactual.label.toLowerCase();
  const { how, horizon } = MECHANISM[scenario];

  if (Math.max(bsGrowth, modelGrowth) < 1.15 && Math.abs(meanShift) < 0.5 * f[model.name].std) {
    return {
      headline: `A ${world} barely moves this ${result.book.maturity_days}-day book.`,
      detail:
        `Hedging error changes by ${fmt((bsGrowth - 1) * 100, 0)}% for ${bs.label}` +
        (same ? '' : ` and ${fmt((modelGrowth - 1) * 100, 0)}% for ${model.label}`) +
        `. In this world ${how}; that ${horizon}. ` +
        (result.book.maturity_days < LONGEST_MATURITY_DAYS
          ? 'Try a longer expiry or an out-of-the-money strike.'
          : 'Try an out-of-the-money strike or a higher severity.'),
    };
  }

  const loss = meanShift < -0.25 * f[model.name].std
    ? ` Every hedger also loses about ${fmt(-meanShift)} on average: the premium was set in the factual world and is too cheap for this one.`
    : '';
  if (same) {
    return {
      headline: `In a ${world}, delta hedging's error grows ${fmt(bsGrowth, 1)}x.`,
      detail: `In this world ${how}. Under Black-Scholes there is one volatility, so the model's delta and ` +
        `Black-Scholes delta are the same hedge, and neither sees the change coming.${loss}`,
    };
  }
  const [lo, hi] = [Math.min(c[bs.name].std, c[model.name].std), Math.max(c[bs.name].std, c[model.name].std)];
  const outcome = hi - lo < TIE * hi
    ? `the two hedges end up with about the same error (${fmt(lo)})`
    : `${c[model.name].std < c[bs.name].std ? model.label : bs.label} ends up with the smaller error (${fmt(lo)} vs ${fmt(hi)})`;
  return {
    headline: `In a ${world}, ${bs.label}'s hedging error grows ${fmt(bsGrowth, 1)}x; ${model.label}'s grows ${fmt(modelGrowth, 1)}x.`,
    detail:
      `${bs.label} keeps hedging at the vol it was set up with, while ${model.label} ${modelPhrase}. ` +
      `In this world ${how}, and ${outcome}.` +
      loss,
  };
}
