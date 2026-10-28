/**
 * Series colors for the Learning to Hedge page.
 *
 * Categorical slots 1-5 of the dataviz reference palette, one per entity and
 * fixed on every chart. Validated against the site's chart surface in both
 * modes for the sets drawn together: slots 1-3 (all pairs) and 3-4 (all
 * pairs). In light mode aqua and yellow sit below 3:1 against the surface,
 * so every chart that uses them labels its lines directly and has a table view.
 */
import type { CliquetPolicyKey, HedgerKey } from '../../services/rlApi';
import { chrome, ThemeName } from '../lab/chartTheme';

const SLOTS: Record<ThemeName, string[]> = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181'],
};

const HEDGER_SLOT: Record<HedgerKey, number> = {
  bs_delta: 0,
  delta_gamma: 1,
  lstm_options: 2,
  lstm_underlying: 3,
  delta_gamma_vega: 4,
};

export const hedgerColor = (theme: ThemeName, key: HedgerKey) => SLOTS[theme][HEDGER_SLOT[key]];

/**
 * One color per cliquet policy, fixed on every chart of the tab: the three
 * methods that learn from observations take slots 1-3, the structured hedges
 * slots 4-5, and the two references (no hedge, the exact-state learner) the
 * site's neutral inks.
 */
export function cliquetColor(theme: ThemeName, key: CliquetPolicyKey): string {
  const slots = SLOTS[theme];
  const c = chrome(theme);
  const colors: Record<CliquetPolicyKey, string> = {
    reactive: slots[0], sequential: slots[1], concurrent: slots[2], static_vanilla: slots[3],
    oracle_cap_exit: slots[4], no_hedge: c.baseline, oracle_state: c.inkSecondary,
  };
  return colors[key];
}
