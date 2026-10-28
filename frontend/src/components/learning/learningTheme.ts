/**
 * Series colors for the Learning to Hedge page.
 *
 * Categorical slots 1-5 of the dataviz reference palette, one per entity and
 * fixed on every chart. Validated against the site's chart surface in both
 * modes for the sets drawn together: slots 1-3 (all pairs) and 3-4 (all
 * pairs). In light mode aqua and yellow sit below 3:1 against the surface,
 * so every chart that uses them labels its lines directly and has a table view.
 */
import type { HedgerKey } from '../../services/rlApi';
import type { ThemeName } from '../lab/chartTheme';

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

/** Slot ``i`` of the palette, for charts whose entities are not hedgers. */
export const slotColor = (theme: ThemeName, i: number) => SLOTS[theme][i];
