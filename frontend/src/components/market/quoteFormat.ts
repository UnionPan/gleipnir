/** How the Market page writes a quote: prices, signed changes, and which way a change went. */

export function priceText(price: number | null): string {
  if (price === null) return 'N/A';
  return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/** Rounded to cents first, so a change that rounds to zero reads 0.00, never -0.00. */
const cents = (v: number) => Number(v.toFixed(2)) || 0;
const signed = (v: number) => `${cents(v) > 0 ? '+' : ''}${cents(v).toFixed(2)}`;

/** "+14.09 (+0.18%)" */
export function changeText(change: number | null, changePercent: number | null): string {
  if (change === null || changePercent === null) return 'N/A';
  return `${signed(change)} (${signed(changePercent)}%)`;
}

/** "-1.35%" */
export function percentText(changePercent: number | null): string {
  return changePercent === null ? 'N/A' : `${signed(changePercent)}%`;
}

/** The badge's class: up, down, or neither (unchanged or unknown). It agrees with the sign written. */
export function direction(change: number | null): '' | 'positive' | 'negative' {
  if (change === null || cents(change) === 0) return '';
  return cents(change) > 0 ? 'positive' : 'negative';
}
