import { changeText, direction, percentText, priceText } from './quoteFormat';

test('a change that rounds to zero is written and coloured as unchanged', () => {
  expect(changeText(-0.0012, -0.000004)).toBe('0.00 (0.00%)');
  expect(direction(-0.0012)).toBe('');
  expect(changeText(-0.09, -0.0003)).toBe('-0.09 (0.00%)');
  expect(direction(-0.09)).toBe('negative');
  expect(percentText(-1.354)).toBe('-1.35%');
});

test('prices and missing quotes', () => {
  expect(priceText(7736.69)).toBe('7,736.69');
  expect(priceText(null)).toBe('N/A');
  expect(changeText(14.09, 0.18)).toBe('+14.09 (+0.18%)');
  expect(changeText(null, 0.1)).toBe('N/A');
  expect(direction(null)).toBe('');
});
