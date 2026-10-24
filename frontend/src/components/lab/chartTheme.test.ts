import { baseLayout, chrome } from './chartTheme';

test('axis title text from a caller keeps the themed title font', () => {
  const layout = baseLayout('dark', { xaxis: { title: { text: 'P&L at expiry' }, zeroline: true } });
  expect(layout.xaxis?.title).toEqual({ text: 'P&L at expiry', font: { color: chrome('dark').inkSecondary, size: 11 } });
  expect(layout.xaxis?.zeroline).toBe(true);
  expect(layout.xaxis?.gridcolor).toBe(chrome('dark').grid);
  expect(layout.yaxis?.title).toEqual({ font: { color: chrome('dark').inkSecondary, size: 11 } });
});
