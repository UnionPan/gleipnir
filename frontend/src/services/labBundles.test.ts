import type { CalibrationBundle } from './labApi';
import { listBundles, markBundlesSeen, saveBundle, unseenBundle } from './labBundles';

const bundle = (asset: string, model: CalibrationBundle['model'] = 'heston'): CalibrationBundle => ({
  asset, model, parameters: { sigma: 0.2 }, derived: {}, std_errors: {}, historical_drift: null,
  window: '2021-01-04..2026-01-02', as_of: '2026-01-02', rate: 0.04, dividend_yield: 0.01,
  method_label: 'test', note: null, hash: 'a'.repeat(64),
});

beforeEach(() => window.localStorage.clear());

test('a saved fit is listed newest first and marked unseen until the lab opens', () => {
  saveBundle(bundle('AAPL'), new Date('2026-01-01T00:00:00Z'));
  saveBundle(bundle('KO', 'gbm'), new Date('2026-01-02T00:00:00Z'));
  expect(listBundles().map((s) => s.key)).toEqual(['KO:gbm', 'AAPL:heston']);
  expect(unseenBundle()).toBe('KO:gbm');
  markBundlesSeen();
  expect(unseenBundle()).toBeNull();
});

test('refitting an asset and model replaces its bundle; the store keeps ten', () => {
  saveBundle(bundle('AAPL'), new Date('2026-01-01T00:00:00Z'));
  saveBundle({ ...bundle('AAPL'), hash: 'b'.repeat(64) }, new Date('2026-01-03T00:00:00Z'));
  expect(listBundles()).toHaveLength(1);
  expect(listBundles()[0].bundle.hash).toBe('b'.repeat(64));
  for (let i = 0; i < 12; i += 1) saveBundle(bundle(`T${i}`), new Date(Date.UTC(2026, 1, i + 1)));
  expect(listBundles()).toHaveLength(10);
  expect(listBundles()[0].key).toBe('T11:heston');
});

test('without storage the site keeps working', () => {
  const setItem = jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
    throw new Error('blocked');
  });
  expect(saveBundle(bundle('AAPL'))).toBe(false);
  expect(listBundles()).toEqual([]);
  setItem.mockRestore();
});
