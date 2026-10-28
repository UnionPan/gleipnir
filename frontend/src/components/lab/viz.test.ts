import { act, renderHook } from '@testing-library/react';
import { useAnimationClock } from './viz';

test('a frame stamped before the clock started reads as time 0, never negative', () => {
  const frames: FrameRequestCallback[] = [];
  const raf = jest.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
    frames.push(cb);
    return frames.length;
  });
  jest.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => undefined);
  const { result } = renderHook(() => useAnimationClock(1000, 'key'));
  const started = performance.now();
  act(() => frames[frames.length - 1](started - 5));  // the frame's timestamp precedes the start
  expect(result.current.t).toBe(0);
  act(() => frames[frames.length - 1](started + 400));
  expect(result.current.t).toBeGreaterThan(0);
  expect(result.current.t).toBeLessThanOrEqual(1000);
  raf.mockRestore();
});
