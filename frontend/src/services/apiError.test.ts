import { describeApiError } from './apiError';

const axiosError = (status: number, data: unknown, headers: Record<string, string> = {}) => ({
  isAxiosError: true,
  response: { status, data, headers },
});

test('the API detail string is shown as is', () => {
  expect(describeApiError(axiosError(400, { detail: 'Unsupported model: foo' }), 'fallback'))
    .toBe('Unsupported model: foo');
});

test('validation errors read as sentences, not [object Object]', () => {
  const error = axiosError(422, {
    detail: [{ loc: ['body'], msg: 'Value error, the history window is limited to 10 years', type: 'value_error' }],
  });
  expect(describeApiError(error, 'fallback')).toBe('The history window is limited to 10 years');
});

test('the compute guard says when to retry', () => {
  const error = axiosError(429, { detail: 'Too many requests; slow down.' }, { 'retry-after': '7' });
  expect(describeApiError(error, 'fallback')).toBe('Too many requests; slow down. (retry in 7 s)');
});

test('a timeout and an unreachable API have their own messages', () => {
  expect(describeApiError({ isAxiosError: true, code: 'ECONNABORTED' }, 'fallback', 'Too slow.')).toBe('Too slow.');
  expect(describeApiError({ isAxiosError: true }, 'fallback')).toBe('Cannot reach the API. Is the backend running?');
  expect(describeApiError(new Error('boom'), 'fallback')).toBe('fallback');
});
