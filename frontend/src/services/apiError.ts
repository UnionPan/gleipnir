import axios from 'axios';

/** "Value error, the history window is limited to 10 years" -> "The history window ..." */
function validationMessage(item: unknown): string {
  const msg = typeof item === 'object' && item !== null ? (item as { msg?: unknown }).msg : undefined;
  if (typeof msg !== 'string') return String(item);
  const text = msg.replace(/^Value error, /, '');
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/**
 * A readable message for a failed API call: the API's own explanation when it
 * gave one (including validation errors and the compute guard's 429/503 with
 * their Retry-After), otherwise ``fallback``.
 */
export function describeApiError(error: unknown, fallback: string, timeoutMessage?: string): string {
  if (axios.isCancel(error)) return 'cancelled';
  if (!axios.isAxiosError(error)) return fallback;
  if (error.code === 'ECONNABORTED') return timeoutMessage ?? 'The request took too long and was stopped.';
  const response = error.response;
  if (!response) return 'Cannot reach the API. Is the backend running?';
  const detail = (response.data as { detail?: unknown } | undefined)?.detail;
  let message = fallback;
  if (typeof detail === 'string') message = detail;
  else if (Array.isArray(detail) && detail.length > 0) message = detail.map(validationMessage).join('; ');
  const retryAfter = Number(response.headers?.['retry-after']);
  if ((response.status === 429 || response.status === 503) && Number.isFinite(retryAfter) && retryAfter > 0) {
    message = `${message} (retry in ${retryAfter} s)`;
  }
  return message;
}
