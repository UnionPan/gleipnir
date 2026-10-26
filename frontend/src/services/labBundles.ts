/**
 * Calibration bundles kept in the visitor's browser, for the Hedging Lab.
 *
 * A successful Calibration-tab fit saves its bundle here (no download, no
 * link): the lab lists them as "Your calibrations", and the Hedging Lab tab
 * shows a marker until the visitor opens it. Storage can be unavailable
 * (private windows, blocked site data), so every access is guarded and the
 * site works without it.
 */
import type { CalibrationBundle } from './labApi';

const STORE_KEY = 'gleipnir.lab.bundles.v1';
const UNSEEN_KEY = 'gleipnir.lab.unseen.v1';
const MAX_BUNDLES = 10;
export const BUNDLES_EVENT = 'gleipnir:lab-bundles';

export interface StoredBundle {
  key: string;
  savedAt: string;
  bundle: CalibrationBundle;
}

export const bundleKey = (b: Pick<CalibrationBundle, 'asset' | 'model'>) => `${b.asset}:${b.model}`;

function read(): StoredBundle[] {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function write(items: StoredBundle[]): boolean {
  try {
    window.localStorage.setItem(STORE_KEY, JSON.stringify(items));
    return true;
  } catch {
    return false;
  }
}

/** Newest first. */
export function listBundles(): StoredBundle[] {
  return read().sort((a, b) => b.savedAt.localeCompare(a.savedAt));
}

/** Save (replacing the same asset and model) and mark it unseen; false if storage is unavailable. */
export function saveBundle(bundle: CalibrationBundle, now: Date = new Date()): boolean {
  const key = bundleKey(bundle);
  const items = [{ key, savedAt: now.toISOString(), bundle }, ...read().filter((s) => s.key !== key)]
    .sort((a, b) => b.savedAt.localeCompare(a.savedAt))
    .slice(0, MAX_BUNDLES);
  if (!write(items)) return false;
  try {
    window.localStorage.setItem(UNSEEN_KEY, key);
  } catch {
    // the marker is a convenience; the bundle itself is saved
  }
  window.dispatchEvent(new Event(BUNDLES_EVENT));
  return true;
}

/** The key of a bundle saved since the lab was last opened, if any. */
export function unseenBundle(): string | null {
  try {
    return window.localStorage.getItem(UNSEEN_KEY);
  } catch {
    return null;
  }
}

export function markBundlesSeen(): void {
  try {
    window.localStorage.removeItem(UNSEEN_KEY);
  } catch {
    // nothing to clear
  }
  window.dispatchEvent(new Event(BUNDLES_EVENT));
}
