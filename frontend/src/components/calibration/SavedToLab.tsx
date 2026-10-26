import React from 'react';
import type { CalibrationBundle } from '../../services/labApi';
import './SavedToLab.css';

interface Props {
  bundles: CalibrationBundle[];
  storageFailed: boolean;
}

const MODEL_LABELS: Record<string, string> = {
  heston: 'Heston', gbm: 'Black-Scholes (GBM)', merton_jump: 'Merton jump-diffusion',
  regime_switching_gbm: 'Regime-switching GBM', garch: 'GARCH(1,1)', rough_bergomi: 'Rough Bergomi',
};

/** Confirms that a fit now waits in the Hedging Lab: nothing to download or click. */
const SavedToLab: React.FC<Props> = ({ bundles, storageFailed }) => {
  if (storageFailed) {
    return (
      <div className="saved-to-lab saved-to-lab-failed" role="status">
        <strong>Not saved for the Hedging Lab.</strong> This browser blocks site storage (a private window or
        blocked site data), so the lab cannot see this fit.
      </div>
    );
  }
  if (bundles.length === 0) return null;
  return (
    <div className="saved-to-lab" role="status">
      <strong>Saved to the Hedging Lab.</strong>{' '}
      {bundles.map((b, i) => (
        <span key={`${b.asset}:${b.model}`}>
          {i > 0 && ', '}
          {b.asset} · {MODEL_LABELS[b.model] ?? b.model} <code>{b.hash.slice(0, 8)}</code>
        </span>
      ))}
      . Open the Hedging Lab tab to sell an option on {bundles.length > 1 ? 'these calibrations' : 'this calibration'},
      hedge it, and stress it.
    </div>
  );
};

export default SavedToLab;
