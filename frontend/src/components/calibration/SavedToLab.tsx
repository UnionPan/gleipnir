import React from 'react';
import type { CalibrationBundle } from '../../services/labApi';
import './SavedToLab.css';

interface Props {
  bundles: CalibrationBundle[];
  storageFailed: boolean;
  onOpenLab?: () => void;
}

/** Confirms that a fit now waits in the Hedging Lab: nothing to download. */
const SavedToLab: React.FC<Props> = ({ bundles, storageFailed, onOpenLab }) => {
  if (storageFailed) {
    return (
      <p className="saved-to-lab saved-to-lab-failed" role="status">
        <strong>Not saved for the Hedging Lab:</strong> this browser blocks site storage (a private window or blocked site
        data), so the lab cannot see this fit.
      </p>
    );
  }
  if (bundles.length === 0) return null;
  return (
    <p className="saved-to-lab" role="status">
      <span>
        Saved for the Hedging Lab{bundles.length > 1 ? `: ${bundles.map((b) => b.asset).join(', ')}` : ''}
        {' '}<code title="The fit's fingerprint">{bundles.map((b) => b.hash.slice(0, 8)).join(' ')}</code>
      </span>
      {onOpenLab && (
        <button type="button" className="lab-link" onClick={onOpenLab}>
          Sell and hedge an option on {bundles.length > 1 ? 'these fits' : 'this fit'} →
        </button>
      )}
    </p>
  );
};

export default SavedToLab;
