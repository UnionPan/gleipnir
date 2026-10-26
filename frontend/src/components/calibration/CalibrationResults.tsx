import React from 'react';
import './CalibrationResults.css';
import { CalibrationResult } from '../../pages/CalibrationPage';
import { derivedRow, driftSentence, formatSig } from './receiptFormat';

interface CalibrationResultsProps {
  results: CalibrationResult[];
}

const CalibrationResults: React.FC<CalibrationResultsProps> = ({ results }) => {
  const getParameterDisplay = (key: string): { name: string; symbol: string; description: string } => {
    const displayInfo: Record<string, { name: string; symbol: string; description: string }> = {
      mu: { name: 'Drift', symbol: 'μ', description: 'Historical drift: educational, not simulated' },
      sigma: { name: 'Volatility', symbol: 'σ', description: 'Standard deviation of returns' },
      kappa: { name: 'Mean Reversion', symbol: 'κ', description: 'Speed of mean reversion' },
      theta: { name: 'Long-run Level', symbol: 'θ', description: 'Where the process reverts to' },
      xi: { name: 'Vol of Vol', symbol: 'ξ', description: 'Volatility of volatility' },
      sigma_v: { name: 'Vol of Vol', symbol: 'σᵥ', description: 'Volatility of variance' },
      rho: { name: 'Correlation', symbol: 'ρ', description: 'Correlation between processes' },
      v0: { name: 'Variance Today', symbol: 'v₀', description: 'Trailing 21-day realized variance' },
      lambda: { name: 'Jump Intensity', symbol: 'λ', description: 'Average jumps per unit time' },
      mu_j: { name: 'Jump Mean', symbol: 'μⱼ', description: 'Average jump size' },
      sigma_j: { name: 'Jump Volatility', symbol: 'σⱼ', description: 'Jump size volatility' },
      omega: { name: 'GARCH Constant', symbol: 'ω', description: 'Long-term variance level' },
      alpha: { name: 'ARCH Coefficient', symbol: 'α', description: 'Reaction to past shocks' },
      beta: { name: 'GARCH Coefficient', symbol: 'β', description: 'Persistence of volatility' },
      nu: { name: 'Nu', symbol: 'ν', description: 'Parameter nu' },
      xi0: { name: 'Forward Variance', symbol: 'ξ₀', description: 'Variance level today (trailing 21 days)' },
      eta: { name: 'Vol of Vol', symbol: 'η', description: 'Volatility of volatility' },
      H: { name: 'Hurst Parameter', symbol: 'H', description: 'Roughness parameter' },
    };
    return displayInfo[key] || { name: key, symbol: key, description: '' };
  };

  const getModelName = (model: string): string => {
    const modelNames: Record<string, string> = {
      gbm: 'Geometric Brownian Motion',
      ou: 'Ornstein-Uhlenbeck',
      heston: 'Heston Stochastic Volatility',
      rough_bergomi: 'Rough Bergomi',
      regime_switching_gbm: 'Regime-Switching GBM',
      merton_jump: 'Merton Jump-Diffusion',
      garch: 'GARCH(1,1)',
    };
    return modelNames[model] || model.toUpperCase();
  };

  return (
    <div className="calibration-results">
      {results.map((result, index) => (
        <div key={index} className="result-card">
          {/* Header */}
          <div className="result-header">
            <div className="header-content">
              <h3>{getModelName(result.model)}</h3>
              <span className="ticker-badge">{result.ticker}</span>
            </div>
            <span className="timestamp">
              {new Date(result.timestamp).toLocaleString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              })}
            </span>
          </div>

          {result.diagnostics.method_label && (
            <p className="result-method">{result.diagnostics.method_label}</p>
          )}
          {result.diagnostics.converged === false && (
            <div className="diagnostic-note diagnostic-warning" role="alert">
              <strong>Not converged:</strong> the optimizer stopped without converging, so treat these values as unreliable.
            </div>
          )}

          {/* Parameters Table */}
          <div className="parameters-section">
            <h4>📊 Calibrated Parameters</h4>
            <div className="table-container">
              <table className="parameters-table">
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th>Symbol</th>
                    <th className="value-column">Estimate</th>
                    <th className="value-column">Std. error</th>
                    {result.diagnostics.bias && <th className="value-column">Bootstrap bias</th>}
                    <th className="description-column">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.parameters).map(([key, value]) => {
                    const info = getParameterDisplay(key);
                    const bias = result.diagnostics.bias?.[key];
                    return (
                      <tr key={key}>
                        <td className="param-name">{info.name}</td>
                        <td className="param-symbol">{info.symbol}</td>
                        <td className="param-value">{formatSig(value)}</td>
                        <td className="param-value param-std-error">{formatSig(result.diagnostics.std_errors?.[key], 2)}</td>
                        {result.diagnostics.bias && (
                          <td className="param-value param-std-error">
                            {bias === null || bias === undefined ? '—' : `${bias > 0 ? '+' : ''}${formatSig(bias, 2)}`}
                          </td>
                        )}
                        <td className="param-description">{info.description}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {result.diagnostics.bias && (
              <p className="result-caption">
                Standard errors and bias come from refitting histories simulated from this fit; a bias of −0.3 means
                the estimator reads 0.3 low on average at these values.
              </p>
            )}
          </div>

          {result.diagnostics.drift && (
            <div className="diagnostic-note">
              {driftSentence(result.diagnostics.drift)} Too uncertain to forecast with: simulations use r − q with
              no excess drift.
            </div>
          )}

          {/* Readable quantities and fit statistics */}
          <div className="diagnostics-section">
            <h4>📈 What the Fit Says</h4>
            <div className="table-container">
              <table className="diagnostics-table">
                <tbody>
                  {Object.entries(result.diagnostics.derived ?? {})
                    .filter((entry): entry is [string, number] => entry[1] !== null && entry[1] !== undefined)
                    .map(([key, value]) => {
                      const row = derivedRow(key, value);
                      return (
                        <tr key={key}>
                          <td className="diag-name">{row.label}</td>
                          <td className="diag-value">{row.value}</td>
                          <td className="diag-interpretation">{row.hint}</td>
                        </tr>
                      );
                    })}
                  {typeof result.diagnostics.rmse_iv === 'number' && (
                    <tr>
                      <td className="diag-name">Fit error</td>
                      <td className="diag-value">{(result.diagnostics.rmse_iv * 100).toFixed(2)} vol pts</td>
                      <td className="diag-interpretation">
                        Root-mean-square implied-vol error over {result.diagnostics.n_quotes} quotes
                      </td>
                    </tr>
                  )}
                  {typeof result.diagnostics.logLikelihood === 'number' && (
                    <tr>
                      <td className="diag-name">Log-likelihood</td>
                      <td className="diag-value">{result.diagnostics.logLikelihood.toFixed(1)}</td>
                      <td className="diag-interpretation">Higher fits the data better</td>
                    </tr>
                  )}
                  {typeof result.diagnostics.aic === 'number' && (
                    <tr>
                      <td className="diag-name">AIC / BIC</td>
                      <td className="diag-value">
                        {result.diagnostics.aic.toFixed(1)} / {formatSig(result.diagnostics.bic, 6)}
                      </td>
                      <td className="diag-interpretation">Lower is better; compares models on the same window</td>
                    </tr>
                  )}
                  {typeof result.diagnostics.variogram_r2 === 'number' && (
                    <tr>
                      <td className="diag-name">Variogram R²</td>
                      <td className="diag-value">{result.diagnostics.variogram_r2.toFixed(3)}</td>
                      <td className="diag-interpretation">How well a + bΔ^2H describes the variogram</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {result.diagnostics.note && (
              <div className="diagnostic-note">
                <strong>ℹ️ Note:</strong> {result.diagnostics.note}
              </div>
            )}
          </div>

          {/* Method Info */}
          <div className="method-info">
            <div className="info-badge">
              <span className="label">Measure:</span>
              <span className="value">{result.measure || 'P-measure'}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default CalibrationResults;
