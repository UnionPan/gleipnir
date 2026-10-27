import React from 'react';
import './CalibrationResults.css';
import { CalibrationResult } from '../../pages/CalibrationPage';
import type { CheckedAspect, PosteriorDiagnostics, PredictiveCheck } from '../../services/calibrationApi';
import {
  checkValue,
  derivedRow,
  driftSentence,
  formatSig,
  identificationLabel,
  intervalText,
  parameterDisplay,
  percent,
} from './receiptFormat';

interface CalibrationResultsProps {
  results: CalibrationResult[];
}

const MODEL_NAMES: Record<string, string> = {
  gbm: 'Geometric Brownian Motion',
  ou: 'Ornstein-Uhlenbeck',
  heston: 'Heston Stochastic Volatility',
  rough_bergomi: 'Rough Bergomi',
  regime_switching_gbm: 'Regime-Switching GBM',
  merton_jump: 'Merton Jump-Diffusion',
  garch: 'GARCH(1,1)',
};

const getModelName = (model: string): string => MODEL_NAMES[model] || model.toUpperCase();

const VERDICTS: Record<CheckedAspect['verdict'], { icon: string; text: string }> = {
  fits: { icon: '✓', text: 'Reproduced' },
  more: { icon: '▲', text: 'More than every simulated history' },
  less: { icon: '▼', text: 'Less than every simulated history' },
};

const count = (n: number | null | undefined) => (typeof n === 'number' ? n.toLocaleString('en-US') : '—');

/** Estimates with their classical standard errors (and bootstrap bias, where the estimator has one). */
const ClassicalTable: React.FC<{ result: CalibrationResult }> = ({ result }) => (
  <>
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
            const info = parameterDisplay(result.model, key);
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
  </>
);

function coverageRange(posterior: PosteriorDiagnostics): string | null {
  const values = Object.values(posterior.identification).map((row) => row.coverage_90);
  if (values.length === 0) return null;
  const low = Math.min(...values);
  const high = Math.max(...values);
  return low.toFixed(2) === high.toFixed(2) ? percent(low, 0) : `${percent(low, 0)}–${percent(high, 0)}`;
}

/** Posterior means with their spread and 90% credible intervals; state and drift rows keep their own errors. */
const PosteriorTable: React.FC<{ result: CalibrationResult; posterior: PosteriorDiagnostics }> = ({ result, posterior }) => {
  const coverage = coverageRange(posterior);
  const network = posterior.network;
  return (
    <>
      <div className="table-container">
        <table className="parameters-table">
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Symbol</th>
              <th className="value-column">Posterior mean</th>
              <th className="value-column sd-column">Std. dev.</th>
              <th className="value-column interval-column wide-only">90% interval</th>
              <th className="description-column">Description</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(result.parameters).map(([key, value]) => {
              const info = parameterDisplay(result.model, key);
              const summary = posterior.parameters[key];
              return (
                <tr key={key}>
                  <td className="param-name">{info.name}</td>
                  <td className="param-symbol">{info.symbol}</td>
                  <td className="param-value">
                    {formatSig(value)}
                    {summary && <span className="narrow-only stacked">{intervalText(summary.q05, summary.q95)}</span>}
                  </td>
                  <td className="param-value param-std-error sd-column">
                    {formatSig(summary ? summary.sd : result.diagnostics.std_errors?.[key], 2)}
                  </td>
                  <td className="param-value param-std-error wide-only">
                    {summary ? intervalText(summary.q05, summary.q95) : '—'}
                  </td>
                  <td className="param-description">{info.description}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="result-caption">
        {posterior.kind === 'exact'
          ? 'The exact posterior under the Jeffreys prior; its 90% interval is exact.'
          : <>
              A mixture-density network trained on {count(network?.trained_on)} simulated histories reads this
              one in a single pass. On {count(network?.validated_on)} fresh simulations its 90% intervals contained
              the true parameters {coverage ?? '—'} of the time. Prior: {posterior.prior}.
            </>}
        {' '}Rows without an interval are read off the data directly: today&apos;s variance, or the drift with its
        standard error.
      </p>
    </>
  );
};

/** How far the history narrowed each inferred quantity, and the network's offline coverage for it. */
const IdentificationTable: React.FC<{ model: string; posterior: PosteriorDiagnostics }> = ({ model, posterior }) => (
  <div className="parameters-section">
    <h4>🎯 What the History Pins Down</h4>
    <div className="table-container">
      <table className="parameters-table">
        <thead>
          <tr>
            <th>Quantity</th>
            <th className="contraction-column">Data vs prior</th>
            <th className="value-column" title="Offline: share of 90% intervals that contained the truth">90% coverage</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(posterior.identification).map(([key, row]) => {
            const share = Math.min(Math.max(row.contraction ?? 0, 0), 1);
            return (
              <tr key={key}>
                <td className="param-name">
                  {parameterDisplay(model, key).name}{' '}
                  <span className="param-symbol">{parameterDisplay(model, key).symbol}</span>
                </td>
                <td>
                  <div className="contraction">
                    <div className="contraction-track" aria-hidden="true">
                      <div className="contraction-fill" style={{ width: `${share * 100}%` }} />
                    </div>
                    <span className="contraction-label">
                      {identificationLabel(row.contraction)}
                      {row.contraction !== null && ` (${percent(share, 0)})`}
                    </span>
                  </div>
                </td>
                <td className="param-value param-std-error">{percent(row.coverage_90, 1)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
    <p className="result-caption">
      Data vs prior is how much the history narrowed each quantity: 100% would pin it exactly, 0% leaves the prior&apos;s
      spread. Coverage near 90% means the intervals can be taken at their word.
      {typeof posterior.effective_samples === 'number' && posterior.prior_kind === 'sector' && (
        <> The sector prior reweights {count(posterior.draws)} posterior draws, keeping the equivalent of
          {' '}{count(Math.round(posterior.effective_samples))}.</>
      )}
    </p>
  </div>
);

/** Each aspect of the history against histories simulated from the posterior. */
const PredictiveCheckTable: React.FC<{ check: PredictiveCheck }> = ({ check }) => (
  <div className="parameters-section">
    <h4>🔁 Does the Model Reproduce This History?</h4>
    <div className="table-container">
      <table className="parameters-table predictive-table">
        <thead>
          <tr>
            <th>Aspect</th>
            <th>Verdict</th>
            <th className="description-column">Measured by</th>
            <th className="value-column">This stock</th>
            <th className="value-column interval-column wide-only">Simulated, 5–95%</th>
          </tr>
        </thead>
        <tbody>
          {check.aspects.map((row) => {
            const verdict = VERDICTS[row.verdict];
            return (
              <tr key={row.aspect}>
                <td className="param-name">
                  {row.aspect}
                  <span className="narrow-only stacked">{row.summary}</span>
                </td>
                <td className={row.verdict === 'fits' ? 'verdict verdict-fits' : 'verdict verdict-misfit'}>
                  <span aria-hidden="true">{verdict.icon}</span>{'\u00a0'}{verdict.text}
                </td>
                <td className="param-description">{row.summary}</td>
                <td className="param-value">
                  {checkValue(row.observed)}
                  <span className="narrow-only stacked">sim. {intervalText(row.q05, row.q95)}</span>
                </td>
                <td className="param-value param-std-error wide-only">{intervalText(row.q05, row.q95)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
    <p className="result-caption">
      {check.n_replicates} histories as long as this one, each simulated from a posterior draw. Each aspect shows its
      most extreme statistic; a history beyond every simulated one is something this model leaves out, however well
      its parameters are pinned down.
    </p>
  </div>
);

const CalibrationResults: React.FC<CalibrationResultsProps> = ({ results }) => (
  <div className="calibration-results">
    {results.map((result, index) => {
      const posterior = result.diagnostics.posterior;
      const check = result.diagnostics.predictive_check;
      return (
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
            <h4>📊 {posterior ? 'Posterior' : 'Calibrated Parameters'}</h4>
            {posterior ? <PosteriorTable result={result} posterior={posterior} /> : <ClassicalTable result={result} />}
          </div>

          {posterior && Object.keys(posterior.identification).length > 0 && (
            <IdentificationTable model={result.model} posterior={posterior} />
          )}
          {check && <PredictiveCheckTable check={check} />}

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
            {posterior?.network && (
              <div className="info-badge" title={`Network weights SHA-256 ${posterior.network.sha256}`}>
                <span className="label">Network:</span>
                <span className="value">{posterior.network.sha256.slice(0, 8)}</span>
              </div>
            )}
          </div>
        </div>
      );
    })}
  </div>
);

export default CalibrationResults;
