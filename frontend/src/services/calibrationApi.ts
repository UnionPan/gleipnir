import axios from 'axios';
import { describeApiError } from './apiError';
import type { CalibrationBundle } from './labApi';
import { API_BASE_URL } from '../config/api.config';

export interface OHLCVData {
  ticker: string;
  dates: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

export type PriorChoice = 'sector' | 'broad';

export interface CalibrationRequest {
  tickers: string[];
  model: string;
  startDate: string;
  endDate: string;
  method: string;
  includeDrift: boolean;
  /** Bayesian fits: the ticker's S&P 500 sector prior when it has one, or the broad training prior. */
  prior?: PriorChoice;
}

/** The window's drift: an educational estimate, never simulated as a forecast. */
export interface HistoricalDrift {
  estimate: number;
  std_error: number;
  ci_low: number;
  ci_high: number;
}

/** A parameter's posterior summary; q05 and q95 bound its 90% credible interval. */
export interface PosteriorParameter {
  mean: number;
  sd: number;
  q05: number;
  q95: number;
}

/** How well the history pins down one of the network's coordinates, and how far to trust the network. */
export interface Identification {
  /** 1 - posterior variance / prior variance: near 0, the interval is the prior's. */
  contraction: number | null;
  /** Offline, on fresh simulations: share of 50% / 90% intervals that contained the truth. */
  coverage_50: number;
  coverage_90: number;
}

export interface PosteriorDiagnostics {
  kind: 'neural' | 'exact';
  parameters: Record<string, PosteriorParameter>;
  identification: Record<string, Identification>;
  prior: string;
  prior_kind: 'sector' | 'broad' | 'jeffreys';
  draws?: number | null;
  effective_samples?: number | null;
  in_prior_share?: number | null;
  n_days: number;
  network?: {
    components: number;
    hidden: number[];
    trained_on?: number | null;
    validated_on?: number | null;
    sha256: string;
    created_at: string;
  } | null;
}

/** One aspect of the history against histories simulated from the posterior. */
export interface CheckedAspect {
  aspect: string;
  /** 'more' / 'less': the history has more / less of it than every simulated history. */
  verdict: 'fits' | 'more' | 'less';
  summary: string;
  observed: number;
  q05: number;
  q50: number;
  q95: number;
  p_value: number;
}

export interface PredictiveCheck {
  n_replicates: number;
  aspects: CheckedAspect[];
}

/** Diagnostics of a P or Q fit; the API reports non-finite numbers as null. */
export interface CalibrationDiagnostics {
  method_label?: string | null;
  converged?: boolean | null;
  logLikelihood?: number | null;
  aic?: number | null;
  bic?: number | null;
  variogram_r2?: number | null;
  std_errors?: Record<string, number | null> | null;
  bias?: Record<string, number | null> | null;
  derived?: Record<string, number | null> | null;
  drift?: HistoricalDrift | null;
  errorMetrics?: Record<string, number> | null;
  note?: string | null;
  rmse_iv?: number | null;
  n_quotes?: number | null;
  n_expiries?: number | null;
  posterior?: PosteriorDiagnostics | null;
  predictive_check?: PredictiveCheck | null;
}

export interface CalibrationResponse {
  ticker: string;
  model: string;
  parameters: Record<string, number>;
  diagnostics: CalibrationDiagnostics;
  /** The fit as the Hedging Lab takes it; null for models the lab does not run. */
  lab_bundle?: CalibrationBundle | null;
  timestamp: string;
  method?: string;
  measure?: string;
}

export interface SimulationRequest {
  model: string;
  parameters: Record<string, number>;
  s0: number;
  nSteps: number;
  nPaths: number;
  dt: number;
  maxPathsReturn?: number;
  randomSeed?: number;
  riskFreeRate?: number;
  dividendYield?: number;
}

export type QuantileName = 'p5' | 'p25' | 'p50' | 'p75' | 'p95';

/** A drift other than the policy's, drawn as the median path it would give. */
export interface DriftScenario {
  kind: 'positive' | 'negative' | 'historical';
  label: string;
  excess_drift: number;
  median: number[];
}

export interface SimulationResponse {
  model: string;
  n_steps: number;
  n_paths: number;
  dt: number;
  time_grid: number[];
  mean_path: number[];
  quantiles: Record<QuantileName, number[]>;
  sample_paths: number[][];
  stats: {
    mean: number;
    std: number;
    p5: number;
    p50: number;
    p95: number;
    min: number;
    max: number;
  };
  spot_drift: number | null;
  drift_note: string;
  scenarios: DriftScenario[];
}

// Q-measure interfaces
export interface OptionQuote {
  strike: number;
  expiry: string;
  option_type: string;
  bid: number;
  ask: number;
  mid: number;
  last: number;
  volume: number;
  open_interest: number;
  implied_volatility?: number;
  moneyness: number;
}

export interface OptionChainData {
  ticker: string;
  spot_price: number;
  reference_date: string;
  risk_free_rate: number;
  dividend_yield: number;
  options: OptionQuote[];
  expiries: string[];
  n_options: number;
}

export interface QMeasureCalibrationRequest {
  ticker: string;
  model: string;
  referenceDate?: string;
  riskFreeRate: number;
  filterParams?: {
    min_volume?: number;
    min_open_interest?: number;
    max_spread_pct?: number;
    moneyness_range?: [number, number];
  };
  calibrationMethod: string;
  maxiter: number;
  /** Shortest expiry a Heston or Bates fit uses, in days (the server's default: 30). */
  minDays?: number;
}

export interface QMeasureCalibrationResponse {
  ticker: string;
  model: string;
  parameters: Record<string, number>;
  diagnostics: CalibrationDiagnostics;
  timestamp: string;
  measure: string;
}

export interface VolSurfaceRequest {
  ticker: string;
  model: string;
  parameters: Record<string, number>;
  spotPrice: number;
  riskFreeRate: number;
  dividendYield?: number;
  nStrikes?: number;
  nMaturities?: number;
  strikeRange?: [number, number];
  maturityRange?: [number, number];
}

export interface VolSurfaceResponse {
  strikes: number[];
  maturities: number[];
  vols: (number | null)[][];  // maturities x strikes; null where the model vol is undefined
  surface_type: 'implied_vol' | 'local_vol';
  ticker: string;
  model: string;
}

class CalibrationApi {
  /**
   * Fetch OHLCV data from yfinance
   */
  async fetchOHLCV(
    tickers: string[],
    startDate: string,
    endDate: string,
    signal?: AbortSignal,
  ): Promise<OHLCVData[]> {
    try {
      const response = await axios.post(`${API_BASE_URL}/calibration/fetch-data`, {
        tickers,
        start_date: startDate,
        end_date: endDate,
      }, { signal });
      return response.data;
    } catch (error) {
      throw new Error(describeApiError(error, 'Failed to fetch OHLCV data'));
    }
  }

  /**
   * Run P-measure calibration
   */
  async calibrate(request: CalibrationRequest, signal?: AbortSignal): Promise<CalibrationResponse[]> {
    try {
      const response = await axios.post(`${API_BASE_URL}/calibration/calibrate`, {
        tickers: request.tickers,
        model: request.model,
        start_date: request.startDate,
        end_date: request.endDate,
        method: request.method,
        include_drift: request.includeDrift,
        prior: request.prior,
      }, { signal });
      return response.data;
    } catch (error) {
      throw new Error(describeApiError(error, 'Calibration failed'));
    }
  }

  /**
   * Fetch option chain from yfinance
   */
  async fetchOptionChain(
    ticker: string,
    referenceDate?: string,
    riskFreeRate: number = 0.05,
    expiry?: string,
    signal?: AbortSignal,
  ): Promise<OptionChainData> {
    try {
      const response = await axios.post(`${API_BASE_URL}/calibration/fetch-option-chain`, {
        ticker,
        reference_date: referenceDate,
        risk_free_rate: riskFreeRate,
        expiry,
      }, { signal });
      return response.data;
    } catch (error) {
      throw new Error(describeApiError(error, 'Failed to fetch option chain'));
    }
  }

  /**
   * Run Q-measure calibration
   */
  async calibrateQMeasure(request: QMeasureCalibrationRequest, signal?: AbortSignal): Promise<QMeasureCalibrationResponse> {
    try {
      const response = await axios.post(`${API_BASE_URL}/calibration/calibrate-qmeasure`, {
        ticker: request.ticker,
        model: request.model,
        reference_date: request.referenceDate,
        risk_free_rate: request.riskFreeRate,
        filter_params: request.filterParams,
        calibration_method: request.calibrationMethod,
        maxiter: request.maxiter,
        min_days: request.minDays,
      }, { signal });
      return response.data;
    } catch (error) {
      throw new Error(describeApiError(error, 'Q-measure calibration failed'));
    }
  }

  /**
   * Run counterfactual simulation from calibrated parameters
   */
  async simulate(request: SimulationRequest, signal?: AbortSignal): Promise<SimulationResponse> {
    try {
      const response = await axios.post(`${API_BASE_URL}/calibration/simulate`, {
        model: request.model,
        parameters: request.parameters,
        s0: request.s0,
        n_steps: request.nSteps,
        n_paths: request.nPaths,
        dt: request.dt,
        max_paths_return: request.maxPathsReturn,
        random_seed: request.randomSeed,
        risk_free_rate: request.riskFreeRate,
        dividend_yield: request.dividendYield,
      }, { signal });
      return response.data;
    } catch (error) {
      throw new Error(describeApiError(error, 'Simulation failed'));
    }
  }

  /**
   * Generate 3D volatility surface from calibrated Q-measure model
   */
  async generateVolSurface(request: VolSurfaceRequest, signal?: AbortSignal): Promise<VolSurfaceResponse> {
    try {
      const response = await axios.post(`${API_BASE_URL}/calibration/vol-surface`, {
        ticker: request.ticker,
        model: request.model,
        parameters: request.parameters,
        spot_price: request.spotPrice,
        risk_free_rate: request.riskFreeRate,
        dividend_yield: request.dividendYield,
        n_strikes: request.nStrikes,
        n_maturities: request.nMaturities,
        strike_range: request.strikeRange,
        maturity_range: request.maturityRange,
      }, { signal });
      return response.data;
    } catch (error) {
      throw new Error(describeApiError(error, 'Volatility surface generation failed'));
    }
  }
}

export const calibrationApi = new CalibrationApi();
