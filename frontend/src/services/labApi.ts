import axios from 'axios';
import { describeApiError } from './apiError';
import { API_BASE_URL } from '../config/api.config';

export type HedgerName = 'no_hedge' | 'bs_delta' | 'heston_delta';
export type ScenarioName = 'vol_spike' | 'correlation_breakdown' | 'regime_shift';

export interface ScenarioInfo {
  name: ScenarioName;
  label: string;
  summary: string;
  severity_min: number;
  severity_max: number;
  severity_default: number;
}

/** Historical drift: educational only, never simulated. */
export interface DriftEstimate {
  estimate: number;
  std_error: number;
  ci_low: number;
  ci_high: number;
  years: number;
}

export interface AssetInfo {
  ticker: string;
  sector: string;
  params: Record<string, number>;
  params_hash: string;
  vol_today: number;
  as_of: string;
  diagnostics: Record<string, number>;
  historical_drift: DriftEstimate;
}

export interface LabCatalog {
  window: string;
  estimator: string;
  assets: AssetInfo[];
  scenarios: ScenarioInfo[];
  maturities_days: number[];
  n_paths: number;
  rho_assumption: number;
}

export interface HedgeRequest {
  asset: string;
  rho: number | null;
  scenario: { name: ScenarioName; severity: number } | null;
  option: { is_call: boolean; moneyness: number; maturity_days: number };
  cost_bps: number;
  seed: number;
}

export interface HedgerStats {
  mean: number;
  std: number;
  cvar_5: number;
  mean_cost: number;
  histogram: number[];
  below: number;
  above: number;
}

export interface WorldResult {
  label: string;
  description: string;
  params: Record<string, number>;
  params_hash: string;
  hedgers: Record<HedgerName, HedgerStats>;
  sample_path: {
    spots: number[];
    positions: Record<HedgerName, number[]>;
    pnl: Record<HedgerName, number>;
  };
}

export interface HedgeResponse {
  book: {
    asset: string;
    params: Record<string, number>;
    params_hash: string;
    parent_hash: string | null;
    rho_source: 'estimated' | 'assumed';
    strike: number;
    maturity_days: number;
    is_call: boolean;
    premium: number;
    implied_vol: number;
  };
  bin_edges: Record<HedgerName, number[]>;
  factual: WorldResult;
  counterfactual: WorldResult | null;
  n_paths: number;
  seed: number;
  cost_bps: number;
  elapsed_ms: number;
  mjollnir_version: string;
}

const REQUEST_TIMEOUT_MS = 60_000;

const TIMEOUT_MESSAGE = 'The simulation took longer than 60 s and was stopped.';

export async function fetchCatalog(): Promise<LabCatalog> {
  try {
    const response = await axios.get<LabCatalog>(`${API_BASE_URL}/lab/assets`, { timeout: REQUEST_TIMEOUT_MS });
    return response.data;
  } catch (error) {
    throw new Error(describeApiError(error, 'Failed to load the lab catalog', TIMEOUT_MESSAGE));
  }
}

export async function runHedge(request: HedgeRequest, signal: AbortSignal): Promise<HedgeResponse> {
  try {
    const response = await axios.post<HedgeResponse>(`${API_BASE_URL}/lab/hedge`, request, {
      timeout: REQUEST_TIMEOUT_MS,
      signal,
    });
    return response.data;
  } catch (error) {
    throw new Error(describeApiError(error, 'The simulation failed', TIMEOUT_MESSAGE));
  }
}
