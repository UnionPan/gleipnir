import axios from 'axios';
import { describeApiError } from './apiError';
import { API_BASE_URL } from '../config/api.config';

export type LabModelName = 'heston' | 'gbm' | 'merton_jump' | 'regime_switching_gbm' | 'garch' | 'rough_bergomi';
export type ScenarioName = 'vol_spike' | 'correlation_breakdown' | 'regime_shift' | 'jump_cascade';
/** Hedger names vary with the model (heston_delta, merton_delta, ...); roles do not. */
export type HedgerName = string;
export type HedgerRole = 'none' | 'bs' | 'model';

export interface ScenarioInfo {
  name: ScenarioName;
  label: string;
  summary: string;
  severity_min: number;
  severity_max: number;
  severity_default: number;
  models: LabModelName[];
}

/** Historical drift: educational only, never simulated. */
export interface DriftEstimate {
  estimate: number;
  std_error: number;
  ci_low: number;
  ci_high: number;
}

/** One model's calibration of an asset: the world it runs and the fit behind it. */
export interface ModelFit {
  params: Record<string, number>;
  params_hash: string;
  fit_params: Record<string, number>;
  std_errors: Record<string, number>;
  derived: Record<string, number>;
  note: string | null;
}

export interface AssetInfo {
  ticker: string;
  sector: string;
  vol_today: number;
  as_of: string;
  dividend_yield: number;
  historical_drift: DriftEstimate;
  models: Record<LabModelName, ModelFit>;
}

export interface LabModelInfo {
  name: LabModelName;
  label: string;
  summary: string;
  hedger_label: string;
  hedger_note: string;
  has_rho: boolean;
}

export interface LabCatalog {
  window: string;
  rate: number;
  rate_source: string;
  assets: AssetInfo[];
  models: LabModelInfo[];
  scenarios: ScenarioInfo[];
  maturities_days: number[];
  n_paths: number;
  rho_assumption: number;
}

export interface HedgeRequest {
  asset: string;
  model: LabModelName;
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

export interface HedgerInfo {
  name: HedgerName;
  role: HedgerRole;
  label: string;
  note: string;
  /** Another hedger this one equals, trade for trade (GBM's delta is Black-Scholes delta). */
  same_as?: HedgerName | null;
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
    model: LabModelName;
    params: Record<string, number>;
    params_hash: string;
    parent_hash: string | null;
    rho_source: 'estimated' | 'assumed' | null;
    strike: number;
    maturity_days: number;
    is_call: boolean;
    premium: number;
    implied_vol: number;
    rate: number;
    dividend_yield: number;
  };
  /** In display order: no hedge, Black-Scholes, the model's own. */
  hedgers: HedgerInfo[];
  bin_edges: Record<HedgerName, number[]>;
  factual: WorldResult;
  counterfactual: WorldResult | null;
  n_paths: number;
  seed: number;
  cost_bps: number;
  elapsed_ms: number;
  mjollnir_version: string;
}

/** The hedger playing ``role`` in a result. */
export function hedgerByRole(result: HedgeResponse, role: HedgerRole): HedgerInfo {
  const hedger = result.hedgers.find((h) => h.role === role);
  if (!hedger) throw new Error(`no ${role} hedger in the result`);
  return hedger;
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
