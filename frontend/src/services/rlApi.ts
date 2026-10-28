import axios from 'axios';
import { describeApiError } from './apiError';
import { API_BASE_URL } from '../config/api.config';

/** A metric across training seeds: its mean, sample standard deviation and per-seed values. */
export interface SeedStat {
  mean: number;
  sd: number;
  values: number[];
}

export type HedgerKey = 'bs_delta' | 'delta_gamma' | 'delta_gamma_vega' | 'lstm_underlying' | 'lstm_options';

export interface HedgerMetrics<T> {
  mean_error: T;
  std_error: T;
  expected_shortfall_95: T;
  mean_cost: T;
  option_turnover: T;
}

export interface LearnedResult extends HedgerMetrics<SeedStat> {
  seeds: number[];
  representative_seed: number;
}

export interface OptionSlot {
  kind: 'call' | 'put';
  tenor_days: number;
  moneyness: number;
}

export interface Bands {
  p5: number[];
  p25: number[];
  p50: number[];
  p75: number[];
  p95: number[];
}

export interface DeepHedgingDemo {
  schema_version: number;
  created_at: string;
  setup: {
    market: { kappa: number; theta: number; sigma_v: number; rho: number; r: number; q: number };
    initial_spot: number;
    initial_variance: number;
    horizon_days: number;
    liability: string;
    options: string;
    costs: { underlying: number; options: number };
    objective: string;
    training: { n_epochs: number; batch_size: number; learning_rate: number; cosine_lr: boolean; seeds: number[] };
    eval_paths: number;
    eval_seed: number;
    mean_payoff: number;
    /** The cash the hedgers start with: 0, so their errors are net of the premium (schema 2). */
    initial_cash?: number;
    /** The call's true-model value on day 0: what a desk selling it at fair value receives (schema 2). */
    fair_premium?: number;
    /** The listed option slots, in the order of the option holdings (schema 2). */
    option_grid?: OptionSlot[];
  };
  hedgers: { key: HedgerKey; label: string; summary: string }[];
  results: Record<'bs_delta' | 'delta_gamma' | 'delta_gamma_vega', HedgerMetrics<number>>
    & Record<'lstm_underlying' | 'lstm_options', LearnedResult>;
  histogram: { edges: number[] } & Record<HedgerKey, { counts: number[] }>;
  samples: {
    spot: number[];
    positions: Record<HedgerKey, number[]>;
    option_value: Record<HedgerKey, number[]>;
    /** The true-model value of the call the desk owes, each day; the payoff at expiry (schema 2). */
    liability_value?: number[];
    /** Each hedger's hedge account (cash received, plus gains, less costs), each day (schema 2). */
    portfolio?: Record<HedgerKey, number[]>;
    /** For hedgers that trade options: holdings of each slot after each day's trade (schema 2). */
    option_holdings?: Partial<Record<HedgerKey, number[][]>>;
  }[];
  /** Per hedger, the book marked to market (account less liability) across all paths, each day (schema 2). */
  wealth_bands?: Record<HedgerKey, Bands>;
  /** Shared-bin histograms of the P&L at expiry (schema 2). */
  terminal_histogram?: { edges: number[] } & Record<HedgerKey, { counts: number[] }>;
  training_curves: Record<'lstm_underlying' | 'lstm_options', Record<string, { epoch: number[]; std_error: number[] }>>;
  provenance: { run: string; checkpoints_sha256: Record<string, string>; gleipnir_commit: string };
}

export type CliquetMethodKey = 'reactive' | 'sequential' | 'concurrent' | 'oracle_state';
export type CliquetBaselineKey = 'no_hedge' | 'static_vanilla' | 'oracle_cap_exit';

export interface CliquetDemo {
  schema_version: number;
  created_at: string;
  setup: {
    contract: string;
    hedge: string;
    hidden: string;
    market: string;
    calibration_hash: string;
    proportional_cost: number;
    training: { iterations: number; batch_size: number; seeds: number[] };
  };
  baselines: Record<CliquetBaselineKey, {
    label: string;
    summary: string;
    terminal_pnl_rmse: number;
    terminal_abs_cvar_95: number;
    mean_trading_cost: number;
  }>;
  methods: Record<CliquetMethodKey, {
    label: string;
    summary: string;
    seeds: number[];
    terminal_pnl_rmse: SeedStat;
    terminal_abs_cvar_95: SeedStat;
    mean_trading_cost: SeedStat;
    period_return_probe_r2: SeedStat;
    accrued_return_probe_r2: SeedStat;
    curves: { iteration: number[]; terminal_pnl_rmse: number[]; reward_mse: number[] }[];
  }>;
  /** What every policy did along the benchmark's evaluation paths (absent from older exports). */
  paths?: CliquetPaths;
  provenance: { results_file: string; results_sha256: string; source: string };
}

export type CliquetPolicyKey = CliquetBaselineKey | CliquetMethodKey;

export interface CliquetPaths {
  /** Decision periods; each spans ``days_per_step`` trading days. */
  steps: number;
  days_per_step: number;
  /** The periods after which the contract resets. */
  reset_steps: number[];
  premium: number;
  notional: number;
  global_cap: number;
  local_cap: number;
  representative_seeds: Record<CliquetMethodKey, number>;
  evaluation_paths: number;
  terminal_histogram: { edges: number[] } & Record<CliquetPolicyKey, number[]>;
  /** Per policy, the running P&L (premium + hedge gains - payoff locked in so far) per period. */
  pnl_bands: Record<CliquetPolicyKey, Bands>;
  /** The payoff locked in so far, the same for every policy on a path. */
  locked_in_bands: Bands;
  /** Share of paths whose payoff has reached the cap, per period. */
  capped_share: number[];
  samples: {
    spot: number[];
    locked_in: number[];
    exposure: Record<CliquetPolicyKey, number[]>;
    pnl: Record<CliquetPolicyKey, number[]>;
  }[];
}

export interface LearningDemos {
  deep_hedging: DeepHedgingDemo | null;
  cliquet_ais: CliquetDemo | null;
}

export async function fetchLearningDemos(): Promise<LearningDemos> {
  try {
    const response = await axios.get<LearningDemos>(`${API_BASE_URL}/rl/demos`, { timeout: 30_000 });
    return response.data;
  } catch (error) {
    throw new Error(describeApiError(error, 'Could not load the trained hedgers'));
  }
}
