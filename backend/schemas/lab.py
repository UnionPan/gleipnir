"""Schemas for the Counterfactual Hedging Lab."""

from typing import Literal

from pydantic import BaseModel, Field

HedgerName = Literal["no_hedge", "bs_delta", "heston_delta"]
ScenarioName = Literal["vol_spike", "correlation_breakdown", "regime_shift"]
MaturityDays = Literal[5, 21, 63]


class ScenarioSpec(BaseModel):
    """A counterfactual world: a named mjollnir scenario at a severity."""

    name: ScenarioName
    severity: float = Field(..., description="Scenario strength; range depends on the scenario")


class OptionSpec(BaseModel):
    """The option the book is short. Spot is normalized to 100."""

    is_call: bool = True
    moneyness: float = Field(1.0, ge=0.8, le=1.2, description="Strike as a fraction of spot")
    maturity_days: MaturityDays = Field(21, description="Trading days to expiry")


class HedgeRequest(BaseModel):
    asset: str = Field(..., description="Ticker from GET /lab/assets")
    rho: float | None = Field(
        None, ge=-0.95, le=0.0,
        description="Spot-vol correlation assumption; null keeps the estimated value",
    )
    scenario: ScenarioSpec | None = Field(None, description="Counterfactual world; null runs only the factual one")
    option: OptionSpec = OptionSpec()
    cost_bps: float = Field(5.0, ge=0.0, le=50.0, description="Proportional trading cost")
    seed: int = Field(7, ge=0, le=2**31 - 1)


class HedgerStats(BaseModel):
    mean: float
    std: float
    cvar_5: float = Field(..., description="Mean P&L over the worst 5% of paths")
    mean_cost: float
    histogram: list[int] = Field(..., description="Counts on this hedger's bin edges (shared by both worlds)")
    below: int = Field(..., description="Paths below the first bin edge (not in the histogram)")
    above: int = Field(..., description="Paths above the last bin edge (not in the histogram)")


class SamplePath(BaseModel):
    spots: list[float]
    positions: dict[str, list[float]] = Field(..., description="Hedge position held after each step")
    pnl: dict[str, float]


class WorldResult(BaseModel):
    label: str
    description: str
    params: dict[str, float]
    params_hash: str
    hedgers: dict[str, HedgerStats]
    sample_path: SamplePath


class Book(BaseModel):
    asset: str
    params: dict[str, float]
    params_hash: str
    parent_hash: str | None
    rho_source: Literal["estimated", "assumed"]
    strike: float
    maturity_days: int
    is_call: bool
    premium: float
    implied_vol: float


class HedgeResponse(BaseModel):
    book: Book
    bin_edges: dict[str, list[float]] = Field(..., description="Per hedger, shared by both worlds")
    factual: WorldResult
    counterfactual: WorldResult | None
    n_paths: int
    seed: int
    cost_bps: float
    elapsed_ms: float
    mjollnir_version: str


class ScenarioInfo(BaseModel):
    name: ScenarioName
    label: str
    summary: str
    severity_min: float
    severity_max: float
    severity_default: float


class DriftEstimate(BaseModel):
    """Historical drift: educational only, never simulated (see the drift policy)."""

    estimate: float = Field(..., description="Annualized drift of the calibration")
    std_error: float = Field(..., description="Approximate standard error, annualized")
    ci_low: float = Field(..., description="Lower end of the 95% confidence interval")
    ci_high: float = Field(..., description="Upper end of the 95% confidence interval")
    years: float = Field(..., description="Length of the estimation window")


class AssetInfo(BaseModel):
    ticker: str
    sector: str
    params: dict[str, float]
    params_hash: str
    vol_today: float
    as_of: str
    diagnostics: dict[str, float]
    historical_drift: DriftEstimate


class LabCatalog(BaseModel):
    window: str
    estimator: str
    assets: list[AssetInfo]
    scenarios: list[ScenarioInfo]
    maturities_days: list[int]
    n_paths: int
    rho_assumption: float
