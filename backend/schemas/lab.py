"""Schemas for the Counterfactual Hedging Lab."""

from typing import Literal

from pydantic import BaseModel, Field

LabModelName = Literal["heston", "gbm", "merton_jump", "regime_switching_gbm", "garch", "rough_bergomi"]
ScenarioName = Literal["vol_spike", "correlation_breakdown", "regime_shift", "jump_cascade"]
MaturityDays = Literal[5, 21, 63]
HedgerRole = Literal["none", "bs", "model"]


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
    model: LabModelName = Field("heston", description="The model the book is priced, hedged and simulated with")
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


class HedgerInfo(BaseModel):
    name: str
    role: HedgerRole = Field(..., description="none, bs (Black-Scholes at the premium's vol) or model")
    label: str
    note: str = Field(..., description="What the hedger knows and does")
    same_as: str | None = Field(None, description="Another hedger this one equals, trade for trade")


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
    model: LabModelName
    params: dict[str, float]
    params_hash: str
    parent_hash: str | None
    rho_source: Literal["estimated", "assumed"] | None = Field(..., description="null when the model has no rho")
    strike: float
    maturity_days: int
    is_call: bool
    premium: float
    implied_vol: float
    rate: float = Field(..., description="Risk-free rate r, continuously compounded")
    dividend_yield: float = Field(..., description="q; the spot drifts at r - q")


class HedgeResponse(BaseModel):
    book: Book
    hedgers: list[HedgerInfo] = Field(..., description="In display order: no hedge, Black-Scholes, the model's own")
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
    models: list[LabModelName] = Field(..., description="Models the scenario means something for")


class DriftEstimate(BaseModel):
    """Historical drift: educational only, never simulated (see the drift policy)."""

    estimate: float = Field(..., description="Annualized drift of the window")
    std_error: float = Field(..., description="Standard error, annualized")
    ci_low: float = Field(..., description="Lower end of the 95% confidence interval")
    ci_high: float = Field(..., description="Upper end of the 95% confidence interval")


class ModelFit(BaseModel):
    """One model's calibration of an asset: the world it runs and the fit behind it."""

    params: dict[str, float] = Field(..., description="World parameters (mjollnir names)")
    params_hash: str
    fit_params: dict[str, float] = Field(..., description="The Calibration tab's parameters")
    std_errors: dict[str, float]
    derived: dict[str, float]
    note: str | None


class AssetInfo(BaseModel):
    ticker: str
    sector: str
    vol_today: float
    as_of: str
    dividend_yield: float
    historical_drift: DriftEstimate
    models: dict[LabModelName, ModelFit]


class LabModelInfo(BaseModel):
    name: LabModelName
    label: str
    summary: str
    hedger_label: str
    hedger_note: str
    has_rho: bool


class LabCatalog(BaseModel):
    window: str
    rate: float
    rate_source: str
    assets: list[AssetInfo]
    models: list[LabModelInfo]
    scenarios: list[ScenarioInfo]
    maturities_days: list[int]
    n_paths: int
    rho_assumption: float
