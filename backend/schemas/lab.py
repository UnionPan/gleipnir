"""Schemas for the Counterfactual Hedging Lab."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

LabModelName = Literal["heston", "gbm", "merton_jump", "regime_switching_gbm", "garch", "rough_bergomi"]
ScenarioName = Literal["vol_spike", "correlation_breakdown", "regime_shift", "jump_cascade"]
MaturityDays = Literal[5, 21, 63]
HedgerRole = Literal["none", "bs", "model", "learned"]


class ScenarioSpec(BaseModel):
    """A counterfactual world: a named mjollnir scenario at a severity."""

    name: ScenarioName
    severity: float = Field(..., description="Scenario strength; range depends on the scenario")


class OptionSpec(BaseModel):
    """The option the book is short. Spot is normalized to 100."""

    is_call: bool = True
    moneyness: float = Field(1.0, ge=0.8, le=1.2, description="Strike as a fraction of spot")
    maturity_days: MaturityDays = Field(21, description="Trading days to expiry")


class DriftEstimate(BaseModel):
    """Historical drift: educational only, never simulated (see the drift policy)."""

    estimate: float = Field(..., description="Annualized drift of the window")
    std_error: float = Field(..., description="Standard error, annualized")
    ci_low: float = Field(..., description="Lower end of the 95% confidence interval")
    ci_high: float = Field(..., description="Upper end of the 95% confidence interval")


class CalibrationBundle(BaseModel):
    """A Calibration-tab fit, carried to the lab by the visitor's browser
    (backend.services.lab_bundle). Every value is re-validated by the lab."""

    asset: str = Field(..., pattern=r"^[A-Za-z0-9.^=\-]{1,15}$")
    model: LabModelName
    parameters: dict[str, float]
    derived: dict[str, float] = Field(default_factory=dict)
    std_errors: dict[str, float] = Field(default_factory=dict)
    historical_drift: DriftEstimate | None = None
    window: str = Field(..., max_length=40)
    as_of: str = Field(..., max_length=20)
    rate: float = Field(..., ge=-0.05, le=0.25)
    dividend_yield: float = Field(..., ge=0.0, le=0.25)
    method_label: str = Field("", max_length=200)
    note: str | None = Field(None, max_length=2000)
    hash: str = Field(..., pattern=r"^[0-9a-f]{64}$")

    @field_validator("parameters", "derived", "std_errors")
    @classmethod
    def _small(cls, value: dict[str, float]) -> dict[str, float]:
        if len(value) > 32:
            raise ValueError("at most 32 entries")
        return value


class HedgeRequest(BaseModel):
    asset: str | None = Field(None, description="Ticker from GET /lab/assets (or send a calibration)")
    calibration: CalibrationBundle | None = Field(None, description="The visitor's own calibration, in place of asset")
    model: LabModelName = Field("heston", description="The model the book is priced, hedged and simulated with")
    rho: float | None = Field(
        None, ge=-0.95, le=0.0,
        description="Spot-vol correlation assumption; null keeps the estimated value",
    )
    scenario: ScenarioSpec | None = Field(None, description="Counterfactual world; null runs only the factual one")
    option: OptionSpec = OptionSpec()
    cost_bps: float = Field(5.0, ge=0.0, le=50.0, description="Proportional trading cost")
    risk_aversion: float = Field(
        5.0, ge=0.5, le=20.0,
        description="The learned hedger's risk aversion: high stays close to delta, low trades less to save costs",
    )
    seed: int = Field(7, ge=0, le=2**31 - 1)

    @model_validator(mode="after")
    def _one_source(self) -> "HedgeRequest":
        if (self.asset is None) == (self.calibration is None):
            raise ValueError("send exactly one of asset or calibration")
        if self.calibration is not None:
            self.model = self.calibration.model  # a bundle fixes its model
        return self


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
    role: HedgerRole = Field(..., description="none, bs (Black-Scholes at the premium's vol), model, or learned")
    label: str
    note: str = Field(..., description="What the hedger knows and does")
    same_as: str | None = Field(None, description="Another hedger this one equals, trade for trade")


class SamplePath(BaseModel):
    spots: list[float]
    positions: dict[str, list[float]] = Field(..., description="Hedge position held after each step")
    pnl: dict[str, float]


class SimulatedPaths(BaseModel):
    """The first simulated paths of a world, whole: what the lab animates.

    Paths are the simulation's own (random, not picked); the same index is the same
    random draw in every world."""

    spots: list[list[float]] = Field(..., description="Per path, the spot on each day, today first")
    positions: dict[str, list[list[float]]] = Field(..., description="Per hedger and path, shares held after each day's trade")
    pnl: dict[str, list[float]] = Field(..., description="Per hedger, each path's P&L at expiry")
    highlight: int = Field(..., description="The path closest to the median outcome, followed by default")
    liability: list[list[float]] = Field(
        ..., description="Per path, the option sold marked at the premium's implied vol: the premium on day 0, "
                         "the payoff at expiry")
    mtm_pnl: dict[str, list[list[float]]] = Field(
        ..., description="Per hedger and path, the book marked to market each day: the hedge account less the "
                         "option's mark; 0 on day 0, the P&L at expiry")


class Bands(BaseModel):
    """Quantiles across every path of a world, day by day."""

    p5: list[float]
    p25: list[float]
    p50: list[float]
    p75: list[float]
    p95: list[float]


class WorldResult(BaseModel):
    label: str
    description: str
    params: dict[str, float]
    params_hash: str
    hedgers: dict[str, HedgerStats]
    sample_path: SamplePath
    paths: SimulatedPaths
    mtm_bands: dict[str, Bands] = Field(..., description="Per hedger, the marked-to-market book across all paths")


class BookOrigin(BaseModel):
    """Where the book's calibration came from, for the market card."""

    kind: Literal["shipped", "visitor"]
    window: str
    as_of: str
    vol_today: float
    historical_drift: DriftEstimate | None
    method_label: str
    note: str | None
    calibration_hash: str = Field(..., description="The shipped ParamSet's hash, or the visitor bundle's")


class Book(BaseModel):
    asset: str
    model: LabModelName
    origin: BookOrigin
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
    hedgers: list[HedgerInfo] = Field(
        ..., description="In display order: no hedge, Black-Scholes, the model's own, and the learned band where trained")
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
