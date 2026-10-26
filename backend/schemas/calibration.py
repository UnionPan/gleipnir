"""
Schemas for calibration endpoints

Every size a request controls is capped here (see backend/core/compute_guard.py).
"""
from datetime import date
from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator

from backend.schemas.lab import CalibrationBundle

MAX_TICKERS = 5
MAX_HISTORY_DAYS = 3653  # 10 years
MAX_OPTIMIZER_ITERATIONS = 500
MAX_SURFACE_STRIKES = 60
MAX_SURFACE_MATURITIES = 40
MAX_SURFACE_MATURITY_YEARS = 5.0
Tickers = Annotated[list[str], Field(min_length=1, max_length=MAX_TICKERS, description="Ticker symbols")]


def _check_window(start: str, end: str) -> None:
    try:
        first, last = date.fromisoformat(start), date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError(f"dates must be YYYY-MM-DD: {exc}") from exc
    if first >= last:
        raise ValueError("start_date must be before end_date")
    if (last - first).days > MAX_HISTORY_DAYS:
        raise ValueError("the history window is limited to 10 years")


class FetchDataRequest(BaseModel):
    """Request to fetch OHLCV data"""
    tickers: Tickers
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")

    @model_validator(mode="after")
    def _window(self) -> "FetchDataRequest":
        _check_window(self.start_date, self.end_date)
        return self


class OHLCVData(BaseModel):
    """OHLCV data for a single ticker"""
    ticker: str
    dates: list[str]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]


class CalibrationRequest(BaseModel):
    """Request to calibrate a model"""
    tickers: Tickers
    model: str = Field(
        ...,
        description=(
            "Model name (gbm, ou, heston, rough_bergomi, regime_switching_gbm, "
            "merton_jump, garch)"
        ),
    )
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    method: str = Field(
        default="mle",
        description=(
            "Calibration method:\n"
            "- 'mle': Maximum Likelihood (for gbm, ou)\n"
            "- 'particle_filter': Particle Filter (for heston, rough_bergomi)\n"
            "- 'moment_matching': Moment Matching (for rough_bergomi)\n"
            "- 'em': EM Algorithm (for regime_switching_gbm)\n"
            "- 'mle': Maximum Likelihood (for merton_jump)\n"
            "- 'mle': Quasi-MLE (for garch)"
        )
    )
    include_drift: bool = Field(default=True, description="Whether to include drift parameter")

    @model_validator(mode="after")
    def _window(self) -> "CalibrationRequest":
        _check_window(self.start_date, self.end_date)
        return self


class HistoricalDrift(BaseModel):
    """The window's drift: an educational estimate, never simulated as a forecast."""
    estimate: float = Field(..., description="Annualized arithmetic drift")
    std_error: float
    ci_low: float = Field(..., description="95% interval, lower end")
    ci_high: float = Field(..., description="95% interval, upper end")


class CalibrationDiagnostics(BaseModel):
    """Calibration diagnostics. Non-finite values are reported as null."""
    method_label: str | None = Field(None, description="The estimator, in words")
    converged: bool | None = None
    logLikelihood: float | None = None
    aic: float | None = None
    bic: float | None = None
    variogram_r2: float | None = None
    std_errors: dict[str, float | None] | None = Field(None, description="Per parameter")
    bias: dict[str, float | None] | None = Field(None, description="Parametric-bootstrap bias, per parameter")
    derived: dict[str, float | None] | None = Field(None, description="Readable quantities, e.g. long-run vol")
    drift: HistoricalDrift | None = None
    errorMetrics: dict[str, float] | None = None
    note: str | None = None


class CalibrationResult(BaseModel):
    """Result of model calibration"""
    ticker: str
    model: str
    parameters: dict[str, float]
    diagnostics: CalibrationDiagnostics
    timestamp: str
    method: str | None = None
    measure: str | None = Field(default="P-measure")
    lab_bundle: CalibrationBundle | None = Field(
        None, description="The fit as the Hedging Lab takes it; null for models the lab does not run")


# ========== Q-measure (Risk-Neutral) Calibration Schemas ==========

class OptionQuoteData(BaseModel):
    """Single option quote"""
    strike: float
    expiry: str
    option_type: str
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float | None = None
    moneyness: float


class FetchOptionChainRequest(BaseModel):
    """Request to fetch option chain data"""
    ticker: str = Field(..., description="Ticker symbol")
    reference_date: str | None = Field(None, description="Reference date (YYYY-MM-DD), defaults to today")
    risk_free_rate: float = Field(default=0.05, description="Risk-free rate (e.g., 0.05 for 5%)")
    expiry: str | None = Field(None, description="Specific expiry date (YYYY-MM-DD) or None for all")


class OptionChainData(BaseModel):
    """Option chain data for a ticker"""
    ticker: str
    spot_price: float
    reference_date: str
    risk_free_rate: float
    dividend_yield: float
    options: list[OptionQuoteData]
    expiries: list[str]
    n_options: int


class QMeasureCalibrationRequest(BaseModel):
    """Request to calibrate Q-measure model from option chain"""
    ticker: str = Field(..., description="Ticker symbol")
    model: str = Field(..., description="Model name (heston)")
    reference_date: str | None = Field(None, description="Reference date (YYYY-MM-DD)")
    risk_free_rate: float = Field(default=0.05, ge=-0.05, le=0.25, description="Risk-free rate")
    filter_params: dict[str, Any] | None = Field(
        default=None,
        description="Optional filtering: min_volume, min_open_interest, max_spread_pct, moneyness_range (list)"
    )
    calibration_method: str = Field(default="differential_evolution", description="Optimization method")
    maxiter: int = Field(default=200, ge=1, le=MAX_OPTIMIZER_ITERATIONS, description="Maximum iterations")


class QMeasureCalibrationResult(BaseModel):
    """Result of Q-measure calibration"""
    ticker: str
    model: str
    parameters: dict[str, float]
    diagnostics: dict[str, Any]
    timestamp: str
    measure: str = "Q-measure"


class VolSurfaceRequest(BaseModel):
    """Request to generate volatility surface"""
    ticker: str
    model: str
    parameters: dict[str, float]
    spot_price: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    n_strikes: int = Field(default=30, ge=2, le=MAX_SURFACE_STRIKES, description="Number of strike points")
    n_maturities: int = Field(default=20, ge=2, le=MAX_SURFACE_MATURITIES, description="Number of maturity points")
    strike_range: list[float] = Field(default=[0.7, 1.3], min_length=2, max_length=2,
                                      description="Strike range as [min_moneyness, max_moneyness]")
    maturity_range: list[float] = Field(default=[0.05, 2.0], min_length=2, max_length=2,
                                        description="Maturity range in years [min, max]")

    @model_validator(mode="after")
    def _ranges(self) -> "VolSurfaceRequest":
        lo, hi = self.strike_range
        if not 0.1 <= lo < hi <= 5.0:
            raise ValueError("strike_range must satisfy 0.1 <= min < max <= 5")
        lo, hi = self.maturity_range
        if not 0.0 < lo < hi <= MAX_SURFACE_MATURITY_YEARS:
            raise ValueError("maturity_range must satisfy 0 < min < max <= 5 years")
        return self


class VolSurfaceResponse(BaseModel):
    """Volatility surface data"""
    strikes: list[float]
    maturities: list[float]
    vols: list[list[float | None]]  # maturities x strikes; None where the model vol is undefined
    surface_type: str  # 'implied_vol' or 'local_vol'
    ticker: str
    model: str  # Distinguish from P-measure results
