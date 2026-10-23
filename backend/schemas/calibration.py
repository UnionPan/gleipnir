"""
Schemas for calibration endpoints
"""
from pydantic import BaseModel, Field
from typing import Any


class FetchDataRequest(BaseModel):
    """Request to fetch OHLCV data"""
    tickers: list[str] = Field(..., min_length=1, description="List of ticker symbols")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


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
    tickers: list[str] = Field(..., min_length=1, description="List of ticker symbols")
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


class CalibrationDiagnostics(BaseModel):
    """Calibration diagnostics"""
    logLikelihood: float | None = None
    aic: float | None = None
    bic: float | None = None
    mean_ess: float | None = None
    n_particles: int | None = None
    variogram_r2: float | None = None
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
    risk_free_rate: float = Field(default=0.05, description="Risk-free rate")
    filter_params: dict[str, Any] | None = Field(
        default=None,
        description="Optional filtering: min_volume, min_open_interest, max_spread_pct, moneyness_range (list)"
    )
    calibration_method: str = Field(default="differential_evolution", description="Optimization method")
    maxiter: int = Field(default=1000, description="Maximum iterations")


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
    n_strikes: int = Field(default=30, description="Number of strike points")
    n_maturities: int = Field(default=20, description="Number of maturity points")
    strike_range: list[float] = Field(default=[0.7, 1.3], description="Strike range as [min_moneyness, max_moneyness]")
    maturity_range: list[float] = Field(default=[0.05, 2.0], description="Maturity range in years [min, max]")


class VolSurfaceResponse(BaseModel):
    """Volatility surface data"""
    strikes: list[float]
    maturities: list[float]
    vols: list[list[float | None]]  # maturities x strikes; None where the model vol is undefined
    surface_type: str  # 'implied_vol' or 'local_vol'
    ticker: str
    model: str  # Distinguish from P-measure results
