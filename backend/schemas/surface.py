"""Volatility surface request/response schemas."""
from pydantic import BaseModel, Field


class VolSurfacePoint(BaseModel):
    """Single point on volatility surface."""

    strike: float
    expiry: float = Field(..., description="Time to expiration in years")
    implied_vol: float
    moneyness: float | None = Field(None, description="Strike/Spot ratio")


class VolSurfaceRequest(BaseModel):
    """Request to build volatility surface."""

    symbol: str = Field(..., description="Underlying ticker symbol")
    spot_price: float | None = Field(None, description="Current spot price. If None, fetches live.")
    min_expiry_days: int = Field(default=7, ge=0, le=730, description="Minimum days to expiration")
    max_expiry_days: int = Field(default=365, ge=1, le=1095, description="Maximum days to expiration")
    interpolate: bool = Field(default=False, description="Whether to interpolate the surface")
    grid_size: int = Field(default=30, ge=5, le=60, description="Grid size for interpolation")


class VolSurfaceResponse(BaseModel):
    """Response with volatility surface data."""

    symbol: str
    spot_price: float
    surface_points: list[VolSurfacePoint]
    num_expirations: int
    num_strikes: int


class VolSmileRequest(BaseModel):
    """Request for volatility smile at specific expiry."""

    symbol: str
    expiration_date: str = Field(..., description="Target expiration date (YYYY-MM-DD)")


class VolSmileResponse(BaseModel):
    """Response with volatility smile."""

    symbol: str
    expiration_date: str
    time_to_expiry: float
    spot_price: float
    strikes: list[float]
    implied_vols: list[float]
