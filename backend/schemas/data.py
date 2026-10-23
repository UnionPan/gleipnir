"""Data fetching request/response schemas."""
from pydantic import BaseModel, Field
from datetime import datetime


class OptionChainRequest(BaseModel):
    """Request for option chain data."""

    symbol: str = Field(..., description="Underlying ticker symbol (e.g., 'AAPL')")
    expiration_date: str | None = Field(
        None, description="Specific expiration date (YYYY-MM-DD). If None, gets nearest."
    )


class OptionContractData(BaseModel):
    """Single option contract data."""

    strike: float
    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    implied_volatility: float | None = None
    option_type: str


class OptionChainResponse(BaseModel):
    """Response with option chain data."""

    symbol: str
    expiration_date: str
    spot_price: float
    contracts: list[OptionContractData]
    fetched_at: datetime = Field(default_factory=datetime.now)


class AvailableExpirationsResponse(BaseModel):
    """Response with available expiration dates."""

    symbol: str
    expirations: list[str] = Field(..., description="List of available expiration dates")
