"""Market data schemas."""
from pydantic import BaseModel


class QuoteData(BaseModel):
    """Individual quote data."""
    symbol: str
    name: str
    last_price: float | None = None
    change: float | None = None
    change_percent: float | None = None
    volume: int | None = None


class HistoricalDataPoint(BaseModel):
    """Single point in historical data."""
    date: str
    close: float


class OHLCDataPoint(BaseModel):
    """OHLC data point."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class IndexChartData(BaseModel):
    """Chart data for an index."""
    symbol: str
    name: str
    data: list[HistoricalDataPoint]


class OHLCChartResponse(BaseModel):
    """OHLC chart data response."""
    symbol: str
    data: list[OHLCDataPoint]


class MarketOverviewResponse(BaseModel):
    """Market overview with indices, stocks, and commodities."""
    indices: list[QuoteData]
    magnificent7: list[QuoteData]
    commodities: list[QuoteData]


class IndexChartsResponse(BaseModel):
    """Historical chart data for indices."""
    charts: list[IndexChartData]
