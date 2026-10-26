"""Market-data boundary shared by the services: typed failures and unit conventions."""
import logging
import math
from collections.abc import Mapping
from datetime import date, timedelta
from functools import lru_cache

import yfinance as yf
from yfinance.exceptions import YFException

logger = logging.getLogger(__name__)


class MarketDataError(RuntimeError):
    """Upstream market data could not be fetched or parsed (served as 502)."""


# What a yfinance call raises for unavailable or malformed data. Network errors
# (requests and curl_cffi) are OSError subclasses. Programming errors such as
# AttributeError are deliberately absent so that bugs are not reported as data issues.
FETCH_ERRORS = (YFException, OSError, ValueError, KeyError, IndexError, TypeError)


def dividend_yield(info: Mapping) -> float:
    """Annual dividend yield as a decimal, from a yfinance ``Ticker.info`` mapping.

    yfinance reports ``dividendYield`` in percent (0.31 means 0.31%), so it is not
    used. The indicated yield ``dividendRate / price`` is preferred; the trailing
    yield (a decimal) is the fallback; a non-payer yields 0.
    """
    rate = info.get("dividendRate")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if rate and price:
        return float(rate) / float(price)
    trailing = info.get("trailingAnnualDividendYield")
    return float(trailing) if trailing else 0.0


def fetch_dividend_yield(ticker: str) -> float:
    """Dividend yield for ``ticker``; 0 (logged) when yfinance cannot provide it."""
    try:
        info = yf.Ticker(ticker).info
    except FETCH_ERRORS as exc:
        logger.warning("dividend yield for %s unavailable (%s); assuming 0", ticker, exc)
        return 0.0
    return dividend_yield(info)


@lru_cache(maxsize=4096)
def _daily_dividend_yield(ticker: str, day: str) -> float:
    return fetch_dividend_yield(ticker)


def todays_dividend_yield(ticker: str) -> float:
    """``fetch_dividend_yield``, fetched once a day per ticker."""
    return _daily_dividend_yield(ticker.upper(), date.today().isoformat())


@lru_cache(maxsize=16)
def _treasury_bill_rate(day: str) -> float:
    on = date.fromisoformat(day)
    history = yf.Ticker("^IRX").history(start=(on - timedelta(days=10)).isoformat(),
                                        end=(on + timedelta(days=1)).isoformat())
    if history.empty:
        raise ValueError(f"no ^IRX quote near {day}")
    return math.log1p(float(history["Close"].iloc[-1]) / 100.0)


def treasury_bill_rate(default: float) -> float:
    """Today's 13-week Treasury bill yield (^IRX), continuously compounded and
    fetched once a day; ``default`` (logged) when it cannot be fetched."""
    try:
        return _treasury_bill_rate(date.today().isoformat())
    except FETCH_ERRORS as exc:
        logger.warning("13-week T-bill yield unavailable (%s); using %.4f", exc, default)
        return default
