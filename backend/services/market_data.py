"""Market-data boundary shared by the services: typed failures and unit conventions."""
import logging
from collections.abc import Mapping

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
