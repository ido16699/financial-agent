"""Fetch and cache daily OHLCV price data via yfinance.

Point-in-time discipline: get_price_history(ticker, as_of) never returns
data for dates after `as_of`. This is the main guard against look-ahead
leakage in the backtest harness.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from stochsignal.config import settings
from stochsignal.ingest import cache
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


def get_price_history(
    ticker: str,
    as_of: pd.Timestamp | str | None = None,
    window_days: int | None = None,
) -> pd.DataFrame:
    """Return daily close prices for `ticker` up to and including `as_of`.

    Parameters
    ----------
    ticker:
        Ticker symbol (e.g. "AAPL", "TEVA.TA").
    as_of:
        Inclusive upper date bound. Defaults to today.
    window_days:
        How many trading days of history to return. Defaults to
        settings.calibration_window_days.

    Returns
    -------
    DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume.
    Sorted ascending by date.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    window_days = window_days or settings.calibration_window_days

    cache_key = f"prices_{ticker}_{as_of.date()}_{window_days}"
    cached = cache.get(cache_key)
    if cached is not None:
        log.debug("Price cache hit: %s", cache_key)
        return cached

    # Fetch a bit more than needed to guarantee enough trading days
    fetch_start = as_of - pd.Timedelta(days=int(window_days * 1.6))
    log.info("Fetching prices: %s  %s → %s", ticker, fetch_start.date(), as_of.date())

    df = yf.download(
        ticker,
        start=fetch_start.strftime("%Y-%m-%d"),
        end=(as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),  # yfinance end is exclusive
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(f"No price data returned for {ticker} up to {as_of.date()}")

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index()

    # Point-in-time: drop anything after as_of
    df = df[df.index <= as_of]

    # Keep last `window_days` trading days
    df = df.tail(window_days)

    cache.set(cache_key, df, ttl_seconds=settings.price_ttl)
    return df


def get_log_returns(
    ticker: str,
    as_of: pd.Timestamp | str | None = None,
    window_days: int | None = None,
) -> pd.Series:
    """Return daily log-returns series for `ticker` up to `as_of`."""
    import numpy as np

    prices = get_price_history(ticker, as_of=as_of, window_days=window_days)
    closes = prices["Close"].dropna()
    return pd.Series(
        np.log(closes.values[1:] / closes.values[:-1]),
        index=closes.index[1:],
        name=ticker,
    )
