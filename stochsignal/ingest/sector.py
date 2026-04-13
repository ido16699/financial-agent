"""Fetch sector/index momentum for neighbourhood correlation.

Each ticker is mapped to a sector ETF/index. The sector momentum is the
rolling log-return of the sector over the last `lookback` trading days,
normalised as a z-score over the calibration window.

This captures the idea that stocks move with their sector: if QQQ is up,
AAPL/NVDA are more likely to go up too.

Sector mapping:
  US Large-Cap  → SPY
  Nasdaq Tech   → QQQ
  TASE          → ^TA35.TA  (Tel Aviv 35 index)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stochsignal.config import settings, watchlist
from stochsignal.ingest.prices import get_price_history
from stochsignal.ingest import cache
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)

# Ticker → sector ETF/index mapping
SECTOR_INDEX = {
    "US Large-Cap": "SPY",
    "Nasdaq Tech": "QQQ",
    "TASE": "TA35.TA",
}

# Lookback for sector momentum (trading days)
SECTOR_MOMENTUM_LOOKBACK = 20  # ~1 month


def get_sector_index(ticker: str) -> str:
    """Return the sector ETF/index ticker for a given stock ticker."""
    group = watchlist.group_of(ticker)
    return SECTOR_INDEX.get(group, "SPY")


def fetch_sector_momentum(
    ticker: str,
    as_of: pd.Timestamp | str | None = None,
) -> float:
    """Return sector momentum z-score for the given ticker's sector.

    Sector momentum = recent N-day log-return of the sector index,
    normalised as z-score over the full calibration window.

    Returns 0.0 on failure.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    sector_ticker = get_sector_index(ticker)

    cache_key = f"sector_{sector_ticker}_{as_of.date()}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        prices = get_price_history(
            sector_ticker, as_of=as_of, window_days=settings.calibration_window_days,
        )
    except Exception as exc:
        log.warning("Could not fetch sector index %s: %s", sector_ticker, exc)
        return 0.0

    closes = prices["Close"].dropna()
    if len(closes) < SECTOR_MOMENTUM_LOOKBACK + 10:
        log.warning("Too little sector data for %s", sector_ticker)
        return 0.0

    # Rolling N-day log-returns over the full window
    log_rets_N = np.log(closes.values[SECTOR_MOMENTUM_LOOKBACK:] / closes.values[:-SECTOR_MOMENTUM_LOOKBACK])

    if len(log_rets_N) < 2 or np.std(log_rets_N) < 1e-9:
        score = 0.0
    else:
        latest = log_rets_N[-1]
        score = float((latest - np.mean(log_rets_N)) / np.std(log_rets_N))

    log.info(
        "Sector momentum %s (→%s): z=%.3f",
        ticker, sector_ticker, score,
    )
    cache.set(cache_key, score, ttl_seconds=settings.price_ttl)
    return score
