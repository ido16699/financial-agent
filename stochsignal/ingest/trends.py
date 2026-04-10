"""Fetch and cache Google Trends interest-over-time, returned as a z-score.

Uses pytrends with exponential back-off to respect rate limits.
Returns z-score of the most recent weekly interest value relative to the
lookback window, so it is dimensionless and comparable across tickers.

Point-in-time note: pytrends data is not natively point-in-time. Like news,
backtest harness skips trends for historical windows; trends are only used
in forward (live) runs.
"""

from __future__ import annotations

import datetime
import time

import numpy as np
import pandas as pd
from pytrends.request import TrendReq

from stochsignal.config import settings
from stochsignal.ingest import cache
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


def _build_query(ticker: str) -> str:
    """Map ticker → human-readable search term for Google Trends."""
    # Strip exchange suffix (.TA) and use the ticker itself as the search term.
    # For well-known tickers this works fine; can be overridden in config later.
    return ticker.split(".")[0]


def fetch_trends_zscore(ticker: str) -> float:
    """Return z-score of the latest Google Trends interest for `ticker`.

    Z-score = (latest_value − mean) / std over the lookback window.
    Returns 0.0 on failure (graceful degradation).
    """
    cache_key = f"trends_{ticker}_{datetime.date.today()}"
    cached = cache.get(cache_key)
    if cached is not None:
        log.debug("Trends cache hit: %s", ticker)
        return cached

    query = _build_query(ticker)
    lookback = settings.trends_lookback_days

    today = datetime.date.today()
    start = today - datetime.timedelta(days=lookback)
    timeframe = f"{start} {today}"

    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 30))

    for attempt in range(3):
        try:
            pytrends.build_payload([query], timeframe=timeframe)
            df: pd.DataFrame = pytrends.interest_over_time()
            break
        except Exception as exc:
            wait = 2 ** attempt * 5
            log.warning("pytrends attempt %d failed for %s: %s — retrying in %ds", attempt + 1, ticker, exc, wait)
            time.sleep(wait)
    else:
        log.error("All pytrends attempts failed for %s — trends_zscore=0.0", ticker)
        score = 0.0
        cache.set(cache_key, score, ttl_seconds=settings.trends_ttl)
        return score

    if df.empty or query not in df.columns:
        log.warning("Empty trends data for %s — zscore=0.0", ticker)
        score = 0.0
    else:
        series = df[query].astype(float)
        if series.std() < 1e-9:
            score = 0.0
        else:
            latest = series.iloc[-1]
            score = float((latest - series.mean()) / series.std())
        log.info("Trends z-score %s: %.3f  (latest=%d)", ticker, score, int(series.iloc[-1]))

    cache.set(cache_key, score, ttl_seconds=settings.trends_ttl)
    return score
