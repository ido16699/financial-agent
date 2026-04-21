"""Point-in-time adaptive universe selection.

Given an `as_of` date, score every ticker in the seed pool using ONLY data
strictly before that date, then return the top N by composite score:

    score = liquidity_rank * 0.6 + vol_rank * 0.2 + completeness_rank * 0.2

where:
    - liquidity = 60-day average dollar volume
    - vol       = 252-day realized annualized volatility
                  (penalized at extremes — too low = untradeable,
                  too high = noise)
    - completeness = fraction of expected trading days present over
                     the last 252 days

Tickers with <252 days of history before `as_of` are excluded.
This naturally filters out pre-IPO and post-delisting periods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from stochsignal.universe.seed_pool import SEED_POOL


@dataclass
class TickerScore:
    ticker: str
    dollar_volume: float
    realized_vol: float
    completeness: float
    composite: float


def _score_one(
    ticker: str,
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = 252,
    vol_lookback: int = 252,
    dv_lookback: int = 60,
) -> Optional[TickerScore]:
    """Score one ticker strictly using data < as_of."""
    if prices is None or prices.empty:
        return None

    pit = prices[prices.index < as_of]
    if len(pit) < lookback_days:
        return None

    closes = pit["Close"].dropna()
    if "Volume" in pit.columns:
        vols = pit["Volume"].fillna(0)
    else:
        vols = pd.Series(0, index=pit.index)

    if len(closes) < lookback_days:
        return None

    # 60-day avg dollar volume
    last_closes = closes.tail(dv_lookback)
    last_vols = vols.reindex(last_closes.index).fillna(0)
    dollar_vol = float((last_closes * last_vols).mean())

    # 252-day realized vol (annualized)
    log_ret = np.log(closes.values[1:] / closes.values[:-1])
    if len(log_ret) < vol_lookback:
        return None
    realized_vol = float(np.std(log_ret[-vol_lookback:]) * np.sqrt(252))

    # Completeness: trading days present vs expected (~252 in a year)
    window_start = as_of - pd.Timedelta(days=365)
    window_closes = closes[(closes.index >= window_start) & (closes.index < as_of)]
    completeness = len(window_closes) / 252.0
    completeness = float(min(completeness, 1.0))

    return TickerScore(
        ticker=ticker,
        dollar_volume=dollar_vol,
        realized_vol=realized_vol,
        completeness=completeness,
        composite=0.0,  # filled later after ranking
    )


def _vol_penalty(vol: float) -> float:
    """Penalize realized vol at extremes. Returns score in [0, 1].

    Sweet spot: ~20%-50% annualized vol.
    - vol < 0.1: too illiquid / not moving → score 0.3
    - 0.20 <= vol <= 0.50: ideal → score 1.0
    - vol > 1.0: way too noisy → score 0.2
    """
    if vol < 0.10:
        return 0.3
    if vol < 0.20:
        return 0.5 + (vol - 0.10) * 5  # 0.5 → 1.0
    if vol <= 0.50:
        return 1.0
    if vol <= 1.00:
        return 1.0 - (vol - 0.50) * 1.6  # 1.0 → 0.2
    return 0.2


def select_universe(
    as_of: pd.Timestamp | str,
    prices: dict[str, pd.DataFrame],
    size: int = 100,
    pool: Optional[list[str]] = None,
    min_dollar_volume: float = 1e7,  # $10M/day minimum
) -> list[str]:
    """Return top `size` tickers by point-in-time composite score.

    Args:
        as_of: cutoff date — only price data strictly before this is used
        prices: mapping ticker -> DataFrame with Close, Volume columns (full
                history; filtering is done here)
        size: number of tickers to return (default 100)
        pool: candidate tickers to consider (defaults to full SEED_POOL)
        min_dollar_volume: reject tickers below this avg daily $-volume

    Returns:
        List of selected tickers, up to `size` (may be smaller if pool runs out).
    """
    as_of = pd.Timestamp(as_of)
    pool = pool if pool is not None else SEED_POOL

    scores: list[TickerScore] = []
    for ticker in pool:
        df = prices.get(ticker)
        s = _score_one(ticker, df, as_of)
        if s is None:
            continue
        if s.dollar_volume < min_dollar_volume:
            continue
        scores.append(s)

    if not scores:
        return []

    # Rank each dimension in [0,1] (higher = better)
    dvs = np.array([s.dollar_volume for s in scores])
    dv_rank = dvs.argsort().argsort() / max(len(scores) - 1, 1)

    vol_scores = np.array([_vol_penalty(s.realized_vol) for s in scores])

    comp_scores = np.array([s.completeness for s in scores])

    composite = dv_rank * 0.6 + vol_scores * 0.2 + comp_scores * 0.2
    for s, c in zip(scores, composite):
        s.composite = float(c)

    scores.sort(key=lambda s: s.composite, reverse=True)
    return [s.ticker for s in scores[:size]]
