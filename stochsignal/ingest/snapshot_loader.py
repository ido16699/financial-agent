"""Load the frozen price snapshot parquet produced by scripts.snapshot_prices.

Returns a dict[ticker -> DataFrame indexed by date] matching the shape that
the rest of the codebase expects from get_price_history().
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_SNAPSHOT = Path("data/prices_snapshot.parquet")


@lru_cache(maxsize=1)
def load_all_prices(path: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """Load the parquet snapshot into a dict[ticker -> DataFrame].

    Each DataFrame is indexed by Date and has columns Open/High/Low/Close/Volume.
    Cached so repeated calls are free.
    """
    p = Path(path) if path else DEFAULT_SNAPSHOT
    if not p.exists():
        raise FileNotFoundError(
            f"Price snapshot not found at {p}. Run: python -m scripts.snapshot_prices"
        )
    df = pd.read_parquet(p)
    out: dict[str, pd.DataFrame] = {}
    for ticker, group in df.groupby("ticker"):
        sub = group.drop(columns=["ticker"]).copy()
        # Ensure Date is the index
        if "Date" in sub.columns:
            sub["Date"] = pd.to_datetime(sub["Date"])
            sub = sub.set_index("Date").sort_index()
        out[ticker] = sub
    return out


def get_prices(ticker: str, as_of: Optional[str] = None,
               path: Optional[str] = None) -> pd.DataFrame:
    """Return price history for one ticker, optionally truncated at `as_of`."""
    all_p = load_all_prices(path)
    if ticker not in all_p:
        raise KeyError(f"Ticker {ticker} not in price snapshot")
    df = all_p[ticker]
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        df = df[df.index <= cutoff]
    return df
