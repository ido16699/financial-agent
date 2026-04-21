"""Bulk-download prices for the full seed pool into a single parquet file.

Run once (locally or in Colab). Output: data/prices_snapshot.parquet
Each row: (ticker, date, Open, High, Low, Close, Volume, Adj Close)

This lets training/backtest skip yfinance and stay reproducible across sessions.

Usage:
    python -m scripts.snapshot_prices --end 2019-12-31 --out data/prices_snapshot.parquet
"""
from __future__ import annotations

import os
from pathlib import Path

import click
import pandas as pd
import yfinance as yf

from stochsignal.universe.seed_pool import SEED_POOL
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


@click.command()
@click.option("--start", default="2000-01-01", show_default=True)
@click.option("--end", default="2019-12-31", show_default=True,
              help="Last date to fetch (inclusive). Keep <= 2019-12-31 for training data.")
@click.option("--out", default="data/prices_snapshot.parquet", show_default=True)
@click.option("--batch-size", default=20, type=int, show_default=True)
def main(start: str, end: str, out: str, batch_size: int):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tickers = SEED_POOL
    log.info("Downloading %d tickers: %s -> %s", len(tickers), start, end)

    all_frames = []
    failed = []

    # Fetch in batches (yfinance supports multi-ticker download)
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        log.info("Batch %d/%d: %s", i // batch_size + 1,
                 (len(tickers) + batch_size - 1) // batch_size, batch)
        try:
            data = yf.download(
                tickers=" ".join(batch),
                start=start,
                end=end_inclusive,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as e:
            log.error("Batch download failed: %s", e)
            failed.extend(batch)
            continue

        if data is None or data.empty:
            failed.extend(batch)
            continue

        # Reshape: yfinance multi-ticker returns MultiIndex columns (ticker, field)
        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = data.copy()
                else:
                    if ticker not in data.columns.get_level_values(0):
                        failed.append(ticker)
                        continue
                    df = data[ticker].copy()
                df = df.dropna(how="all")
                if df.empty:
                    failed.append(ticker)
                    continue
                df["ticker"] = ticker
                df = df.reset_index()
                df = df.rename(columns={"index": "Date"})
                all_frames.append(df)
            except Exception as e:
                log.warning("Failed to parse %s: %s", ticker, e)
                failed.append(ticker)

    if not all_frames:
        log.error("No data fetched!")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    # Normalize column names
    combined.columns = [str(c) for c in combined.columns]
    if "Date" not in combined.columns:
        # yfinance sometimes names it "Datetime"
        for candidate in ("Datetime", "date"):
            if candidate in combined.columns:
                combined = combined.rename(columns={candidate: "Date"})
                break

    combined.to_parquet(out_path, index=False)
    log.info("Saved %d rows to %s", len(combined), out_path)
    if failed:
        log.warning("Failed tickers (%d): %s", len(failed), failed)


if __name__ == "__main__":
    main()
