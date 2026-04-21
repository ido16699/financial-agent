"""Build quarterly point-in-time universe snapshots for adaptive training.

For each quarter-start date from 2010-01-01 to 2019-10-01 (inclusive),
select the top-N tickers by liquidity/volatility/completeness score using
ONLY data strictly before that date, then write:

    config/universe_snapshots/YYYY-MM-DD.json

Each snapshot is the list of tickers to trade/train on for the next quarter.

Usage:
    python -m scripts.build_universe_snapshots --end 2019-12-31 --size 100
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from stochsignal.ingest.snapshot_loader import load_all_prices
from stochsignal.universe.adaptive import select_universe
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


def _quarter_starts(start: str, end: str) -> list[pd.Timestamp]:
    """Return the first Monday of each quarter in [start, end]."""
    starts = pd.date_range(start=start, end=end, freq="QS")
    out = []
    for d in starts:
        # Move to the first Monday on or after this quarter start
        while d.weekday() != 0:
            d = d + pd.Timedelta(days=1)
        out.append(d)
    return out


@click.command()
@click.option("--start", default="2010-01-01", show_default=True)
@click.option("--end", default="2019-12-31", show_default=True,
              help="Must be <= 2019-12-31 for training data (no look-ahead).")
@click.option("--size", default=100, type=int, show_default=True)
@click.option("--out-dir", default="config/universe_snapshots", show_default=True)
def main(start: str, end: str, size: int, out_dir: str):
    end_ts = pd.Timestamp(end)
    if end_ts > pd.Timestamp("2019-12-31"):
        log.warning("end=%s > 2019-12-31 — ensure this is intentional for OOS testing", end)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log.info("Loading price snapshot...")
    prices = load_all_prices()
    log.info("Loaded %d tickers from snapshot", len(prices))

    quarters = _quarter_starts(start, end)
    log.info("Building %d quarterly universe snapshots", len(quarters))

    manifest = {}
    for as_of in quarters:
        # STRICT: never use data >= as_of
        if as_of > end_ts:
            continue
        tickers = select_universe(as_of, prices, size=size)
        if not tickers:
            log.warning("Empty universe for %s", as_of.date())
            continue

        snapshot_file = out_path / f"{as_of.date()}.json"
        snapshot_file.write_text(json.dumps({
            "as_of": str(as_of.date()),
            "size": len(tickers),
            "tickers": tickers,
        }, indent=2))
        manifest[str(as_of.date())] = tickers
        log.info("%s: %d tickers (first 5: %s)",
                 as_of.date(), len(tickers), tickers[:5])

    # Save a combined manifest for easy loading
    manifest_file = out_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    log.info("Saved manifest to %s", manifest_file)


if __name__ == "__main__":
    main()
