"""Train the model on historical data.

Usage (local):
    python -m scripts.train_model

Usage (Colab / 100-stock adaptive universe, 2010-2019):
    python -m scripts.train_model \\
      --start 2010-01-01 --end 2019-12-31 \\
      --snapshot data/prices_snapshot.parquet \\
      --universe-dir config/universe_snapshots
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from stochsignal.model.train import train


@click.command()
@click.option("--start", default="2010-01-01", show_default=True)
@click.option("--end", default="2019-12-31", show_default=True,
              help="Must be <= 2019-12-31 for strict training (no look-ahead).")
@click.option("--tickers", default=None, help="Comma-separated tickers override.")
@click.option("--snapshot", default=None,
              help="Path to prices_snapshot.parquet (skips yfinance).")
@click.option("--universe-dir", default=None,
              help="Path to config/universe_snapshots/ for adaptive universe.")
def main(start: str, end: str, tickers: str | None,
         snapshot: str | None, universe_dir: str | None) -> None:
    """Train signal weights from historical backtest data."""
    ticker_list = [t.strip() for t in tickers.split(",")] if tickers else None

    # Load price snapshot if provided
    prices_override = None
    spy_override = None
    if snapshot:
        from stochsignal.ingest.snapshot_loader import load_all_prices
        prices_override = load_all_prices(snapshot)
        spy_override = prices_override.get("SPY")
        print(f"Loaded {len(prices_override)} tickers from snapshot {snapshot}")

    # Load universe snapshots if provided
    universe_snapshots = None
    if universe_dir:
        manifest = Path(universe_dir) / "manifest.json"
        if manifest.exists():
            universe_snapshots = json.loads(manifest.read_text())
            # Expand tickers to union of all snapshots
            ticker_list = sorted({t for ts in universe_snapshots.values() for t in ts})
            print(f"Adaptive universe: {len(universe_snapshots)} snapshot dates, "
                  f"{len(ticker_list)} unique tickers over training period")

    print(f"\n{'='*60}")
    print(f"  StochSignal — Training Phase")
    print(f"  Data: {start} → {end}")
    if prices_override is not None:
        print(f"  Source: snapshot parquet ({len(prices_override)} tickers)")
    if universe_snapshots:
        print(f"  Universe: adaptive ({len(universe_snapshots)} quarterly snapshots)")
    print(f"{'='*60}\n")

    weights = train(
        tickers=ticker_list, start=start, end=end, save=True,
        prices_override=prices_override,
        universe_snapshots=universe_snapshots,
        spy_override=spy_override,
    )

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")
    print(f"  Samples          : {weights.n_training_samples}")
    print(f"  Hit rate         : {weights.training_hit_rate*100:.1f}%")
    print(f"  Optimal threshold: {weights.optimal_confidence_threshold:.2f}")
    print(f"  Avg weekly win   : {weights.avg_weekly_win*100:.2f}%")
    print(f"  Avg weekly loss  : {weights.avg_weekly_loss*100:.2f}%")
    print(f"  ---")
    print(f"  Learned betas:")
    print(f"    intercept  : {weights.beta_intercept:.4f}")
    print(f"    GBM prob   : {weights.beta_gbm_prob:.4f}")
    print(f"    sector     : {weights.beta_sector:.4f}")
    print(f"    wave       : {weights.beta_wave:.4f}")
    print(f"    regime     : {weights.beta_regime:.4f}")
    print(f"    market_mom : {weights.beta_market_mom:.4f}")
    print(f"  ---")
    print(f"  Learned risk params (from validation {weights.validation_start} → {weights.validation_end}):")
    print(f"    stop_loss      : {weights.learned_stop_loss_pct*100:.2f}%")
    print(f"    kelly_fraction : {weights.learned_kelly_fraction:.2f}")
    print(f"    max_allocation : {weights.learned_max_allocation*100:.0f}%")
    print(f"    max_position   : {weights.learned_max_position_pct*100:.0f}%")
    print(f"  Learned signal weights:")
    print(f"    market_mom : {weights.w_market_mom:.2f}")
    print(f"    wave       : {weights.w_wave:.2f}")
    print(f"    sector     : {weights.w_sector:.2f}")
    print(f"    gbm        : {weights.w_gbm:.2f}")
    print(f"    interfere  : {weights.w_interference:.2f}")
    print(f"  Validation return: {weights.validation_return_pct:+.2f}%")
    print(f"  ---")
    print(f"  Weights saved to config/trained_weights.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
