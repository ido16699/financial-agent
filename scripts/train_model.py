"""Train the model on historical data (2020-2025).

Usage:
    python -m scripts.train_model
    python -m scripts.train_model --start 2020-01-01 --end 2025-12-31
"""

from __future__ import annotations

import click

from stochsignal.model.train import train
from stochsignal.config import watchlist


@click.command()
@click.option("--start", default="2020-01-01", show_default=True)
@click.option("--end", default="2025-12-31", show_default=True)
@click.option("--tickers", default=None, help="Comma-separated tickers. Default: full watchlist.")
def main(start: str, end: str, tickers: str | None) -> None:
    """Train signal weights from historical backtest data."""
    ticker_list = [t.strip() for t in tickers.split(",")] if tickers else None

    print(f"\n{'='*60}")
    print(f"  StochSignal — Training Phase")
    print(f"  Data: {start} → {end}")
    print(f"{'='*60}\n")

    weights = train(tickers=ticker_list, start=start, end=end, save=True)

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
    print(f"  Validation return: {weights.validation_return_pct:+.2f}%")
    print(f"  ---")
    print(f"  Weights saved to config/trained_weights.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
