"""Debug script: calibrate and forecast a single ticker, print details.

Usage:
    calibrate AAPL
    calibrate TEVA.TA --as-of 2025-06-01
    calibrate NVDA --skip-external
"""

from __future__ import annotations

import click
import pandas as pd

from stochsignal.ingest.prices import get_log_returns, get_price_history
from stochsignal.ingest.news import fetch_news_sentiment
from stochsignal.ingest.trends import fetch_trends_zscore
from stochsignal.ingest.sector import fetch_sector_momentum
from stochsignal.model.gbm import calibrate as gbm_calibrate, prob_up_closed_form
from stochsignal.model.perturbation import compute as compute_forecast
from stochsignal.config import settings
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


@click.command()
@click.argument("ticker")
@click.option("--as-of", "as_of_str", default=None, help="Point-in-time date (YYYY-MM-DD). Default: today.")
@click.option("--skip-external", is_flag=True, help="Skip news/trends, use zeros.")
def main(ticker: str, as_of_str: str | None, skip_external: bool) -> None:
    """Calibrate GBM and show perturbed forecast for TICKER."""
    as_of = pd.Timestamp(as_of_str) if as_of_str else pd.Timestamp.today().normalize()

    print(f"\n{'='*60}")
    print(f"  StochSignal — Single Ticker Debug: {ticker}")
    print(f"  as_of = {as_of.date()}")
    print(f"{'='*60}\n")

    # --- Price history ---
    prices = get_price_history(ticker, as_of=as_of)
    print(f"Price history: {len(prices)} rows  ({prices.index[0].date()} → {prices.index[-1].date()})")
    print(f"  Last close: {prices['Close'].iloc[-1]:.4f}\n")

    # --- Log returns + GBM calibration ---
    log_returns = get_log_returns(ticker, as_of=as_of)
    params = gbm_calibrate(ticker, log_returns)
    p_up_base = prob_up_closed_form(params, settings.forecast_horizon_days)

    print(f"GBM Calibration (n={params.n_obs} daily log-returns):")
    print(f"  mu (ann.) = {params.mu*100:.3f}%")
    print(f"  sigma (ann.) = {params.sigma*100:.3f}%")
    print(f"  P(up) baseline (closed-form) = {p_up_base*100:.1f}%\n")

    # --- External signals ---
    if skip_external:
        sentiment, trends_z, sector_z = 0.0, 0.0, 0.0
        print("External signals: SKIPPED (zeroed)\n")
    else:
        print("Fetching external signals...")
        sentiment = fetch_news_sentiment(ticker)
        trends_z = fetch_trends_zscore(ticker)
        sector_z = fetch_sector_momentum(ticker, as_of=as_of)
        print(f"  Sentiment (VADER) = {sentiment:.4f}")
        print(f"  Trends z-score    = {trends_z:.4f}")
        print(f"  Sector momentum z = {sector_z:.4f}\n")

    # --- Perturbation ---
    fc = compute_forecast(
        params, sentiment=sentiment, trends_zscore=trends_z, sector_zscore=sector_z,
    )

    print("Perturbation Series:")
    print(f"  mu_0 = {fc.mu_0*100:.4f}%  (raw GBM drift)")
    print(f"  D1   = {fc.delta_1*100:.4f}%  (first-order)")
    print(f"  D2   = {fc.delta_2*100:.4f}%  (second-order)")
    print(f"  mu*  = {fc.mu_perturbed*100:.4f}%  (perturbed drift)\n")

    direction = "UP" if fc.prob_up >= 0.5 else "DOWN"
    print(f"Forecast:")
    print(f"  P(up) [MC]   = {fc.prob_up*100:.1f}%")
    print(f"  Direction    = {direction}")
    print(f"  Confidence   = {fc.confidence:.3f}")
    print(f"  Price range  = {fc.range_floor_pct:+.1f}% to {fc.range_ceil_pct:+.1f}%  (median {fc.range_median_pct:+.1f}%)")
    print()


if __name__ == "__main__":
    main()
