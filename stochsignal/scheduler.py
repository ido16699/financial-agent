"""Weekly entry point: ingest → calibrate → forecast → render → notify.

Usage
-----
    stochsignal               # full run, sends email
    stochsignal --dry-run     # print to console only
    stochsignal --as-of 2026-03-03   # simulate as if run on that date
"""

from __future__ import annotations

import datetime
import sys

import click
import pandas as pd

from stochsignal.config import settings, watchlist
from stochsignal.ingest.prices import get_log_returns
from stochsignal.ingest.news import fetch_news_sentiment
from stochsignal.ingest.trends import fetch_trends_zscore
from stochsignal.model.gbm import calibrate
from stochsignal.model.perturbation import compute as compute_forecast, PerturbedForecast
from stochsignal.digest.renderer import render
from stochsignal.digest.notifier import ConsoleNotifier, SMTPNotifier
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


def _forecast_ticker(
    ticker: str,
    as_of: pd.Timestamp,
    skip_external: bool = False,
) -> PerturbedForecast | None:
    """Calibrate and forecast one ticker.

    Parameters
    ----------
    ticker:
        Symbol.
    as_of:
        Point-in-time date for price data.
    skip_external:
        If True, use sentiment=0.0 and trends=0.0 (backtest / replay mode
        where external data is not available point-in-time).
    """
    try:
        log_returns = get_log_returns(ticker, as_of=as_of)
        params = calibrate(ticker, log_returns)

        if skip_external:
            sentiment = 0.0
            trends_z = 0.0
        else:
            sentiment = fetch_news_sentiment(ticker)
            trends_z = fetch_trends_zscore(ticker)

        return compute_forecast(params, sentiment=sentiment, trends_zscore=trends_z)

    except Exception as exc:
        log.error("Failed to forecast %s: %s", ticker, exc)
        return None


@click.command()
@click.option("--dry-run", is_flag=True, help="Print digest to console instead of sending email.")
@click.option("--as-of", "as_of_str", default=None, help="Run as if today is this date (YYYY-MM-DD).")
@click.option("--skip-external", is_flag=True, help="Skip news/trends (use zeroed perturbation).")
@click.option("--tickers", default=None, help="Comma-separated list of tickers to run (default: full watchlist).")
def main(dry_run: bool, as_of_str: str | None, skip_external: bool, tickers: str | None) -> None:
    """StochSignal: weekly directional stock research digest."""

    as_of = pd.Timestamp(as_of_str) if as_of_str else pd.Timestamp.today().normalize()
    run_date = as_of.date()
    log.info("StochSignal run  as_of=%s  dry_run=%s", run_date, dry_run)

    ticker_list: list[str]
    if tickers:
        ticker_list = [t.strip() for t in tickers.split(",")]
    else:
        ticker_list = watchlist.all_tickers

    forecasts: list[PerturbedForecast] = []
    for ticker in ticker_list:
        log.info("Processing %s ...", ticker)
        fc = _forecast_ticker(ticker, as_of=as_of, skip_external=skip_external)
        if fc is not None:
            forecasts.append(fc)

    if not forecasts:
        log.error("No forecasts produced — aborting digest.")
        sys.exit(1)

    subject, html, text = render(forecasts, run_date=run_date)

    notifier = ConsoleNotifier() if dry_run else SMTPNotifier()
    notifier.send(subject=subject, html=html, text=text)

    high_conf = [f for f in forecasts if f.confidence >= settings.min_confidence_to_report]
    log.info(
        "Done. %d/%d tickers above confidence threshold %.2f",
        len(high_conf), len(forecasts), settings.min_confidence_to_report,
    )


if __name__ == "__main__":
    main()
