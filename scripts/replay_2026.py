"""2026 day-by-day replay simulation.

Simulates paper-trading by calling the scheduler week-by-week from
`--start` to `--end`, always passing --as-of so the model only sees
data up to that date (point-in-time discipline).

News and Trends are skipped during replay (no point-in-time source),
so the perturbed model reduces to GBM-only. This gives a clean baseline.

Results are written to replay_results.csv and a summary is printed.

Usage:
    replay --start 2026-01-05 --end 2026-04-07
    replay --start 2026-01-05 --end 2026-04-07 --tickers AAPL,MSFT
"""

from __future__ import annotations

import csv
import datetime

import click
import pandas as pd
import numpy as np

from stochsignal.config import settings, watchlist
from stochsignal.ingest.prices import get_log_returns, get_price_history
from stochsignal.model.gbm import calibrate, prob_up_closed_form
from stochsignal.model.perturbation import compute as compute_forecast
from stochsignal.backtest.scoring import hit_rate, brier_score, simulated_pnl
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)

OUTPUT_CSV = "replay_results.csv"


def _mondays_in_range(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq="W-MON"))


@click.command()
@click.option("--start", default="2026-01-05", show_default=True, help="First Monday of the replay (YYYY-MM-DD).")
@click.option("--end", default="2026-12-28", show_default=True, help="Last Monday of the replay (YYYY-MM-DD).")
@click.option("--tickers", default=None, help="Comma-separated tickers. Default: full watchlist.")
def main(start: str, end: str, tickers: str | None) -> None:
    """Day-by-day 2026 replay: model only sees data up to as_of each week."""

    ticker_list: list[str]
    if tickers:
        ticker_list = [t.strip() for t in tickers.split(",")]
    else:
        ticker_list = watchlist.all_tickers

    mondays = _mondays_in_range(start, end)
    if len(mondays) < 2:
        log.error("Need at least 2 Mondays in the replay window.")
        return

    log.info("Replay %s → %s  (%d weeks, %d tickers)", start, end, len(mondays) - 1, len(ticker_list))

    rows = []

    for ticker in ticker_list:
        log.info("Replaying %s ...", ticker)
        try:
            # Fetch full 2025+2026 price history once
            full_end = (pd.Timestamp(end) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            all_prices = get_price_history(ticker, as_of=full_end, window_days=99999)
            all_prices = all_prices.sort_index()
        except Exception as exc:
            log.error("Could not fetch prices for %s: %s", ticker, exc)
            continue

        for i, monday in enumerate(mondays[:-1]):
            next_monday = mondays[i + 1]

            # Point-in-time price slice
            pit = all_prices[all_prices.index <= monday]
            if len(pit) < 30:
                continue

            closes = pit["Close"].dropna()
            log_ret = np.log(closes.values[1:] / closes.values[:-1])
            log_ret_series = pd.Series(log_ret, index=closes.index[1:])

            try:
                params = calibrate(ticker, log_ret_series)
            except ValueError:
                continue

            # Perturbed forecast with zeros for external signals
            fc = compute_forecast(params, sentiment=0.0, trends_zscore=0.0)

            # Outcome
            future = all_prices[(all_prices.index > monday) & (all_prices.index <= next_monday)]
            if future.empty:
                continue
            price_start = float(pit["Close"].iloc[-1])
            price_end = float(future["Close"].iloc[-1])
            weekly_log_ret = np.log(price_end / price_start)
            outcome = int(price_end > price_start)

            rows.append({
                "as_of": monday.date(),
                "ticker": ticker,
                "prob_up": fc.prob_up,
                "confidence": fc.confidence,
                "outcome": outcome,
                "weekly_log_return": weekly_log_ret,
                "mu_perturbed": fc.mu_perturbed,
            })

    if not rows:
        log.error("No replay results generated.")
        return

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Results written to %s (%d rows)", OUTPUT_CSV, len(rows))

    # Aggregate summary
    df = pd.DataFrame(rows)
    prob_up_arr = df["prob_up"].values
    outcome_arr = df["outcome"].values
    ret_arr = df["weekly_log_return"].values

    hr = hit_rate(prob_up_arr, outcome_arr)
    bs = brier_score(prob_up_arr, outcome_arr)
    pnl = simulated_pnl(prob_up_arr, ret_arr, settings.transaction_cost_bps)

    print("\n" + "=" * 50)
    print("  2026 Replay Summary")
    print("=" * 50)
    print(f"  Weeks replayed : {len(mondays) - 1}")
    print(f"  Ticker-weeks   : {len(df)}")
    print(f"  Hit rate       : {hr*100:.1f}%")
    print(f"  Brier score    : {bs:.4f}")
    print(f"  Simulated PnL  : {pnl:.2f}%  (net {settings.transaction_cost_bps}bps/trade)")
    print("=" * 50)
    print(f"\nFull results: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
