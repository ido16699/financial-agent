"""2026 day-by-day replay simulation with 100K portfolio tracking.

Simulates paper-trading by calling the model week-by-week from
`--start` to `--end`, always passing --as-of so the model only sees
data up to that date (point-in-time discipline).

Portfolio mechanics:
- Start with a cash balance (default $100,000).
- Each week, allocate equally across all tickers with confidence ≥ threshold.
- Go long if model says UP, short if DOWN.
- Track wins, losses, and portfolio value week by week.
- Report final P&L including transaction costs.

Sector momentum IS available during replay (it uses price data only,
which is point-in-time safe). News and Trends are zeroed.

Usage:
    python -m scripts.replay_2026
    python -m scripts.replay_2026 --start 2026-01-05 --end 2026-04-07
    python -m scripts.replay_2026 --tickers AAPL,MSFT --capital 50000
"""

from __future__ import annotations

import csv
import datetime

import click
import pandas as pd
import numpy as np

from stochsignal.config import settings, watchlist
from stochsignal.ingest.prices import get_price_history
from stochsignal.ingest.sector import fetch_sector_momentum
from stochsignal.model.gbm import calibrate
from stochsignal.model.perturbation import compute as compute_forecast
from stochsignal.backtest.scoring import hit_rate, brier_score
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)

OUTPUT_CSV = "replay_results.csv"
PORTFOLIO_CSV = "replay_portfolio.csv"


def _mondays_in_range(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq="W-MON"))


@click.command()
@click.option("--start", default="2026-01-05", show_default=True)
@click.option("--end", default="2026-04-07", show_default=True)
@click.option("--tickers", default=None, help="Comma-separated tickers. Default: full watchlist.")
@click.option("--capital", default=100_000.0, show_default=True, help="Starting capital in USD.")
@click.option("--min-confidence", default=0.0, show_default=True, help="Min confidence to trade (0 = trade all).")
def main(start: str, end: str, tickers: str | None, capital: float, min_confidence: float) -> None:
    """2026 replay: model only sees data up to as_of, tracks portfolio from starting capital."""

    ticker_list: list[str]
    if tickers:
        ticker_list = [t.strip() for t in tickers.split(",")]
    else:
        ticker_list = watchlist.all_tickers

    mondays = _mondays_in_range(start, end)
    if len(mondays) < 2:
        log.error("Need at least 2 Mondays in the replay window.")
        return

    n_weeks = len(mondays) - 1
    log.info("Replay %s → %s  (%d weeks, %d tickers, $%.0f capital)",
             start, end, n_weeks, len(ticker_list), capital)

    # Pre-fetch all price histories
    all_prices: dict[str, pd.DataFrame] = {}
    full_end = (pd.Timestamp(end) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    for ticker in ticker_list:
        try:
            df = get_price_history(ticker, as_of=full_end, window_days=99999)
            all_prices[ticker] = df.sort_index()
        except Exception as exc:
            log.error("Could not fetch prices for %s: %s", ticker, exc)

    tc_rate = settings.transaction_cost_bps / 10_000

    # Portfolio tracking
    portfolio_value = capital
    trade_rows = []
    portfolio_snapshots = [{"week": 0, "date": mondays[0].date(), "portfolio_value": portfolio_value,
                            "weekly_return_pct": 0.0, "n_trades": 0, "wins": 0, "losses": 0}]
    total_wins = 0
    total_losses = 0

    for week_idx in range(n_weeks):
        monday = mondays[week_idx]
        next_monday = mondays[week_idx + 1]

        # Generate forecasts for all tickers this week
        week_forecasts = []
        for ticker in ticker_list:
            if ticker not in all_prices:
                continue
            prices = all_prices[ticker]

            # Point-in-time slice
            pit = prices[prices.index <= monday]
            if len(pit) < 30:
                continue

            closes = pit["Close"].dropna()
            log_ret = np.log(closes.values[1:] / closes.values[:-1])
            log_ret_series = pd.Series(log_ret, index=closes.index[1:])

            try:
                params = calibrate(ticker, log_ret_series)
            except ValueError:
                continue

            # Sector momentum is point-in-time safe (uses price data)
            sector_z = fetch_sector_momentum(ticker, as_of=monday)

            fc = compute_forecast(
                params, sentiment=0.0, trends_zscore=0.0, sector_zscore=sector_z,
            )

            # Outcome: actual price change over the week
            future = prices[(prices.index > monday) & (prices.index <= next_monday)]
            if future.empty:
                continue
            price_start = float(pit["Close"].iloc[-1])
            price_end = float(future["Close"].iloc[-1])
            weekly_return = (price_end - price_start) / price_start

            week_forecasts.append({
                "ticker": ticker,
                "forecast": fc,
                "price_start": price_start,
                "price_end": price_end,
                "weekly_return": weekly_return,
            })

        # Filter by confidence
        tradeable = [w for w in week_forecasts if w["forecast"].confidence >= min_confidence]

        if not tradeable:
            portfolio_snapshots.append({
                "week": week_idx + 1, "date": next_monday.date(),
                "portfolio_value": portfolio_value, "weekly_return_pct": 0.0,
                "n_trades": 0, "wins": 0, "losses": 0,
            })
            continue

        # Equal allocation across tradeable tickers
        allocation_per_ticker = portfolio_value / len(tradeable)
        week_pnl = 0.0
        week_wins = 0
        week_losses = 0

        for entry in tradeable:
            fc = entry["forecast"]
            weekly_return = entry["weekly_return"]

            # Direction: long if P(up) >= 0.5, short otherwise
            direction = 1.0 if fc.prob_up >= 0.5 else -1.0
            gross_pnl = direction * weekly_return * allocation_per_ticker
            net_pnl = gross_pnl - (tc_rate * allocation_per_ticker)

            is_win = net_pnl > 0
            if is_win:
                week_wins += 1
                total_wins += 1
            else:
                week_losses += 1
                total_losses += 1

            week_pnl += net_pnl

            trade_rows.append({
                "as_of": monday.date(),
                "ticker": entry["ticker"],
                "direction": "LONG" if direction > 0 else "SHORT",
                "prob_up": fc.prob_up,
                "confidence": fc.confidence,
                "sector_z": fc.sector_zscore,
                "allocation": allocation_per_ticker,
                "weekly_return_pct": weekly_return * 100,
                "gross_pnl": direction * weekly_return * allocation_per_ticker,
                "net_pnl": net_pnl,
                "outcome": "WIN" if is_win else "LOSS",
                "range_floor_pct": fc.range_floor_pct,
                "range_ceil_pct": fc.range_ceil_pct,
            })

        portfolio_value += week_pnl
        weekly_return_pct = (week_pnl / (portfolio_value - week_pnl)) * 100 if portfolio_value != week_pnl else 0.0

        portfolio_snapshots.append({
            "week": week_idx + 1,
            "date": next_monday.date(),
            "portfolio_value": portfolio_value,
            "weekly_return_pct": weekly_return_pct,
            "n_trades": len(tradeable),
            "wins": week_wins,
            "losses": week_losses,
        })

        log.info(
            "Week %d (%s): %d trades, %d wins, %d losses, PnL=$%.2f, Portfolio=$%.2f",
            week_idx + 1, monday.date(), len(tradeable), week_wins, week_losses,
            week_pnl, portfolio_value,
        )

    # Write trade-level CSV
    if trade_rows:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trade_rows)
        log.info("Trade results: %s (%d rows)", OUTPUT_CSV, len(trade_rows))

    # Write portfolio snapshots CSV
    with open(PORTFOLIO_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(portfolio_snapshots[0].keys()))
        writer.writeheader()
        writer.writerows(portfolio_snapshots)

    # Summary
    total_return = portfolio_value - capital
    total_return_pct = (total_return / capital) * 100
    total_trades = total_wins + total_losses

    print("\n" + "=" * 60)
    print("  2026 PORTFOLIO REPLAY SUMMARY")
    print("=" * 60)
    print(f"  Period         : {start} → {end} ({n_weeks} weeks)")
    print(f"  Tickers        : {len(ticker_list)}")
    print(f"  Starting capital: ${capital:,.2f}")
    print(f"  Final value     : ${portfolio_value:,.2f}")
    print(f"  Total return    : ${total_return:,.2f} ({total_return_pct:+.2f}%)")
    print(f"  Transaction cost: {settings.transaction_cost_bps} bps/trade")
    print(f"  ---")
    print(f"  Total trades    : {total_trades}")
    print(f"  Wins            : {total_wins} ({total_wins/total_trades*100:.1f}%)" if total_trades > 0 else "  Wins            : 0")
    print(f"  Losses          : {total_losses} ({total_losses/total_trades*100:.1f}%)" if total_trades > 0 else "  Losses          : 0")

    # Hit rate and Brier score from the raw predictions
    if trade_rows:
        df = pd.DataFrame(trade_rows)
        outcomes = (df["outcome"] == "WIN").astype(int).values
        probs = df["prob_up"].values
        print(f"  ---")
        print(f"  Directional hit : {np.mean(outcomes)*100:.1f}%")

    # Max drawdown
    values = [s["portfolio_value"] for s in portfolio_snapshots]
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)
    print(f"  Max drawdown    : {max_dd*100:.2f}%")

    print("=" * 60)
    print(f"\nTrade details : {OUTPUT_CSV}")
    print(f"Portfolio curve: {PORTFOLIO_CSV}")


if __name__ == "__main__":
    main()
