"""Walk-forward backtest harness.

Price-only GBM baseline (2020–2025):
- At each weekly step, calibrate GBM on the prior `calibration_window_days`
  of data (as_of = Monday of the week).
- Record P(up) using the closed-form formula.
- Outcome = 1 if the stock closed higher at end of the week, 0 otherwise.
- NO news/trends look-up — this is the price-only baseline.

Point-in-time discipline:
- get_price_history(..., as_of=as_of) is called with the exact Monday date.
- The outcome is computed from prices that are strictly after `as_of`.

Usage (CLI via scripts/calibrate.py or directly):
    from stochsignal.backtest.harness import run_backtest
    results = run_backtest(["AAPL", "MSFT"])
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from stochsignal.config import settings
from stochsignal.ingest.prices import get_price_history, get_log_returns
from stochsignal.model.gbm import calibrate, prob_up_closed_form
from stochsignal.backtest.scoring import score_ticker, BacktestResults, print_summary
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


def _weekly_mondays(start: str, end: str) -> list[pd.Timestamp]:
    """Return all Mondays in [start, end] as Timestamps."""
    idx = pd.date_range(start=start, end=end, freq="W-MON")
    return list(idx)


def run_ticker_backtest(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> BacktestResults:
    """Run price-only GBM backtest for a single ticker.

    Fetches the full price history once, then slices it walk-forward.
    """
    start = start or settings.backtest_start
    end = end or settings.backtest_end

    log.info("Backtesting %s  %s → %s", ticker, start, end)

    mondays = _weekly_mondays(start, end)
    if len(mondays) < 2:
        raise ValueError(f"Too few weeks in backtest window for {ticker}")

    # Fetch full history once (up to end + buffer for outcome)
    full_end = (pd.Timestamp(end) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    all_prices = get_price_history(ticker, as_of=full_end, window_days=2000)
    all_prices = all_prices.sort_index()

    prob_ups, outcomes, log_returns_list, dates = [], [], [], []

    for i, monday in enumerate(mondays[:-1]):
        next_monday = mondays[i + 1]

        # Prices available as of this Monday (point-in-time slice)
        pit_prices = all_prices[all_prices.index <= monday]
        if len(pit_prices) < 30:
            continue

        # Calibrate on last `calibration_window_days` rows
        window = settings.calibration_window_days
        pit_prices = pit_prices.tail(window)
        closes = pit_prices["Close"].dropna()
        if len(closes) < 20:
            continue

        log_ret = np.log(closes.values[1:] / closes.values[:-1])
        log_ret_series = pd.Series(log_ret, index=closes.index[1:])

        try:
            params = calibrate(ticker, log_ret_series)
        except ValueError:
            continue

        prob = prob_up_closed_form(params, settings.forecast_horizon_days)

        # Outcome: did price go up between monday and next_monday?
        future = all_prices[(all_prices.index > monday) & (all_prices.index <= next_monday)]
        if future.empty:
            continue

        price_at_monday_rows = all_prices[all_prices.index <= monday]
        if price_at_monday_rows.empty:
            continue
        price_start = float(price_at_monday_rows["Close"].iloc[-1])
        price_end = float(future["Close"].iloc[-1])

        weekly_log_return = np.log(price_end / price_start)
        outcome = int(price_end > price_start)

        prob_ups.append(prob)
        outcomes.append(outcome)
        log_returns_list.append(weekly_log_return)
        dates.append(monday)

    if not prob_ups:
        raise ValueError(f"No backtest samples generated for {ticker}")

    return score_ticker(
        ticker=ticker,
        prob_up=np.array(prob_ups),
        outcome=np.array(outcomes),
        weekly_log_returns=np.array(log_returns_list),
        transaction_cost_bps=settings.transaction_cost_bps,
        dates=pd.DatetimeIndex(dates),
    )


def run_backtest(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    plot: bool = False,
) -> list[BacktestResults]:
    """Run price-only GBM backtest across all tickers and print summary.

    Parameters
    ----------
    tickers:
        List of ticker symbols. Defaults to full watchlist.
    start / end:
        Date range strings. Defaults to settings.backtest_start/end.
    plot:
        If True, show a calibration/reliability diagram using matplotlib.
    """
    from stochsignal.config import watchlist as wl
    tickers = tickers or wl.all_tickers

    results = []
    for ticker in tickers:
        try:
            r = run_ticker_backtest(ticker, start=start, end=end)
            results.append(r)
        except Exception as exc:
            log.error("Backtest failed for %s: %s", ticker, exc)

    print_summary(results)

    if plot and results:
        _plot_calibration(results)

    return results


def _plot_calibration(results: list[BacktestResults]) -> None:
    """Plot reliability diagram for all tickers combined."""
    import matplotlib.pyplot as plt

    all_prob, all_outcome = [], []
    for r in results:
        if r.predictions is not None:
            all_prob.extend(r.predictions["prob_up"].tolist())
            all_outcome.extend(r.predictions["outcome"].tolist())

    from stochsignal.backtest.scoring import calibration_curve
    cal = calibration_curve(np.array(all_prob), np.array(all_outcome))

    if not cal:
        print("Not enough data for calibration plot.")
        return

    mean_probs, frac_pos = zip(*cal)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(mean_probs, frac_pos, s=80, zorder=5, label="Model")
    ax.plot(mean_probs, frac_pos, alpha=0.5)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram (price-only GBM baseline)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("calibration.png", dpi=150)
    print("Calibration plot saved to calibration.png")
    plt.show()
