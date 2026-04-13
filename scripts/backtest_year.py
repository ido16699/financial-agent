"""Backtest a full year vs S&P 500.

Trains on data BEFORE the test year, then runs the test year blind.
Compares portfolio return to SPY buy-and-hold.

Usage:
    python -m scripts.backtest_year --year 2023
    python -m scripts.backtest_year --year 2024
"""
from __future__ import annotations

import csv

import click
import pandas as pd
import numpy as np

from stochsignal.config import settings, watchlist
from stochsignal.ingest.prices import get_price_history
from stochsignal.ingest.sector import fetch_sector_momentum
from stochsignal.model.gbm import calibrate, prob_up_closed_form
from stochsignal.model.heston import calibrate_heston, current_vol_regime
from stochsignal.model.chaos import detect_regime
from stochsignal.model.waves import analyse_waves
from stochsignal.model.kelly import size_positions
from stochsignal.model.train import load_weights
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


def _mondays_in_range(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq="W-MON"))


def run_replay(
    start: str,
    end: str,
    tickers: list[str],
    capital: float,
    stop_loss_pct: float,
    kelly_frac: float,
    max_alloc: float,
    max_pos: float,
    w_mkt: float,
    w_wave: float,
    w_sector: float,
    w_gbm: float,
    w_interference: float,
    avg_win: float,
    avg_loss: float,
) -> list[dict]:
    """Run weekly replay, return list of portfolio snapshots."""

    mondays = _mondays_in_range(start, end)
    if len(mondays) < 2:
        return []

    n_weeks = len(mondays) - 1
    tc_rate = settings.transaction_cost_bps / 10_000

    # Pre-fetch prices
    full_end = (pd.Timestamp(end) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    all_prices: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = get_price_history(ticker, as_of=full_end, window_days=2000)
            all_prices[ticker] = df.sort_index()
        except Exception:
            pass

    spy_prices = None
    try:
        spy_prices = get_price_history("SPY", as_of=full_end, window_days=2000).sort_index()
    except Exception:
        pass

    portfolio_value = capital
    snapshots = [{"week": 0, "date": str(mondays[0].date()), "portfolio_value": portfolio_value,
                  "weekly_return_pct": 0.0, "wins": 0, "losses": 0}]
    total_wins = 0
    total_losses = 0

    for week_idx in range(n_weeks):
        monday = mondays[week_idx]
        next_monday = mondays[week_idx + 1]

        # Market momentum
        market_mom = 0.0
        if spy_prices is not None:
            spy_pit = spy_prices[spy_prices.index <= monday]
            if len(spy_pit) >= 63:
                spy_c = spy_pit["Close"].dropna().values
                if len(spy_c) > 21:
                    ret_20d = spy_c[-1] / spy_c[-21] - 1
                    rolling_rets = np.array([spy_c[j] / spy_c[j - 20] - 1 for j in range(20, len(spy_c))])
                    if len(rolling_rets) > 5 and np.std(rolling_rets) > 1e-9:
                        market_mom = float(np.clip(
                            (ret_20d - np.mean(rolling_rets)) / np.std(rolling_rets), -3, 3
                        ))

        week_forecasts = []
        regime_multipliers = {}

        for ticker in tickers:
            if ticker not in all_prices:
                continue
            prices = all_prices[ticker]
            pit = prices[prices.index <= monday]
            if len(pit) < 80:
                continue

            closes = pit["Close"].dropna()
            close_vals = closes.values
            log_ret = np.log(close_vals[1:] / close_vals[:-1])
            log_ret_series = pd.Series(log_ret, index=closes.index[1:])

            try:
                gbm_params = calibrate(ticker, log_ret_series)
            except ValueError:
                continue

            gbm_prob = prob_up_closed_form(gbm_params, settings.forecast_horizon_days)
            gbm_signal = gbm_prob - 0.5

            try:
                sector_z = fetch_sector_momentum(ticker, as_of=monday)
            except Exception:
                sector_z = 0.0

            try:
                wave = analyse_waves(log_ret, close_vals, settings.forecast_horizon_days)
                wave_signal = wave.wave_signal
            except Exception:
                wave_signal = 0.0

            try:
                regime_info = detect_regime(log_ret)
                regime_mult = regime_info.confidence_multiplier
            except Exception:
                regime_mult = 1.0

            regime_multipliers[ticker] = regime_mult

            gbm_rescaled = gbm_signal * 5
            linear = (
                w_mkt * market_mom +
                w_wave * wave_signal +
                w_sector * sector_z +
                w_gbm * gbm_rescaled
            )
            interference = (
                market_mom * wave_signal +
                market_mom * sector_z +
                market_mom * gbm_rescaled +
                wave_signal * sector_z +
                wave_signal * gbm_rescaled +
                sector_z * gbm_rescaled
            ) / 6.0
            composite = linear + w_interference * interference
            prob_up = 1.0 / (1.0 + np.exp(-composite))

            signals = [market_mom, wave_signal, sector_z, gbm_signal * 5]
            signs = [1 if s > 0 else -1 for s in signals if abs(s) > 0.1]
            agreement = abs(sum(signs)) / len(signs) if signs else 0.0
            signal_strength = min(abs(composite) / 1.5, 1.0)
            confidence = float(np.clip(agreement * signal_strength * regime_mult, 0, 1))

            future = prices[(prices.index > monday) & (prices.index <= next_monday)]
            if future.empty:
                continue
            price_start = float(close_vals[-1])
            price_end = float(future["Close"].iloc[-1])
            weekly_return = (price_end - price_start) / price_start

            week_forecasts.append({
                "ticker": ticker,
                "prob_up": prob_up,
                "confidence": confidence,
                "avg_weekly_win": avg_win,
                "avg_weekly_loss": avg_loss,
                "weekly_return": weekly_return,
            })

        if not week_forecasts:
            snapshots.append({"week": week_idx + 1, "date": str(next_monday.date()),
                              "portfolio_value": portfolio_value, "weekly_return_pct": 0.0,
                              "wins": 0, "losses": 0})
            continue

        positions = size_positions(
            week_forecasts, portfolio_value,
            kelly_mult=kelly_frac,
            max_position_pct=max_pos,
            max_total_alloc=max_alloc,
            regime_multipliers=regime_multipliers,
        )
        pos_map = {p.ticker: p for p in positions}

        week_pnl = 0.0
        week_wins = 0
        week_losses = 0

        for fc in week_forecasts:
            pos = pos_map.get(fc["ticker"])
            if pos is None:
                continue
            gross_pnl = pos.direction * fc["weekly_return"] * pos.capital_allocated
            net_pnl = gross_pnl - (tc_rate * pos.capital_allocated)
            if net_pnl > 0:
                week_wins += 1
                total_wins += 1
            else:
                week_losses += 1
                total_losses += 1
            week_pnl += net_pnl

        # Stop-loss
        max_loss = portfolio_value * stop_loss_pct
        if week_pnl < -max_loss:
            week_pnl = -max_loss

        prev = portfolio_value
        portfolio_value += week_pnl

        snapshots.append({
            "week": week_idx + 1,
            "date": str(next_monday.date()),
            "portfolio_value": portfolio_value,
            "weekly_return_pct": (week_pnl / prev) * 100 if prev > 0 else 0.0,
            "wins": week_wins,
            "losses": week_losses,
        })

    return snapshots


def get_spy_return(start: str, end: str) -> tuple[float, list[dict]]:
    """Get S&P 500 buy-and-hold return and weekly curve for the period."""
    mondays = _mondays_in_range(start, end)
    full_end = (pd.Timestamp(end) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    try:
        spy = get_price_history("SPY", as_of=full_end, window_days=2000).sort_index()
    except Exception:
        return 0.0, []

    snapshots = []
    spy_start_price = None
    for i, monday in enumerate(mondays):
        pit = spy[spy.index <= monday]
        if pit.empty:
            continue
        price = float(pit["Close"].iloc[-1])
        if spy_start_price is None:
            spy_start_price = price
        spy_value = 100_000 * (price / spy_start_price)
        snapshots.append({"week": i, "date": str(monday.date()), "spy_value": spy_value})

    if spy_start_price and snapshots:
        total_return = (snapshots[-1]["spy_value"] / 100_000 - 1) * 100
        return total_return, snapshots
    return 0.0, []


@click.command()
@click.option("--year", default=2023, type=int, show_default=True)
@click.option("--capital", default=100_000.0, show_default=True)
def main(year: int, capital: float):
    """Backtest a full year using trained params, compare to S&P 500."""

    weights = load_weights()
    if weights is None:
        log.error("No trained weights found!")
        return

    start = f"{year}-01-02"
    end = f"{year}-12-31"
    tickers = watchlist.all_tickers

    log.info("=" * 60)
    log.info("BACKTEST %d — StochSignal vs S&P 500", year)
    log.info("=" * 60)

    # Run model
    snapshots = run_replay(
        start, end, tickers, capital,
        stop_loss_pct=weights.learned_stop_loss_pct,
        kelly_frac=weights.learned_kelly_fraction,
        max_alloc=weights.learned_max_allocation,
        max_pos=weights.learned_max_position_pct,
        w_mkt=weights.w_market_mom,
        w_wave=weights.w_wave,
        w_sector=weights.w_sector,
        w_gbm=weights.w_gbm,
        w_interference=weights.w_interference,
        avg_win=weights.avg_weekly_win,
        avg_loss=weights.avg_weekly_loss,
    )

    # SPY benchmark
    spy_return, spy_snapshots = get_spy_return(start, end)
    spy_map = {s["date"]: s["spy_value"] for s in spy_snapshots}

    # Portfolio stats
    if not snapshots:
        print("No data!")
        return

    final_value = snapshots[-1]["portfolio_value"]
    total_return = (final_value / capital - 1) * 100
    n_weeks = len(snapshots) - 1
    annualised = ((final_value / capital) ** (52 / max(n_weeks, 1)) - 1) * 100

    total_wins = sum(s["wins"] for s in snapshots)
    total_losses = sum(s["losses"] for s in snapshots)
    total_trades = total_wins + total_losses
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    values = [s["portfolio_value"] for s in snapshots]
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    # SPY stats
    spy_annual = ((1 + spy_return / 100) ** (52 / max(n_weeks, 1)) - 1) * 100 if n_weeks > 0 else 0

    # Print combined weekly chart
    print(f"\n{'='*80}")
    print(f"  {year} FULL-YEAR BACKTEST — StochSignal vs S&P 500")
    print(f"{'='*80}")
    print(f"  Training: {weights.training_start} → {weights.training_end}")
    print(f"  Validation: {weights.validation_start} → {weights.validation_end}")
    print(f"  Learned: stop_loss={weights.learned_stop_loss_pct*100:.1f}%  kelly={weights.learned_kelly_fraction:.1f}  "
          f"weights=({weights.w_market_mom:.2f}/{weights.w_wave:.2f}/{weights.w_sector:.2f}/{weights.w_gbm:.2f})")
    print(f"{'='*80}")
    print(f"\n  {'Week':<6} {'Date':<12} {'Model $':>12} {'Model %':>9} {'SPY $':>12} {'SPY %':>9} {'Alpha':>8}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*9} {'-'*12} {'-'*9} {'-'*8}")

    for s in snapshots:
        model_pct = (s["portfolio_value"] / capital - 1) * 100
        spy_val = spy_map.get(s["date"], capital)
        spy_pct = (spy_val / capital - 1) * 100
        alpha = model_pct - spy_pct
        if s["week"] % 4 == 0 or s["week"] == n_weeks:  # print monthly + last
            print(f"  {s['week']:<6} {s['date']:<12} ${s['portfolio_value']:>10,.0f} {model_pct:>+8.2f}% ${spy_val:>10,.0f} {spy_pct:>+8.2f}% {alpha:>+7.2f}%")

    print(f"\n{'='*80}")
    print(f"  SUMMARY — {year}")
    print(f"{'='*80}")
    print(f"                    {'StochSignal':>15} {'S&P 500':>15} {'Alpha':>10}")
    print(f"  {'─'*60}")
    print(f"  Total return      {total_return:>+14.2f}% {spy_return:>+14.2f}% {total_return - spy_return:>+9.2f}%")
    print(f"  Annualised        {annualised:>+14.1f}% {spy_annual:>+14.1f}% {annualised - spy_annual:>+9.1f}%")
    print(f"  Max drawdown      {max_dd*100:>14.2f}%")
    print(f"  Win rate          {win_rate:>14.1f}%")
    print(f"  Total trades      {total_trades:>14}")
    print(f"{'='*80}\n")

    # Save CSV
    csv_file = f"backtest_{year}.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["week", "date", "model_value", "model_pct", "spy_value", "spy_pct", "alpha_pct"])
        writer.writeheader()
        for s in snapshots:
            spy_val = spy_map.get(s["date"], capital)
            model_pct = (s["portfolio_value"] / capital - 1) * 100
            spy_pct = (spy_val / capital - 1) * 100
            writer.writerow({
                "week": s["week"], "date": s["date"],
                "model_value": f"{s['portfolio_value']:.2f}",
                "model_pct": f"{model_pct:.2f}",
                "spy_value": f"{spy_val:.2f}",
                "spy_pct": f"{spy_pct:.2f}",
                "alpha_pct": f"{model_pct - spy_pct:.2f}",
            })
    print(f"  Saved to {csv_file}")


if __name__ == "__main__":
    main()
