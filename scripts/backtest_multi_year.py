"""Multi-year backtest: StochSignal vs S&P 500.

Usage:
    python -m scripts.backtest_multi_year --start-year 2010 --end-year 2019
"""
from __future__ import annotations

import click
import numpy as np

from scripts.backtest_year import run_replay, get_spy_return
from stochsignal.config import settings, watchlist
from stochsignal.model.train import load_weights
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


@click.command()
@click.option("--start-year", default=2010, type=int)
@click.option("--end-year", default=2019, type=int)
@click.option("--capital", default=100_000.0)
def main(start_year: int, end_year: int, capital: float):
    weights = load_weights()
    if weights is None:
        print("No trained weights found!")
        return

    tickers = watchlist.all_tickers
    results = []

    for year in range(start_year, end_year + 1):
        start = f"{year}-01-02"
        end = f"{year}-12-31"

        log.info("=" * 40)
        log.info("BACKTESTING %d", year)
        log.info("=" * 40)

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

        spy_return, _ = get_spy_return(start, end)

        if not snapshots:
            log.warning("No data for %d", year)
            continue

        final = snapshots[-1]["portfolio_value"]
        model_return = (final / capital - 1) * 100
        n_weeks = len(snapshots) - 1

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

        results.append({
            "year": year,
            "model_return": model_return,
            "spy_return": spy_return,
            "alpha": model_return - spy_return,
            "max_dd": max_dd * 100,
            "win_rate": win_rate,
            "trades": total_trades,
            "weeks": n_weeks,
        })

        log.info("  %d done: model=%+.1f%% SPY=%+.1f%% alpha=%+.1f%%",
                 year, model_return, spy_return, model_return - spy_return)

    # Print summary table
    print(f"\n{'='*90}")
    print(f"  STOCHSIGNAL vs S&P 500 — {start_year} to {end_year} (all out-of-sample)")
    print(f"  Training: {weights.training_start} → {weights.training_end}")
    print(f"  Learned: stop_loss={weights.learned_stop_loss_pct*100:.1f}%  kelly={weights.learned_kelly_fraction:.1f}  "
          f"weights=({weights.w_market_mom:.2f}/{weights.w_wave:.2f}/{weights.w_sector:.2f}/{weights.w_gbm:.2f})")
    print(f"{'='*90}")
    print(f"  {'Year':<6} {'Model':>9} {'S&P 500':>9} {'Alpha':>8} {'MaxDD':>7} {'WinRate':>8} {'Trades':>7}")
    print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")

    for r in results:
        beat = "✓" if r["alpha"] > 0 else " "
        print(f"  {r['year']:<6} {r['model_return']:>+8.1f}% {r['spy_return']:>+8.1f}% {r['alpha']:>+7.1f}% {r['max_dd']:>6.1f}% {r['win_rate']:>7.1f}% {r['trades']:>7} {beat}")

    print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*8} {'-'*7}")

    # Averages
    avg_model = np.mean([r["model_return"] for r in results])
    avg_spy = np.mean([r["spy_return"] for r in results])
    avg_alpha = np.mean([r["alpha"] for r in results])
    avg_dd = np.mean([r["max_dd"] for r in results])
    avg_wr = np.mean([r["win_rate"] for r in results])
    years_beat = sum(1 for r in results if r["alpha"] > 0)

    print(f"  {'AVG':<6} {avg_model:>+8.1f}% {avg_spy:>+8.1f}% {avg_alpha:>+7.1f}% {avg_dd:>6.1f}% {avg_wr:>7.1f}%")
    print(f"\n  Years beating S&P: {years_beat}/{len(results)}")

    # Cumulative (compounded)
    model_cumulative = capital
    spy_cumulative = capital
    for r in results:
        model_cumulative *= (1 + r["model_return"] / 100)
        spy_cumulative *= (1 + r["spy_return"] / 100)

    model_cagr = ((model_cumulative / capital) ** (1 / len(results)) - 1) * 100 if results else 0
    spy_cagr = ((spy_cumulative / capital) ** (1 / len(results)) - 1) * 100 if results else 0

    print(f"\n  Compounded $100K over {len(results)} years:")
    print(f"    StochSignal : ${model_cumulative:>12,.0f}  (CAGR {model_cagr:+.1f}%)")
    print(f"    S&P 500     : ${spy_cumulative:>12,.0f}  (CAGR {spy_cagr:+.1f}%)")
    print(f"    Alpha CAGR  : {model_cagr - spy_cagr:+.1f}%")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
