"""Multi-year backtest vs multiple benchmarks (SPY, QQQ, VGT).

Usage:
    python -m scripts.backtest_vs_benchmarks --start-year 2020 --end-year 2025
"""
from __future__ import annotations

import click
import numpy as np

from scripts.backtest_year import run_replay, get_benchmark_return
from stochsignal.config import settings, watchlist
from stochsignal.model.train import load_weights
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)

BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "VGT": "Info Tech",
}


@click.command()
@click.option("--start-year", default=2020, type=int)
@click.option("--end-year", default=2025, type=int)
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

        snapshots, _ = run_replay(
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

        # Get all benchmark returns
        bench_returns = {}
        for bticker, bname in BENCHMARKS.items():
            ret, _ = get_benchmark_return(bticker, start, end)
            bench_returns[bticker] = ret

        if not snapshots:
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

        row = {
            "year": year,
            "model_return": model_return,
            "max_dd": max_dd * 100,
            "win_rate": win_rate,
            "trades": total_trades,
        }
        for bticker in BENCHMARKS:
            row[f"bench_{bticker}"] = bench_returns[bticker]
            row[f"alpha_{bticker}"] = model_return - bench_returns[bticker]
        results.append(row)

        log.info("  %d done: model=%+.1f%% SPY=%+.1f%% QQQ=%+.1f%%",
                 year, model_return, bench_returns.get("SPY", 0), bench_returns.get("QQQ", 0))

    # Print summary table
    bench_keys = list(BENCHMARKS.keys())
    bench_names = list(BENCHMARKS.values())

    print(f"\n{'='*100}")
    print(f"  STOCHSIGNAL vs BENCHMARKS — {start_year} to {end_year}")
    print(f"  Training: {weights.training_start} → {weights.training_end}")
    print(f"  Learned: sl={weights.learned_stop_loss_pct*100:.1f}%  kelly={weights.learned_kelly_fraction:.1f}  "
          f"interf={weights.w_interference:.2f}  weights=({weights.w_market_mom:.2f}/{weights.w_wave:.2f}/{weights.w_sector:.2f}/{weights.w_gbm:.2f})")
    print(f"{'='*100}")

    # Header
    hdr = f"  {'Year':<6} {'Model':>8} {'MaxDD':>6} {'WR':>5}"
    for bname in bench_names:
        hdr += f" {'|':>2} {bname:>10} {'Alpha':>8}"
    print(hdr)
    print(f"  {'-'*6} {'-'*8} {'-'*6} {'-'*5}" + (" {0} {1} {2}".format('-'*2, '-'*10, '-'*8)) * len(bench_keys))

    for r in results:
        line = f"  {r['year']:<6} {r['model_return']:>+7.1f}% {r['max_dd']:>5.1f}% {r['win_rate']:>4.0f}%"
        for bk in bench_keys:
            alpha = r[f'alpha_{bk}']
            beat = "+" if alpha > 0 else " "
            line += f"  | {r[f'bench_{bk}']:>+9.1f}% {alpha:>+7.1f}%{beat}"
        print(line)

    print(f"  {'-'*96}")

    # Averages
    n = len(results)
    avg_model = np.mean([r["model_return"] for r in results])
    avg_dd = np.mean([r["max_dd"] for r in results])

    avg_line = f"  {'AVG':<6} {avg_model:>+7.1f}% {avg_dd:>5.1f}%      "
    for bk in bench_keys:
        avg_bench = np.mean([r[f"bench_{bk}"] for r in results])
        avg_alpha = np.mean([r[f"alpha_{bk}"] for r in results])
        years_beat = sum(1 for r in results if r[f"alpha_{bk}"] > 0)
        avg_line += f"  | {avg_bench:>+9.1f}% {avg_alpha:>+7.1f}% "
    print(avg_line)

    # Compounded returns
    print(f"\n  Compounded $100K over {n} years:")
    model_cum = capital
    bench_cum = {bk: capital for bk in bench_keys}
    for r in results:
        model_cum *= (1 + r["model_return"] / 100)
        for bk in bench_keys:
            bench_cum[bk] *= (1 + r[f"bench_{bk}"] / 100)

    model_cagr = ((model_cum / capital) ** (1 / n) - 1) * 100 if n > 0 else 0
    print(f"    {'StochSignal':<15}: ${model_cum:>12,.0f}  (CAGR {model_cagr:>+5.1f}%)")

    for bk, bname in BENCHMARKS.items():
        bcagr = ((bench_cum[bk] / capital) ** (1 / n) - 1) * 100 if n > 0 else 0
        alpha_cagr = model_cagr - bcagr
        years_beat = sum(1 for r in results if r[f"alpha_{bk}"] > 0)
        print(f"    {bname:<15}: ${bench_cum[bk]:>12,.0f}  (CAGR {bcagr:>+5.1f}%)  alpha={alpha_cagr:>+5.1f}%  beat {years_beat}/{n} yrs")

    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
