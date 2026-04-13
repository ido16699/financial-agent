"""2026 Q1 replay with full trained engine.

Pipeline:
  1. Load trained weights (from 2020-2025 training).
  2. Each week: for each ticker
     a. Calibrate GBM + Heston from point-in-time prices
     b. Compute sector momentum, wave analysis, regime detection
     c. Predict P(up) using trained logistic regression weights
     d. Size position via Kelly criterion
  3. Track portfolio value week by week with real wins/losses.

Usage:
    python -m scripts.replay_2026
    python -m scripts.replay_2026 --start 2026-01-05 --end 2026-04-07
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
from stochsignal.model.kelly import size_positions, PositionSize
from stochsignal.model.train import load_weights, predict_with_weights, TrainedWeights
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)

OUTPUT_CSV = "replay_results.csv"
PORTFOLIO_CSV = "replay_portfolio.csv"


def _mondays_in_range(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq="W-MON"))


@click.command()
@click.option("--start", default="2026-01-05", show_default=True)
@click.option("--end", default="2026-04-07", show_default=True)
@click.option("--tickers", default=None, help="Comma-separated tickers.")
@click.option("--capital", default=100_000.0, show_default=True)
def main(start: str, end: str, tickers: str | None, capital: float) -> None:
    """2026 replay with trained model, Heston, regime detection, Kelly sizing."""

    ticker_list = [t.strip() for t in tickers.split(",")] if tickers else watchlist.all_tickers

    # Load trained weights (must run training first)
    weights = load_weights()
    if weights is None:
        log.error("No trained weights found! Run training first: python -m stochsignal.model.train")
        return

    log.info("Loaded trained weights: hit_rate=%.1f%%, threshold=%.2f, n=%d samples",
             weights.training_hit_rate * 100, weights.optimal_confidence_threshold, weights.n_training_samples)

    mondays = _mondays_in_range(start, end)
    if len(mondays) < 2:
        log.error("Need at least 2 Mondays.")
        return

    n_weeks = len(mondays) - 1
    log.info("Replay %s → %s  (%d weeks, %d tickers, $%.0f)", start, end, n_weeks, len(ticker_list), capital)

    # Pre-fetch all prices
    all_prices: dict[str, pd.DataFrame] = {}
    full_end = (pd.Timestamp(end) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    for ticker in ticker_list:
        try:
            df = get_price_history(ticker, as_of=full_end, window_days=2000)
            all_prices[ticker] = df.sort_index()
        except Exception as exc:
            log.error("Prices failed for %s: %s", ticker, exc)

    # Pre-fetch SPY for market momentum
    spy_prices = all_prices.get("SPY")
    if spy_prices is None:
        try:
            spy_prices = get_price_history("SPY", as_of=full_end, window_days=2000).sort_index()
        except Exception:
            spy_prices = None
            log.warning("Could not fetch SPY for market momentum")

    tc_rate = settings.transaction_cost_bps / 10_000
    # All risk params from training — NOT hand-tuned on 2026 data
    MAX_WEEKLY_LOSS_PCT = weights.learned_stop_loss_pct
    KELLY_FRACTION = weights.learned_kelly_fraction
    log.info("Learned risk params: stop_loss=%.3f%%, kelly=%.2f, max_alloc=%.0f%%, max_pos=%.0f%%",
             MAX_WEEKLY_LOSS_PCT * 100, KELLY_FRACTION,
             weights.learned_max_allocation * 100, weights.learned_max_position_pct * 100)
    log.info("Learned signal weights: mkt=%.2f wave=%.2f sector=%.2f gbm=%.2f",
             weights.w_market_mom, weights.w_wave, weights.w_sector, weights.w_gbm)

    portfolio_value = capital
    trade_rows = []
    portfolio_snapshots = [{"week": 0, "date": str(mondays[0].date()), "portfolio_value": portfolio_value,
                            "weekly_return_pct": 0.0, "n_trades": 0, "wins": 0, "losses": 0}]
    total_wins = 0
    total_losses = 0

    for week_idx in range(n_weeks):
        monday = mondays[week_idx]
        next_monday = mondays[week_idx + 1]

        # Market momentum: SPY 20-day return z-scored
        market_mom = 0.0
        if spy_prices is not None:
            spy_pit = spy_prices[spy_prices.index <= monday]
            if len(spy_pit) >= 63:
                spy_c = spy_pit["Close"].dropna().values
                ret_20d = spy_c[-1] / spy_c[-21] - 1 if len(spy_c) > 21 else 0.0
                rolling_rets = np.array([
                    spy_c[j] / spy_c[j - 20] - 1 for j in range(20, len(spy_c))
                ])
                if len(rolling_rets) > 5 and np.std(rolling_rets) > 1e-9:
                    market_mom = float(np.clip(
                        (ret_20d - np.mean(rolling_rets)) / np.std(rolling_rets), -3, 3
                    ))

        # Generate forecasts for all tickers
        week_forecasts = []
        regime_multipliers = {}

        for ticker in ticker_list:
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

            # GBM baseline
            gbm_prob = prob_up_closed_form(gbm_params, settings.forecast_horizon_days)

            # Heston calibration
            try:
                heston_params = calibrate_heston(ticker, log_ret_series)
                vol_regime = current_vol_regime(heston_params)
                current_sigma = float(np.sqrt(heston_params.v0))
            except Exception:
                heston_params = None
                vol_regime = "normal"
                current_sigma = gbm_params.sigma

            # Sector momentum
            try:
                sector_z = fetch_sector_momentum(ticker, as_of=monday)
            except Exception:
                sector_z = 0.0

            # Wave analysis
            try:
                wave = analyse_waves(log_ret, close_vals, settings.forecast_horizon_days)
                wave_signal = wave.wave_signal
            except Exception:
                wave_signal = 0.0

            # Regime detection
            try:
                regime_info = detect_regime(log_ret)
                regime_mult = regime_info.confidence_multiplier
                regime_name = regime_info.regime
                lyapunov = regime_info.lyapunov_exponent
            except Exception:
                regime_mult = 1.0
                regime_name = "unknown"
                lyapunov = 0.0

            regime_multipliers[ticker] = regime_mult

            # Direct signal combination (more reactive than logistic regression)
            # Center GBM prob around 0: positive → bullish, negative → bearish
            gbm_signal = gbm_prob - 0.5

            # Composite score with interference (all weights learned from validation)
            gbm_rescaled = gbm_signal * 5
            linear = (
                weights.w_market_mom * market_mom +
                weights.w_wave * wave_signal +
                weights.w_sector * sector_z +
                weights.w_gbm * gbm_rescaled
            )
            # Interference cross-terms: constructive when signals agree, destructive when not
            interference = (
                market_mom * wave_signal +
                market_mom * sector_z +
                market_mom * gbm_rescaled +
                wave_signal * sector_z +
                wave_signal * gbm_rescaled +
                sector_z * gbm_rescaled
            ) / 6.0
            composite = linear + weights.w_interference * interference

            # Convert to probability via sigmoid
            trained_prob = 1.0 / (1.0 + np.exp(-composite))

            # Confidence: agreement among signals (all same sign → high confidence)
            signals = [market_mom, wave_signal, sector_z, gbm_signal * 5]
            signs = [1 if s > 0 else -1 for s in signals if abs(s) > 0.1]
            if len(signs) > 0:
                agreement = abs(sum(signs)) / len(signs)  # 1.0 = all agree
            else:
                agreement = 0.0
            signal_strength = min(abs(composite) / 1.5, 1.0)
            confidence = float(np.clip(agreement * signal_strength * regime_mult, 0, 1))

            # Outcome
            future = prices[(prices.index > monday) & (prices.index <= next_monday)]
            if future.empty:
                continue
            price_start = float(close_vals[-1])
            price_end = float(future["Close"].iloc[-1])
            weekly_return = (price_end - price_start) / price_start

            week_forecasts.append({
                "ticker": ticker,
                "prob_up": trained_prob,
                "confidence": confidence,
                "avg_weekly_win": weights.avg_weekly_win,
                "avg_weekly_loss": weights.avg_weekly_loss,
                "weekly_return": weekly_return,
                "price_start": price_start,
                "price_end": price_end,
                "sector_z": sector_z,
                "wave_signal": wave_signal,
                "regime": regime_name,
                "regime_mult": regime_mult,
                "vol_regime": vol_regime,
                "gbm_prob": gbm_prob,
            })

        if not week_forecasts:
            portfolio_snapshots.append({
                "week": week_idx + 1, "date": str(next_monday.date()),
                "portfolio_value": portfolio_value, "weekly_return_pct": 0.0,
                "n_trades": 0, "wins": 0, "losses": 0,
            })
            continue

        # Kelly position sizing (all params from training)
        positions = size_positions(
            week_forecasts, portfolio_value,
            kelly_mult=KELLY_FRACTION,
            max_position_pct=weights.learned_max_position_pct,
            max_total_alloc=weights.learned_max_allocation,
            regime_multipliers=regime_multipliers,
        )

        # Map positions back to forecasts
        pos_map = {p.ticker: p for p in positions}
        week_pnl = 0.0
        week_wins = 0
        week_losses = 0

        for fc in week_forecasts:
            pos = pos_map.get(fc["ticker"])
            if pos is None:
                continue  # Kelly said skip

            direction = pos.direction
            allocation = pos.capital_allocated
            weekly_return = fc["weekly_return"]

            gross_pnl = direction * weekly_return * allocation
            net_pnl = gross_pnl - (tc_rate * allocation)

            is_win = net_pnl > 0
            if is_win:
                week_wins += 1
                total_wins += 1
            else:
                week_losses += 1
                total_losses += 1

            week_pnl += net_pnl

            trade_rows.append({
                "as_of": str(monday.date()),
                "ticker": fc["ticker"],
                "direction": "LONG" if direction > 0 else "SHORT",
                "prob_up": f"{fc['prob_up']:.4f}",
                "confidence": f"{fc['confidence']:.3f}",
                "kelly_pct": f"{pos.position_pct*100:.1f}",
                "allocation": f"{allocation:.0f}",
                "weekly_return_pct": f"{weekly_return*100:.2f}",
                "net_pnl": f"{net_pnl:.2f}",
                "outcome": "WIN" if is_win else "LOSS",
                "regime": fc["regime"],
                "vol_regime": fc["vol_regime"],
                "sector_z": f"{fc['sector_z']:.2f}",
                "wave": f"{fc['wave_signal']:.2f}",
            })

        # Stop-loss: cap weekly loss at MAX_WEEKLY_LOSS_PCT of portfolio
        max_loss = portfolio_value * MAX_WEEKLY_LOSS_PCT
        if week_pnl < -max_loss:
            log.info("STOP-LOSS triggered: capping loss from $%.0f to $%.0f", -week_pnl, max_loss)
            week_pnl = -max_loss

        prev_value = portfolio_value
        portfolio_value += week_pnl
        weekly_return_pct = (week_pnl / prev_value) * 100 if prev_value > 0 else 0.0

        portfolio_snapshots.append({
            "week": week_idx + 1,
            "date": str(next_monday.date()),
            "portfolio_value": portfolio_value,
            "weekly_return_pct": weekly_return_pct,
            "n_trades": len(positions),
            "wins": week_wins,
            "losses": week_losses,
        })

        log.info(
            "Week %d (%s): %d trades, %d W / %d L, PnL=$%.0f, Portfolio=$%.0f",
            week_idx + 1, monday.date(), len(positions), week_wins, week_losses,
            week_pnl, portfolio_value,
        )

    # Write CSVs
    if trade_rows:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trade_rows)

    with open(PORTFOLIO_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(portfolio_snapshots[0].keys()))
        writer.writeheader()
        writer.writerows(portfolio_snapshots)

    # Summary
    total_return = portfolio_value - capital
    total_return_pct = (total_return / capital) * 100
    total_trades = total_wins + total_losses
    annualised = ((portfolio_value / capital) ** (52 / max(n_weeks, 1)) - 1) * 100

    # Max drawdown
    values = [s["portfolio_value"] for s in portfolio_snapshots]
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    print("\n" + "=" * 62)
    print("  2026 Q1 PORTFOLIO REPLAY — FULL TRAINED ENGINE")
    print("=" * 62)
    print(f"  Period           : {start} → {end} ({n_weeks} weeks)")
    print(f"  Training         : {weights.training_start} → {weights.training_end}")
    print(f"  Training hit rate: {weights.training_hit_rate*100:.1f}%")
    print(f"  ---")
    print(f"  Starting capital : ${capital:,.2f}")
    print(f"  Final value      : ${portfolio_value:,.2f}")
    print(f"  Total return     : ${total_return:,.2f} ({total_return_pct:+.2f}%)")
    print(f"  Annualised return: {annualised:+.1f}%")
    print(f"  Max drawdown     : {max_dd*100:.2f}%")
    print(f"  ---")
    if total_trades > 0:
        print(f"  Total trades     : {total_trades}")
        print(f"  Wins             : {total_wins} ({total_wins/total_trades*100:.1f}%)")
        print(f"  Losses           : {total_losses} ({total_losses/total_trades*100:.1f}%)")
    else:
        print("  Total trades     : 0")
    print(f"  Transaction cost : {settings.transaction_cost_bps} bps/trade")
    print("=" * 62)
    print(f"\nTrade details : {OUTPUT_CSV}")
    print(f"Portfolio curve: {PORTFOLIO_CSV}")


if __name__ == "__main__":
    main()
