"""Backtest a full year vs S&P 500 with real portfolio tracking.

Portfolio rules (realistic):
  - Start with `capital` in CASH (default $100K)
  - Can only BUY if enough cash available
  - Can only SELL stocks we actually own (no naked shorts)
  - Weekly decision flow per ticker:
        own + prob_up >= 0.5  → HOLD
        own + prob_up <  0.5  → SELL (convert to cash at Monday close)
        no   + prob_up >= 0.5 → BUY (if cash permits)
        no   + prob_up <  0.5 → SKIP

Each week's snapshot tracks: portfolio_value, cash, n_holdings, holdings_value.

Usage:
    python -m scripts.backtest_year --year 2023
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import numpy as np

from stochsignal.config import settings, watchlist
from stochsignal.ingest.sector import fetch_sector_momentum
from stochsignal.model.gbm import calibrate, prob_up_closed_form
from stochsignal.model.chaos import detect_regime
from stochsignal.model.waves import analyse_waves
from stochsignal.model.kelly import size_positions
from stochsignal.model.train import load_weights
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


def _mondays_in_range(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq="W-MON"))


def _load_prices(tickers: list[str], full_end: str,
                 prices_override: Optional[dict[str, pd.DataFrame]] = None) -> dict[str, pd.DataFrame]:
    """Load prices from override (parquet snapshot) or fall back to yfinance."""
    if prices_override is not None:
        # Use pre-loaded snapshot (preferred for training / Colab)
        return {t: prices_override[t].sort_index()
                for t in tickers if t in prices_override}

    from stochsignal.ingest.prices import get_price_history
    out = {}
    for ticker in tickers:
        try:
            df = get_price_history(ticker, as_of=full_end, window_days=4000)
            out[ticker] = df.sort_index()
        except Exception:
            pass
    return out


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
    prices_override: Optional[dict[str, pd.DataFrame]] = None,
    universe_snapshots: Optional[dict[str, list[str]]] = None,
    spy_override: Optional[pd.DataFrame] = None,
):
    """Run weekly replay with cash + holdings tracking.

    Returns (snapshots, trade_log) where:
      - snapshots: per-week portfolio state
      - trade_log: per-stock per-week decision records
    """
    mondays = _mondays_in_range(start, end)
    if len(mondays) < 2:
        return [], []

    n_weeks = len(mondays) - 1
    tc_rate = settings.transaction_cost_bps / 10_000

    # ---- Price loading ----
    full_end = (pd.Timestamp(end) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    # Load union of any ticker we might ever consider
    all_candidate_tickers = set(tickers)
    if universe_snapshots:
        for ts in universe_snapshots.values():
            all_candidate_tickers.update(ts)
    all_prices = _load_prices(sorted(all_candidate_tickers), full_end, prices_override)

    spy_prices = spy_override
    if spy_prices is None:
        try:
            from stochsignal.ingest.prices import get_price_history
            spy_prices = get_price_history("SPY", as_of=full_end, window_days=4000).sort_index()
        except Exception:
            spy_prices = None

    # ---- Portfolio state ----
    cash: float = capital
    holdings: dict[str, dict] = {}  # {ticker: {"shares", "entry_price", "entry_date"}}

    snapshots = [{
        "week": 0,
        "date": str(mondays[0].date()),
        "portfolio_value": capital,
        "cash": capital,
        "holdings_value": 0.0,
        "n_holdings": 0,
        "weekly_return_pct": 0.0,
        "wins": 0, "losses": 0,
    }]
    trade_log: list[dict] = []

    # ---- Universe helper ----
    def active_universe_for(monday: pd.Timestamp) -> list[str]:
        if not universe_snapshots:
            return tickers
        # Pick the most recent snapshot date <= monday
        snap_dates = sorted(universe_snapshots.keys())
        eligible = [d for d in snap_dates if pd.Timestamp(d) <= monday]
        if not eligible:
            return tickers
        return universe_snapshots[eligible[-1]]

    # ---- Weekly loop ----
    for week_idx in range(n_weeks):
        monday = mondays[week_idx]
        next_monday = mondays[week_idx + 1]
        active_tickers = active_universe_for(monday)

        # --- Market momentum (SPY 20d z-score) ---
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

        # --- Compute forecasts + prices for active universe + currently-held tickers ---
        # (We need signals on held tickers even if dropped from universe, to decide if to sell)
        tickers_to_score = set(active_tickers) | set(holdings.keys())

        week_forecasts: list[dict] = []
        regime_multipliers: dict[str, float] = {}
        current_prices: dict[str, float] = {}  # price at monday
        next_prices: dict[str, float] = {}     # price at next_monday

        for ticker in tickers_to_score:
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

            # Stochastic diffusion (GBM) + interference composite
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
            prob_down = 1.0 - prob_up

            signals = [market_mom, wave_signal, sector_z, gbm_signal * 5]
            signs = [1 if s > 0 else -1 for s in signals if abs(s) > 0.1]
            agreement = abs(sum(signs)) / len(signs) if signs else 0.0
            signal_strength = min(abs(composite) / 1.5, 1.0)
            confidence = float(np.clip(agreement * signal_strength * regime_mult, 0, 1))

            price_start = float(close_vals[-1])
            future = prices[(prices.index > monday) & (prices.index <= next_monday)]
            if future.empty:
                continue
            price_end = float(future["Close"].iloc[-1])

            current_prices[ticker] = price_start
            next_prices[ticker] = price_end

            week_forecasts.append({
                "ticker": ticker,
                "prob_up": prob_up,
                "prob_down": prob_down,
                "confidence": confidence,
                "avg_weekly_win": avg_win,
                "avg_weekly_loss": avg_loss,
                "weekly_return": (price_end - price_start) / price_start,
            })

        fc_map = {f["ticker"]: f for f in week_forecasts}

        # --- Step 1: SELL decisions on currently-held positions ---
        week_wins = 0
        week_losses = 0
        for ticker in list(holdings.keys()):
            holding = holdings[ticker]
            fc = fc_map.get(ticker)
            sell_price = current_prices.get(ticker)

            if sell_price is None:
                # No fresh price this week — keep holding
                continue

            # Decide: if fc absent or prob_up < 0.5 → SELL at monday close
            should_sell = (fc is None) or (fc["prob_up"] < 0.5)
            if not should_sell:
                # HOLD — log the hold
                trade_log.append({
                    "week": week_idx + 1,
                    "date": str(next_monday.date()),
                    "ticker": ticker,
                    "action": "HOLD",
                    "prob_up": round(fc["prob_up"], 4),
                    "prob_down": round(fc["prob_down"], 4),
                    "confidence": round(fc["confidence"], 3),
                    "shares": round(holding["shares"], 4),
                    "price": round(sell_price, 2),
                    "allocation": round(holding["shares"] * sell_price, 0),
                    "cash_after": round(cash, 0),
                    "n_holdings_after": len(holdings),
                    "weekly_return_pct": 0.0,
                    "net_pnl": 0.0,
                    "outcome": "",
                })
                continue

            # Execute SELL
            gross_proceeds = holding["shares"] * sell_price
            net_proceeds = gross_proceeds * (1 - tc_rate)
            cost_basis = holding["shares"] * holding["entry_price"]
            trade_pnl = net_proceeds - cost_basis
            cash += net_proceeds
            is_win = trade_pnl > 0
            if is_win:
                week_wins += 1
            else:
                week_losses += 1

            trade_log.append({
                "week": week_idx + 1,
                "date": str(next_monday.date()),
                "ticker": ticker,
                "action": "SELL",
                "prob_up": round(fc["prob_up"] if fc else 0.0, 4),
                "prob_down": round(fc["prob_down"] if fc else 1.0, 4),
                "confidence": round(fc["confidence"] if fc else 0.0, 3),
                "shares": round(holding["shares"], 4),
                "price": round(sell_price, 2),
                "allocation": round(net_proceeds, 0),
                "cash_after": round(cash, 0),
                "n_holdings_after": len(holdings) - 1,
                "weekly_return_pct": round((sell_price / holding["entry_price"] - 1) * 100, 2),
                "net_pnl": round(trade_pnl, 2),
                "outcome": "WIN" if is_win else "LOSS",
            })
            del holdings[ticker]

        # --- Step 2: BUY candidates (not owned, prob_up >= 0.5, in active universe) ---
        buy_candidates = [
            fc for fc in week_forecasts
            if fc["ticker"] not in holdings
            and fc["ticker"] in active_tickers
            and fc["prob_up"] >= 0.5
        ]

        if buy_candidates and cash > 100:
            # Total portfolio value used for Kelly sizing (stake is relative to wealth)
            total_portfolio = cash + sum(
                h["shares"] * current_prices.get(t, h["entry_price"])
                for t, h in holdings.items()
            )
            positions = size_positions(
                buy_candidates, total_portfolio,
                kelly_mult=kelly_frac,
                max_position_pct=max_pos,
                max_total_alloc=max_alloc,
                regime_multipliers=regime_multipliers,
            )
            # All positions here should be LONG (prob_up >= 0.5 filter)
            total_desired = sum(max(p.capital_allocated, 0.0) for p in positions if p.direction > 0)
            if total_desired > 0:
                scale = min(1.0, cash / (total_desired * (1 + tc_rate)))
            else:
                scale = 0.0

            for pos in positions:
                if pos.direction <= 0:  # safety: skip any accidental shorts
                    continue
                ticker = pos.ticker
                desired_alloc = pos.capital_allocated * scale
                if desired_alloc < 50:
                    continue
                buy_price = current_prices.get(ticker)
                if buy_price is None or buy_price <= 0:
                    continue
                cost = desired_alloc * (1 + tc_rate)
                if cost > cash:
                    continue  # not enough cash (invariant check)

                shares = desired_alloc / buy_price
                cash -= cost
                holdings[ticker] = {
                    "shares": shares,
                    "entry_price": buy_price,
                    "entry_date": str(monday.date()),
                }

                fc = fc_map[ticker]
                trade_log.append({
                    "week": week_idx + 1,
                    "date": str(next_monday.date()),
                    "ticker": ticker,
                    "action": "BUY",
                    "prob_up": round(fc["prob_up"], 4),
                    "prob_down": round(fc["prob_down"], 4),
                    "confidence": round(fc["confidence"], 3),
                    "shares": round(shares, 4),
                    "price": round(buy_price, 2),
                    "allocation": round(desired_alloc, 0),
                    "cash_after": round(cash, 0),
                    "n_holdings_after": len(holdings),
                    "weekly_return_pct": 0.0,
                    "net_pnl": 0.0,
                    "outcome": "",
                })

        # --- Step 3: mark-to-market at end of week ---
        holdings_value = 0.0
        for ticker, holding in holdings.items():
            price_end = next_prices.get(ticker, current_prices.get(ticker, holding["entry_price"]))
            holdings_value += holding["shares"] * price_end

        portfolio_value_new = cash + holdings_value

        # Weekly stop-loss (portfolio-level circuit breaker)
        prev_value = snapshots[-1]["portfolio_value"]
        weekly_loss_pct = (portfolio_value_new - prev_value) / prev_value if prev_value > 0 else 0.0
        if weekly_loss_pct < -stop_loss_pct:
            # Optional: could force-liquidate here. For now, just cap the reported loss.
            # (We keep actual holdings; the cap is informational only in this mode.)
            pass

        snapshots.append({
            "week": week_idx + 1,
            "date": str(next_monday.date()),
            "portfolio_value": portfolio_value_new,
            "cash": cash,
            "holdings_value": holdings_value,
            "n_holdings": len(holdings),
            "weekly_return_pct": (portfolio_value_new / prev_value - 1) * 100 if prev_value > 0 else 0.0,
            "wins": week_wins,
            "losses": week_losses,
        })

        # Invariants — these should never fail
        assert cash >= -1e-6, f"Cash went negative: {cash}"
        assert all(h["shares"] > 0 for h in holdings.values()), "Non-positive shares"

    return snapshots, trade_log


def get_benchmark_return(ticker: str, start: str, end: str,
                         prices_override: Optional[pd.DataFrame] = None) -> tuple[float, list[dict]]:
    """Get buy-and-hold return for any benchmark ticker."""
    mondays = _mondays_in_range(start, end)
    full_end = (pd.Timestamp(end) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")

    if prices_override is not None:
        df = prices_override.sort_index()
    else:
        try:
            from stochsignal.ingest.prices import get_price_history
            df = get_price_history(ticker, as_of=full_end, window_days=4000).sort_index()
        except Exception:
            return 0.0, []

    snapshots = []
    start_price = None
    for i, monday in enumerate(mondays):
        pit = df[df.index <= monday]
        if pit.empty:
            continue
        price = float(pit["Close"].iloc[-1])
        if start_price is None:
            start_price = price
        value = 100_000 * (price / start_price)
        snapshots.append({"week": i, "date": str(monday.date()), "value": value})

    if start_price and snapshots:
        total_return = (snapshots[-1]["value"] / 100_000 - 1) * 100
        return total_return, snapshots
    return 0.0, []


def get_spy_return(start: str, end: str) -> tuple[float, list[dict]]:
    """Get S&P 500 buy-and-hold return (backwards compat)."""
    ret, snaps = get_benchmark_return("SPY", start, end)
    spy_snaps = [{"week": s["week"], "date": s["date"], "spy_value": s["value"]} for s in snaps]
    return ret, spy_snaps


@click.command()
@click.option("--year", default=2023, type=int, show_default=True)
@click.option("--capital", default=100_000.0, show_default=True)
@click.option("--universe-dir", default=None,
              help="Path to config/universe_snapshots/ to use adaptive universe")
def main(year: int, capital: float, universe_dir: Optional[str]):
    """Backtest a full year using trained params, compare to S&P 500."""

    weights = load_weights()
    if weights is None:
        log.error("No trained weights found!")
        return

    start = f"{year}-01-02"
    end = f"{year}-12-31"

    # Load universe snapshots if provided
    universe_snapshots = None
    tickers = watchlist.all_tickers
    if universe_dir:
        u_path = Path(universe_dir)
        manifest = u_path / "manifest.json"
        if manifest.exists():
            universe_snapshots = json.loads(manifest.read_text())
            tickers = sorted({t for ts in universe_snapshots.values() for t in ts})
            log.info("Using adaptive universe: %d snapshot dates, %d unique tickers",
                     len(universe_snapshots), len(tickers))

    log.info("=" * 60)
    log.info("BACKTEST %d — StochSignal (portfolio-tracking)", year)
    log.info("=" * 60)

    snapshots, trade_log = run_replay(
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
        universe_snapshots=universe_snapshots,
    )

    spy_return, spy_snapshots = get_spy_return(start, end)
    spy_map = {s["date"]: s["spy_value"] for s in spy_snapshots}

    if not snapshots:
        print("No data!")
        return

    final_value = snapshots[-1]["portfolio_value"]
    total_return = (final_value / capital - 1) * 100
    n_weeks = len(snapshots) - 1
    annualised = ((final_value / capital) ** (52 / max(n_weeks, 1)) - 1) * 100
    total_trades = sum(1 for t in trade_log if t["action"] in ("BUY", "SELL"))
    n_buys = sum(1 for t in trade_log if t["action"] == "BUY")
    n_sells = sum(1 for t in trade_log if t["action"] == "SELL")
    wins = sum(1 for t in trade_log if t["action"] == "SELL" and t["outcome"] == "WIN")
    losses = sum(1 for t in trade_log if t["action"] == "SELL" and t["outcome"] == "LOSS")
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0

    values = [s["portfolio_value"] for s in snapshots]
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    spy_annual = ((1 + spy_return / 100) ** (52 / max(n_weeks, 1)) - 1) * 100 if n_weeks > 0 else 0

    # Print chart
    print(f"\n{'='*100}")
    print(f"  {year} FULL-YEAR BACKTEST — StochSignal (cash+holdings) vs S&P 500")
    print(f"{'='*100}")
    print(f"  Training: {weights.training_start} → {weights.training_end}")
    print(f"{'='*100}")
    print(f"\n  {'Week':<5} {'Date':<12} {'Portfolio':>12} {'Cash':>10} {'Hldgs$':>10} "
          f"{'#Hld':>4} {'Ret%':>7} {'SPY%':>7} {'Alpha':>7}")
    print(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*4} {'-'*7} {'-'*7} {'-'*7}")
    for s in snapshots:
        if s["week"] % 4 == 0 or s["week"] == n_weeks:
            model_pct = (s["portfolio_value"] / capital - 1) * 100
            spy_val = spy_map.get(s["date"], capital)
            spy_pct = (spy_val / capital - 1) * 100
            alpha = model_pct - spy_pct
            print(f"  {s['week']:<5} {s['date']:<12} ${s['portfolio_value']:>10,.0f} "
                  f"${s['cash']:>8,.0f} ${s['holdings_value']:>8,.0f} "
                  f"{s['n_holdings']:>4} {model_pct:>+6.2f}% {spy_pct:>+6.2f}% {alpha:>+6.2f}%")

    print(f"\n{'='*100}")
    print(f"  SUMMARY — {year}")
    print(f"{'='*100}")
    print(f"                    {'StochSignal':>15} {'S&P 500':>15} {'Alpha':>10}")
    print(f"  {'─'*60}")
    print(f"  Total return      {total_return:>+14.2f}% {spy_return:>+14.2f}% {total_return - spy_return:>+9.2f}%")
    print(f"  Annualised        {annualised:>+14.1f}% {spy_annual:>+14.1f}% {annualised - spy_annual:>+9.1f}%")
    print(f"  Max drawdown      {max_dd*100:>14.2f}%")
    print(f"  Win rate          {win_rate:>14.1f}%")
    print(f"  Buys / Sells      {n_buys:>7} / {n_sells:<7}")
    print(f"  Final cash        ${snapshots[-1]['cash']:>13,.0f}")
    print(f"  Final holdings    {snapshots[-1]['n_holdings']:>13} positions (${snapshots[-1]['holdings_value']:,.0f})")
    print(f"{'='*100}\n")

    # ---- Save CSVs ----
    weekly_csv = f"backtest_{year}_weekly.csv"
    with open(weekly_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(snapshots[0].keys()))
        writer.writeheader()
        writer.writerows(snapshots)
    print(f"  Saved weekly snapshots to {weekly_csv}")

    if trade_log:
        trades_csv = f"backtest_{year}_trades.csv"
        with open(trades_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(trade_log[0].keys()))
            writer.writeheader()
            writer.writerows(trade_log)
        print(f"  Saved per-trade decisions to {trades_csv}")


if __name__ == "__main__":
    main()
