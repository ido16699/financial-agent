"""Generate next-week forecasts using the fully trained engine.

All parameters come from trained_weights.json (learned 2020-2025).
No hand-tuning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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


def main():
    weights = load_weights()
    if weights is None:
        print("No trained weights found!")
        return

    as_of = pd.Timestamp.now().normalize()
    tickers = watchlist.all_tickers

    # Fetch SPY for market momentum
    try:
        spy_df = get_price_history("SPY", as_of=as_of.strftime("%Y-%m-%d"), window_days=2000).sort_index()
        spy_c = spy_df["Close"].dropna().values
        ret_20d = spy_c[-1] / spy_c[-21] - 1
        rolling_rets = np.array([spy_c[j] / spy_c[j - 20] - 1 for j in range(20, len(spy_c))])
        market_mom = float(np.clip(
            (ret_20d - np.mean(rolling_rets)) / np.std(rolling_rets), -3, 3
        ))
    except Exception:
        market_mom = 0.0

    forecasts = []
    regime_multipliers = {}

    for ticker in tickers:
        try:
            prices = get_price_history(ticker, as_of=as_of.strftime("%Y-%m-%d"), window_days=2000).sort_index()
        except Exception:
            continue
        if len(prices) < 80:
            continue

        closes = prices["Close"].dropna()
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
            heston_params = calibrate_heston(ticker, log_ret_series)
            vol_regime = current_vol_regime(heston_params)
            current_sigma = float(np.sqrt(heston_params.v0))
        except Exception:
            vol_regime = "normal"
            current_sigma = gbm_params.sigma

        try:
            sector_z = fetch_sector_momentum(ticker, as_of=as_of)
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
            regime_name = regime_info.regime
            lyapunov = regime_info.lyapunov_exponent
        except Exception:
            regime_mult = 1.0
            regime_name = "unknown"
            lyapunov = 0.0

        regime_multipliers[ticker] = regime_mult

        # Composite with interference (all weights learned)
        gbm_rescaled = gbm_signal * 5
        linear = (
            weights.w_market_mom * market_mom +
            weights.w_wave * wave_signal +
            weights.w_sector * sector_z +
            weights.w_gbm * gbm_rescaled
        )
        interference = (
            market_mom * wave_signal +
            market_mom * sector_z +
            market_mom * gbm_rescaled +
            wave_signal * sector_z +
            wave_signal * gbm_rescaled +
            sector_z * gbm_rescaled
        ) / 6.0
        composite = linear + weights.w_interference * interference
        prob_up = 1.0 / (1.0 + np.exp(-composite))

        # Confidence
        signals = [market_mom, wave_signal, sector_z, gbm_signal * 5]
        signs = [1 if s > 0 else -1 for s in signals if abs(s) > 0.1]
        agreement = abs(sum(signs)) / len(signs) if signs else 0.0
        signal_strength = min(abs(composite) / 1.5, 1.0)
        confidence = float(np.clip(agreement * signal_strength * regime_mult, 0, 1))

        direction = "LONG" if prob_up >= 0.5 else "SHORT"
        current_price = float(close_vals[-1])

        forecasts.append({
            "ticker": ticker,
            "prob_up": prob_up,
            "confidence": confidence,
            "avg_weekly_win": weights.avg_weekly_win,
            "avg_weekly_loss": weights.avg_weekly_loss,
            "direction": direction,
            "current_price": current_price,
            "market_mom": market_mom,
            "wave_signal": wave_signal,
            "sector_z": sector_z,
            "gbm_prob": gbm_prob,
            "regime": regime_name,
            "regime_mult": regime_mult,
            "vol_regime": vol_regime,
            "sigma": current_sigma,
        })

    # Kelly sizing
    positions = size_positions(
        forecasts, 100_000,
        kelly_mult=weights.learned_kelly_fraction,
        max_position_pct=weights.learned_max_position_pct,
        max_total_alloc=weights.learned_max_allocation,
        regime_multipliers=regime_multipliers,
    )
    pos_map = {p.ticker: p for p in positions}

    # Print
    print(f"\n{'='*80}")
    print(f"  STOCHSIGNAL — NEXT WEEK FORECAST (as of {as_of.date()})")
    print(f"  Learned params: stop_loss={weights.learned_stop_loss_pct*100:.1f}%  kelly={weights.learned_kelly_fraction:.1f}  max_alloc={weights.learned_max_allocation*100:.0f}%")
    print(f"  Market momentum (SPY 20d z-score): {market_mom:+.2f}")
    print(f"{'='*80}")
    print(f"  {'Ticker':<10} {'Dir':>5} {'P(up)':>7} {'Conf':>6} {'Alloc%':>7} {'Price':>10} {'Regime':>12} {'Vol':>6} {'Signals'}")
    print(f"  {'-'*10} {'-'*5} {'-'*7} {'-'*6} {'-'*7} {'-'*10} {'-'*12} {'-'*6} {'-'*30}")

    # Sort by allocation
    sorted_fc = sorted(forecasts, key=lambda f: pos_map[f["ticker"]].position_pct if f["ticker"] in pos_map else 0, reverse=True)

    active_count = 0
    for fc in sorted_fc:
        pos = pos_map.get(fc["ticker"])
        alloc_pct = pos.position_pct * 100 if pos else 0.0
        if alloc_pct < 0.01:
            continue
        active_count += 1
        sig_str = f"mkt={fc['market_mom']:+.1f} wave={fc['wave_signal']:+.2f} sec={fc['sector_z']:+.2f} gbm={fc['gbm_prob']:.2f}"
        print(f"  {fc['ticker']:<10} {fc['direction']:>5} {fc['prob_up']:>7.1%} {fc['confidence']:>6.2f} {alloc_pct:>6.1f}% {fc['current_price']:>10.2f} {fc['regime']:>12} {fc['sigma']:>5.1f}% {sig_str}")

    print(f"  {'-'*80}")

    total_alloc = sum(p.position_pct for p in positions) * 100
    n_long = sum(1 for f in forecasts if f["ticker"] in pos_map and f["direction"] == "LONG")
    n_short = sum(1 for f in forecasts if f["ticker"] in pos_map and f["direction"] == "SHORT")
    print(f"  Active positions: {active_count} ({n_long} long, {n_short} short)")
    print(f"  Total allocation: {total_alloc:.1f}%  |  Cash reserve: {100-total_alloc:.1f}%")
    print(f"  Stop-loss cap: {weights.learned_stop_loss_pct*100:.1f}% max weekly loss")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
