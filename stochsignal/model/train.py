"""Training module: learn signal weights from historical data (2020-2025).

This is the "renormalization from backtest residuals" step.

For each ticker, walk-forward over 2020-2025:
  1. At each week, compute all AVAILABLE signals:
     - Sector momentum (price-based, point-in-time safe)
     - Wave analysis (FFT, OU, momentum — all price-based)
     - GBM baseline
  2. Record the signal values and the actual weekly outcome.
  3. After collecting all samples, run logistic regression:
     P(up) = sigmoid(β₀ + β_sector·sector_z + β_wave·wave_z + β_baseline·prob_up_baseline)
  4. The learned β coefficients become the optimized epsilons.

Note: News and Trends are NOT available point-in-time for 2020-2025,
so they are excluded from training. Their epsilons remain config defaults
and only activate in live/forward mode.

Output: a TrainedWeights object saved as JSON, loaded at prediction time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from stochsignal.config import settings, watchlist
from stochsignal.ingest.prices import get_price_history
from stochsignal.ingest.sector import fetch_sector_momentum
from stochsignal.model.gbm import calibrate, prob_up_closed_form, TRADING_DAYS_PER_YEAR
from stochsignal.model.waves import analyse_waves
from stochsignal.model.chaos import detect_regime
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)

WEIGHTS_PATH = Path(__file__).parent.parent.parent / "config" / "trained_weights.json"


@dataclass
class TrainedWeights:
    """Learned signal weights from historical backtest."""
    beta_intercept: float
    beta_gbm_prob: float       # weight on GBM baseline P(up)
    beta_sector: float         # weight on sector momentum z-score
    beta_wave: float           # weight on wave composite signal
    beta_regime: float         # weight on regime multiplier
    beta_market_mom: float     # weight on market-wide momentum (SPY)
    optimal_confidence_threshold: float
    avg_weekly_win: float      # average weekly return when right
    avg_weekly_loss: float     # average weekly return when wrong (positive)
    n_training_samples: int
    training_hit_rate: float
    training_start: str
    training_end: str
    # Risk management params (learned from validation period)
    learned_stop_loss_pct: float = 0.015
    learned_kelly_fraction: float = 0.5
    learned_max_allocation: float = 0.80
    learned_max_position_pct: float = 0.10
    # Signal combination weights (learned from validation)
    w_market_mom: float = 0.35
    w_wave: float = 0.25
    w_sector: float = 0.20
    w_gbm: float = 0.20
    # Interference (cross-term) coefficient — how much signal interactions matter
    # Positive = constructive interference amplifies agreement, destructive cancels disagreement
    w_interference: float = 0.0
    validation_start: str = ""
    validation_end: str = ""
    validation_return_pct: float = 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _log_loss(betas: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood for logistic regression with L2 regularization."""
    p = _sigmoid(X @ betas)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    # L2 regularization (skip intercept)
    reg = 0.01 * np.sum(betas[1:] ** 2)
    return nll + reg


def train(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    save: bool = True,
    prices_override: dict | None = None,
    universe_snapshots: dict | None = None,
    spy_override=None,
) -> TrainedWeights:
    """Train signal weights from historical walk-forward backtest.

    Parameters
    ----------
    tickers : list of tickers. Default: full watchlist.
              If universe_snapshots is provided, this is the union set.
    start / end : training date range. Default: settings backtest start/end.
    save : if True, save weights to config/trained_weights.json.
    prices_override : optional {ticker -> DataFrame} (from parquet snapshot).
                      Skips yfinance if provided.
    universe_snapshots : optional {date -> [tickers]} for adaptive universe.
                         At each training week, only samples from the most-recent
                         active universe are collected.
    spy_override : optional SPY DataFrame for market momentum.
    """
    tickers = tickers or watchlist.all_tickers
    start = start or settings.backtest_start
    end = end or settings.backtest_end

    # Build active-universe helper
    snap_dates = sorted(universe_snapshots.keys()) if universe_snapshots else None
    def _active_universe(monday: pd.Timestamp) -> set[str]:
        if not universe_snapshots:
            return set(tickers)
        eligible = [d for d in snap_dates if pd.Timestamp(d) <= monday]
        if not eligible:
            return set()
        return set(universe_snapshots[eligible[-1]])

    log.info("Training on %d tickers, %s → %s (adaptive_universe=%s)",
             len(tickers), start, end, bool(universe_snapshots))

    mondays = list(pd.date_range(start=start, end=end, freq="W-MON"))
    if len(mondays) < 10:
        raise ValueError("Too few weeks for training")

    # Collect training samples
    # Features: [1, gbm_prob, sector_z, wave_signal, regime_mult, market_mom]
    samples_X = []
    samples_y = []
    weekly_returns_right = []
    weekly_returns_wrong = []

    # Pre-fetch SPY for market momentum
    if spy_override is not None:
        spy_prices = spy_override.sort_index()
    else:
        try:
            full_end_ts = (pd.Timestamp(end) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            spy_prices = get_price_history("SPY", as_of=full_end_ts, window_days=2000)
            spy_prices = spy_prices.sort_index()
        except Exception:
            spy_prices = None
            log.warning("Could not fetch SPY for market momentum")

    for ticker in tickers:
        log.info("Training: collecting samples for %s ...", ticker)
        if prices_override is not None:
            if ticker not in prices_override:
                log.warning("No snapshot data for %s, skipping", ticker)
                continue
            all_prices = prices_override[ticker].sort_index()
        else:
            try:
                full_end = (pd.Timestamp(end) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
                all_prices = get_price_history(ticker, as_of=full_end, window_days=2000)
                all_prices = all_prices.sort_index()
            except Exception as exc:
                log.error("Could not fetch %s: %s", ticker, exc)
                continue

        for i in range(len(mondays) - 1):
            monday = mondays[i]
            next_monday = mondays[i + 1]

            # Adaptive universe: only collect samples if ticker is in active universe
            if universe_snapshots and ticker not in _active_universe(monday):
                continue

            # Point-in-time slice
            pit = all_prices[all_prices.index <= monday]
            if len(pit) < 80:
                continue

            closes = pit["Close"].dropna()
            close_vals = closes.values
            log_ret = np.log(close_vals[1:] / close_vals[:-1])
            log_ret_series = pd.Series(log_ret, index=closes.index[1:])

            try:
                params = calibrate(ticker, log_ret_series)
            except ValueError:
                continue

            # GBM baseline probability
            gbm_prob = prob_up_closed_form(params, settings.forecast_horizon_days)

            # Market momentum: SPY 20-day return z-scored
            market_mom = 0.0
            if spy_prices is not None:
                spy_pit = spy_prices[spy_prices.index <= monday]
                if len(spy_pit) >= 63:
                    spy_c = spy_pit["Close"].dropna().values
                    # 20-day SPY return
                    ret_20d = spy_c[-1] / spy_c[-21] - 1 if len(spy_c) > 21 else 0.0
                    # z-score over last ~60 days of 20d returns
                    rolling_rets = np.array([
                        spy_c[j] / spy_c[j - 20] - 1
                        for j in range(20, len(spy_c))
                    ])
                    if len(rolling_rets) > 5 and np.std(rolling_rets) > 1e-9:
                        market_mom = float((ret_20d - np.mean(rolling_rets)) / np.std(rolling_rets))
                        market_mom = np.clip(market_mom, -3, 3)

            # Sector momentum (point-in-time safe)
            try:
                sector_z = fetch_sector_momentum(ticker, as_of=monday)
            except Exception:
                sector_z = 0.0

            # Wave analysis (price-based, point-in-time safe)
            try:
                wave = analyse_waves(log_ret, close_vals, horizon_days=settings.forecast_horizon_days)
                wave_signal = wave.wave_signal
            except Exception:
                wave_signal = 0.0

            # Regime detection
            try:
                regime_info = detect_regime(log_ret)
                regime_mult = regime_info.confidence_multiplier
            except Exception:
                regime_mult = 1.0

            # Outcome
            future = all_prices[(all_prices.index > monday) & (all_prices.index <= next_monday)]
            if future.empty:
                continue
            price_start = float(close_vals[-1])
            price_end = float(future["Close"].iloc[-1])
            weekly_return = (price_end - price_start) / price_start
            outcome = int(price_end > price_start)

            # Feature row
            samples_X.append([1.0, gbm_prob, sector_z, wave_signal, regime_mult, market_mom])
            samples_y.append(outcome)

            # Track win/loss returns for Kelly
            predicted_up = gbm_prob >= 0.5
            actually_up = outcome == 1
            if predicted_up == actually_up:
                weekly_returns_right.append(abs(weekly_return))
            else:
                weekly_returns_wrong.append(abs(weekly_return))

    X = np.array(samples_X)
    y = np.array(samples_y)
    n = len(y)

    log.info("Training samples: %d  (%.1f%% positive)", n, np.mean(y) * 100)

    if n < 50:
        raise ValueError(f"Too few training samples: {n}")

    # Optimize logistic regression weights
    n_features = X.shape[1]
    x0 = np.zeros(n_features)
    result = minimize(_log_loss, x0, args=(X, y), method="L-BFGS-B")
    betas = result.x

    log.info("Learned betas: intercept=%.3f gbm=%.3f sector=%.3f wave=%.3f regime=%.3f mkt_mom=%.3f",
             betas[0], betas[1], betas[2], betas[3], betas[4], betas[5] if len(betas) > 5 else 0.0)

    # Compute training hit rate with learned model
    probs = _sigmoid(X @ betas)
    predicted = (probs >= 0.5).astype(int)
    hit_rate = float(np.mean(predicted == y))

    # Find optimal confidence threshold (maximize hit rate on confident predictions)
    best_threshold = 0.5
    best_hit = 0.0
    for thresh in np.arange(0.50, 0.80, 0.02):
        mask = np.abs(probs - 0.5) >= (thresh - 0.5)
        if mask.sum() < 20:
            continue
        hr = float(np.mean(predicted[mask] == y[mask]))
        if hr > best_hit:
            best_hit = hr
            best_threshold = float(thresh)

    # Average win/loss for Kelly
    avg_win = float(np.mean(weekly_returns_right)) if weekly_returns_right else 0.02
    avg_loss = float(np.mean(weekly_returns_wrong)) if weekly_returns_wrong else 0.02

    log.info(
        "Training complete: hit_rate=%.1f%%, optimal_threshold=%.2f, "
        "avg_win=%.3f%%, avg_loss=%.3f%%",
        hit_rate * 100, best_threshold, avg_win * 100, avg_loss * 100,
    )

    # Phase 2: Optimize risk management params on validation period (last 2 years)
    # Use 2024-2025 as validation (or last 2 years of the training range)
    val_start_ts = pd.Timestamp(end) - pd.Timedelta(days=730)  # ~2 years back
    val_start_str = max(val_start_ts, pd.Timestamp(start) + pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    val_end_str = end

    log.info("Phase 2: Optimizing risk params on validation period %s → %s", val_start_str, val_end_str)

    # Collect all prices for validation (reuse already-fetched data or snapshot)
    val_prices: dict[str, pd.DataFrame] = {}
    val_full_end = (pd.Timestamp(val_end_str) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    if prices_override is not None:
        val_prices = {t: df.sort_index() for t, df in prices_override.items() if t in tickers}
    else:
        for ticker in tickers:
            try:
                df = get_price_history(ticker, as_of=val_full_end, window_days=2000)
                val_prices[ticker] = df.sort_index()
            except Exception:
                pass

    if spy_override is not None:
        val_spy = spy_override.sort_index()
    else:
        val_spy = None
        try:
            val_spy = get_price_history("SPY", as_of=val_full_end, window_days=2000).sort_index()
        except Exception:
            pass

    risk_params = _optimize_risk_params(
        val_prices, val_spy, val_start_str, val_end_str, tickers, avg_win, avg_loss,
    )

    weights = TrainedWeights(
        beta_intercept=float(betas[0]),
        beta_gbm_prob=float(betas[1]),
        beta_sector=float(betas[2]),
        beta_wave=float(betas[3]),
        beta_regime=float(betas[4]),
        beta_market_mom=float(betas[5]),
        optimal_confidence_threshold=best_threshold,
        avg_weekly_win=avg_win,
        avg_weekly_loss=avg_loss,
        n_training_samples=n,
        training_hit_rate=hit_rate,
        training_start=start,
        training_end=end,
        # Learned risk params
        learned_stop_loss_pct=risk_params.get("stop_loss_pct", 0.015),
        learned_kelly_fraction=risk_params.get("kelly_fraction", 0.5),
        learned_max_allocation=risk_params.get("max_allocation", 0.80),
        learned_max_position_pct=risk_params.get("max_position_pct", 0.10),
        w_market_mom=risk_params.get("w_market_mom", 0.35),
        w_wave=risk_params.get("w_wave", 0.25),
        w_sector=risk_params.get("w_sector", 0.20),
        w_gbm=risk_params.get("w_gbm", 0.20),
        w_interference=risk_params.get("w_interference", 0.0),
        validation_start=val_start_str,
        validation_end=val_end_str,
        validation_return_pct=risk_params.get("val_return_pct", 0.0),
    )

    if save:
        WEIGHTS_PATH.write_text(json.dumps(asdict(weights), indent=2))
        log.info("Weights saved to %s", WEIGHTS_PATH)

    return weights


def _run_validation_replay(
    all_prices: dict[str, pd.DataFrame],
    spy_prices: pd.DataFrame | None,
    mondays: list[pd.Timestamp],
    tickers: list[str],
    w_market: float,
    w_wave: float,
    w_sector: float,
    w_gbm: float,
    w_interference: float,
    stop_loss_pct: float,
    kelly_frac: float,
    max_alloc: float,
    max_pos: float,
    avg_win: float,
    avg_loss: float,
    tc_rate: float,
) -> float:
    """Run a mini portfolio replay and return total return %."""
    from stochsignal.model.gbm import calibrate, prob_up_closed_form
    from stochsignal.model.waves import analyse_waves
    from stochsignal.model.chaos import detect_regime
    from stochsignal.ingest.sector import fetch_sector_momentum

    portfolio = 100_000.0
    n_weeks = len(mondays) - 1

    for week_idx in range(n_weeks):
        monday = mondays[week_idx]
        next_monday = mondays[week_idx + 1]

        # Market momentum
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

        week_pnl = 0.0
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

            # Composite signal with interference cross-terms
            gbm_rescaled = gbm_signal * 5
            linear = (
                w_market * market_mom +
                w_wave * wave_signal +
                w_sector * sector_z +
                w_gbm * gbm_rescaled
            )
            # Interference: cross-products of signal pairs
            # When two signals agree (same sign), product is positive → constructive
            # When they disagree, product is negative → destructive
            interference = (
                market_mom * wave_signal +
                market_mom * sector_z +
                market_mom * gbm_rescaled +
                wave_signal * sector_z +
                wave_signal * gbm_rescaled +
                sector_z * gbm_rescaled
            ) / 6.0  # normalize by number of pairs
            composite = linear + w_interference * interference
            prob_up = 1.0 / (1.0 + np.exp(-composite))

            # Confidence
            signals = [market_mom, wave_signal, sector_z, gbm_signal * 5]
            signs = [1 if s > 0 else -1 for s in signals if abs(s) > 0.1]
            agreement = abs(sum(signs)) / len(signs) if signs else 0.0
            signal_strength = min(abs(composite) / 1.5, 1.0)
            confidence = float(np.clip(agreement * signal_strength * regime_mult, 0, 1))

            # Direction and sizing
            if prob_up >= 0.5:
                direction = 1
                prob_win = prob_up
            else:
                direction = -1
                prob_win = 1.0 - prob_up

            edge = prob_win - 0.5
            if avg_loss < 1e-9:
                continue
            b = avg_win / avg_loss
            raw_f = (prob_win * b - (1 - prob_win)) / b
            adjusted_f = raw_f * kelly_frac * confidence * regime_mult
            adjusted_f = np.clip(adjusted_f, 0.0, max_pos)
            if adjusted_f < 0.003:
                continue

            allocation = adjusted_f * portfolio

            # Outcome
            future = prices[(prices.index > monday) & (prices.index <= next_monday)]
            if future.empty:
                continue
            price_start = float(close_vals[-1])
            price_end = float(future["Close"].iloc[-1])
            weekly_return = (price_end - price_start) / price_start

            gross_pnl = direction * weekly_return * allocation
            net_pnl = gross_pnl - (tc_rate * allocation)
            week_pnl += net_pnl

        # Stop-loss
        max_loss = portfolio * stop_loss_pct
        if week_pnl < -max_loss:
            week_pnl = -max_loss

        portfolio += week_pnl

    return (portfolio - 100_000.0) / 100_000.0 * 100


def _optimize_risk_params(
    all_prices: dict[str, pd.DataFrame],
    spy_prices: pd.DataFrame | None,
    val_start: str,
    val_end: str,
    tickers: list[str],
    avg_win: float,
    avg_loss: float,
) -> dict:
    """Grid search over risk management params on validation period.

    Uses only a subset of tickers (10 most liquid) and biweekly evaluation
    to keep runtime reasonable.
    """
    # Use biweekly mondays for speed (every other Monday)
    all_mondays = list(pd.date_range(start=val_start, end=val_end, freq="W-MON"))
    mondays = all_mondays[::2]  # every other Monday → ~52 weeks instead of 104
    if len(mondays) < 4:
        log.warning("Too few validation weeks, using defaults")
        return {}

    # Use only the 10 most liquid US tickers for validation speed
    liquid_tickers = [t for t in tickers if not t.endswith(".TA")][:10]
    if not liquid_tickers:
        liquid_tickers = tickers[:10]

    tc_rate = settings.transaction_cost_bps / 10_000

    best_return = -999.0
    best_params = {}

    param_grid = [
        # (stop_loss, kelly_frac, max_alloc, max_pos, w_mkt, w_wave, w_sector, w_gbm, w_interf)
        (sl, kf, 0.80, 0.10, wm, ww, ws, wg, wi)
        for sl in [0.005, 0.01, 0.02, 0.04]
        for kf in [0.5, 0.75, 1.0]
        for wm, ww, ws, wg in [
            (0.35, 0.25, 0.20, 0.20),
            (0.25, 0.25, 0.25, 0.25),
        ]
        for wi in [0.0, 0.15, 0.30]  # interference: none, moderate, strong
    ]

    log.info("Validation grid search: %d combos on %s → %s (%d biweekly periods, %d tickers)",
             len(param_grid), val_start, val_end, len(mondays) - 1, len(liquid_tickers))

    for i, (sl, kf, ma, mp, wm, ww, ws, wg, wi) in enumerate(param_grid):
        ret = _run_validation_replay(
            all_prices, spy_prices, mondays, liquid_tickers,
            wm, ww, ws, wg, wi, sl, kf, ma, mp, avg_win, avg_loss, tc_rate,
        )
        if ret > best_return:
            best_return = ret
            best_params = {
                "stop_loss_pct": sl,
                "kelly_fraction": kf,
                "max_allocation": ma,
                "max_position_pct": mp,
                "w_market_mom": wm,
                "w_wave": ww,
                "w_sector": ws,
                "w_gbm": wg,
                "w_interference": wi,
                "val_return_pct": ret,
            }
        log.info("  Combo %d/%d: sl=%.3f kf=%.2f wm=%.2f interf=%.2f → ret=%.2f%% (best=%.2f%%)",
                 i + 1, len(param_grid), sl, kf, wm, wi, ret, best_return)

    log.info("Validation best: %.2f%% return with params %s", best_return, best_params)
    return best_params


def load_weights() -> TrainedWeights | None:
    """Load trained weights from config/trained_weights.json if available."""
    if not WEIGHTS_PATH.exists():
        log.warning("No trained weights found at %s — using config defaults.", WEIGHTS_PATH)
        return None
    raw = json.loads(WEIGHTS_PATH.read_text())
    return TrainedWeights(**raw)


def predict_with_weights(
    weights: TrainedWeights,
    gbm_prob: float,
    sector_z: float,
    wave_signal: float,
    regime_mult: float,
    market_mom: float = 0.0,
) -> float:
    """Predict P(up) using trained logistic regression weights.

    Returns probability in [0, 1].
    """
    x = np.array([1.0, gbm_prob, sector_z, wave_signal, regime_mult, market_mom])
    betas = np.array([
        weights.beta_intercept,
        weights.beta_gbm_prob,
        weights.beta_sector,
        weights.beta_wave,
        weights.beta_regime,
        weights.beta_market_mom,
    ])
    return float(_sigmoid(x @ betas))
