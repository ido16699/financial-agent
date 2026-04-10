"""Scoring metrics for the backtest: hit rate, Brier score, calibration.

All functions operate on arrays of predictions and outcomes.

Glossary
--------
prob_up : float in [0, 1] — model's predicted probability of up move.
outcome  : int {0, 1}     — 1 if the stock actually went up, 0 if down/flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResults:
    """Aggregate scoring results for one ticker or the full watchlist."""
    ticker: str
    n_predictions: int
    hit_rate: float          # fraction of correct directional calls
    brier_score: float       # mean squared probability error
    # For calibration plot: bin edges and (mean_prob, frac_positive) per bin
    calibration_bins: list[tuple[float, float]] = field(default_factory=list)
    # Simulated PnL (in %, net of transaction costs)
    simulated_pnl_pct: float = 0.0
    # Predictions detail for further analysis
    predictions: pd.DataFrame | None = None


def hit_rate(prob_up: np.ndarray, outcome: np.ndarray) -> float:
    """Fraction of correct directional calls.

    Predicted direction = 1 if prob_up >= 0.5, else 0.
    """
    predicted = (prob_up >= 0.5).astype(int)
    return float(np.mean(predicted == outcome))


def brier_score(prob_up: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    return float(np.mean((prob_up - outcome) ** 2))


def calibration_curve(
    prob_up: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 10,
) -> list[tuple[float, float]]:
    """Compute calibration curve (reliability diagram data).

    Returns list of (mean_predicted_prob, fraction_positive) per bin.
    Bins with no predictions are omitted.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    result = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob_up >= lo) & (prob_up < hi)
        if mask.sum() == 0:
            continue
        mean_prob = float(prob_up[mask].mean())
        frac_pos = float(outcome[mask].mean())
        result.append((mean_prob, frac_pos))
    return result


def simulated_pnl(
    prob_up: np.ndarray,
    weekly_log_returns: np.ndarray,
    transaction_cost_bps: int = 10,
) -> float:
    """Simulate paper-trading PnL (net of transaction costs) in %.

    Strategy: go long if prob_up >= 0.5, short otherwise.
    Each trade costs `transaction_cost_bps` basis points round-trip.
    Returns cumulative percentage return over all weeks.
    """
    direction = np.where(prob_up >= 0.5, 1.0, -1.0)
    tc = transaction_cost_bps / 10_000
    gross_returns = direction * weekly_log_returns
    net_returns = gross_returns - tc
    cumulative = float(np.sum(net_returns) * 100)
    return cumulative


def score_ticker(
    ticker: str,
    prob_up: np.ndarray,
    outcome: np.ndarray,
    weekly_log_returns: np.ndarray,
    transaction_cost_bps: int = 10,
    dates: pd.DatetimeIndex | None = None,
) -> BacktestResults:
    """Compute all scores for a single ticker."""
    n = len(prob_up)
    hr = hit_rate(prob_up, outcome)
    bs = brier_score(prob_up, outcome)
    cal = calibration_curve(prob_up, outcome)
    pnl = simulated_pnl(prob_up, weekly_log_returns, transaction_cost_bps)

    detail_df = pd.DataFrame({
        "date": dates if dates is not None else range(n),
        "prob_up": prob_up,
        "outcome": outcome,
        "weekly_log_return": weekly_log_returns,
    })

    return BacktestResults(
        ticker=ticker,
        n_predictions=n,
        hit_rate=hr,
        brier_score=bs,
        calibration_bins=cal,
        simulated_pnl_pct=pnl,
        predictions=detail_df,
    )


def print_summary(results: list[BacktestResults]) -> None:
    """Print a human-readable summary table to stdout."""
    header = f"{'Ticker':<12} {'N':>5} {'Hit%':>7} {'Brier':>7} {'PnL%':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.ticker:<12} {r.n_predictions:>5} "
            f"{r.hit_rate*100:>6.1f}% {r.brier_score:>7.4f} {r.simulated_pnl_pct:>7.2f}%"
        )
    # Aggregate
    all_hr = np.mean([r.hit_rate for r in results])
    all_bs = np.mean([r.brier_score for r in results])
    all_pnl = sum(r.simulated_pnl_pct for r in results)
    print("-" * len(header))
    print(f"{'AGGREGATE':<12} {'':>5} {all_hr*100:>6.1f}% {all_bs:>7.4f} {all_pnl:>7.2f}%")
