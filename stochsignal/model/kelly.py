"""Kelly criterion portfolio sizing.

The Kelly criterion maximises long-run geometric growth rate.

Full Kelly fraction:
  f* = (p · b - q) / b

where:
  p = probability of winning (prob_up or 1-prob_up depending on direction)
  q = 1 - p
  b = odds ratio (average win / average loss)

In practice, full Kelly is too aggressive — we use fractional Kelly:
  f = fraction · f*

Default fraction = 0.25 (quarter-Kelly), which is common in practice.
This gives ~75% of the growth rate with ~50% of the variance.

Position sizing:
  For each ticker, the allocated capital = f * total_portfolio
  Summed across tickers must not exceed leverage limit (1.0 = no leverage).

Risk controls:
  - Max position per ticker: 20% of portfolio
  - Min position: skip if Kelly says < 1%
  - Regime multiplier from chaos.py adjusts confidence → adjusts Kelly
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stochsignal.logging_utils import get_logger

log = get_logger(__name__)

# Default fractional Kelly
DEFAULT_KELLY_FRACTION = 1.0
MAX_POSITION_PCT = 0.10
MIN_POSITION_PCT = 0.003
MAX_TOTAL_ALLOCATION = 0.85  # deploy up to 85% of capital


@dataclass
class PositionSize:
    """Sized position for one ticker."""
    ticker: str
    direction: int         # +1 long, -1 short
    kelly_fraction: float  # raw Kelly f*
    position_pct: float    # actual allocation after fractional + caps
    capital_allocated: float
    edge: float            # estimated edge (p - 0.5)
    confidence: float


def kelly_fraction(
    prob_win: float,
    avg_win: float = 1.0,
    avg_loss: float = 1.0,
) -> float:
    """Compute raw Kelly fraction.

    Parameters
    ----------
    prob_win : probability of winning the trade (0 to 1)
    avg_win  : average gain when right (as multiple, e.g. 1.02 for 2%)
    avg_loss : average loss when wrong (as positive multiple, e.g. 1.01 for 1%)

    Returns
    -------
    Optimal fraction of capital to bet. Can be negative (means don't bet).
    """
    if avg_loss < 1e-9:
        return 0.0
    b = avg_win / avg_loss  # odds ratio
    q = 1.0 - prob_win
    f = (prob_win * b - q) / b
    return f


def size_positions(
    forecasts: list[dict],
    total_capital: float,
    kelly_mult: float = DEFAULT_KELLY_FRACTION,
    max_position_pct: float = MAX_POSITION_PCT,
    max_total_alloc: float = MAX_TOTAL_ALLOCATION,
    regime_multipliers: dict[str, float] | None = None,
) -> list[PositionSize]:
    """Size positions for a portfolio using fractional Kelly.

    Parameters
    ----------
    forecasts : list of dicts with keys:
        ticker, prob_up, confidence, regime_mult (optional),
        avg_weekly_win, avg_weekly_loss
    total_capital : current portfolio value
    kelly_mult : fractional Kelly multiplier (0.25 = quarter Kelly)
    max_position_pct : max allocation per ticker
    max_total_alloc : max total portfolio allocation
    regime_multipliers : ticker → chaos.py confidence_multiplier

    Returns
    -------
    List of PositionSize objects, sorted by allocation descending.
    """
    regime_multipliers = regime_multipliers or {}
    positions = []

    for fc in forecasts:
        ticker = fc["ticker"]
        prob_up = fc["prob_up"]
        confidence = fc["confidence"]
        avg_win = fc.get("avg_weekly_win", 0.02)   # default 2%
        avg_loss = fc.get("avg_weekly_loss", 0.02)  # default 2%
        regime_mult = regime_multipliers.get(ticker, 1.0)

        # Direction
        if prob_up >= 0.5:
            direction = 1
            prob_win = prob_up
        else:
            direction = -1
            prob_win = 1.0 - prob_up

        # Edge: how far from 50/50
        edge = prob_win - 0.5

        # Raw Kelly
        raw_f = kelly_fraction(prob_win, avg_win, avg_loss)

        # Apply fractional Kelly + confidence + regime
        adjusted_f = raw_f * kelly_mult * confidence * regime_mult

        # Clamp
        adjusted_f = np.clip(adjusted_f, 0.0, max_position_pct)

        if adjusted_f < MIN_POSITION_PCT:
            continue

        capital_alloc = adjusted_f * total_capital

        positions.append(PositionSize(
            ticker=ticker,
            direction=direction,
            kelly_fraction=raw_f,
            position_pct=adjusted_f,
            capital_allocated=capital_alloc,
            edge=edge,
            confidence=confidence,
        ))

    # Sort by allocation descending
    positions.sort(key=lambda p: p.position_pct, reverse=True)

    # Cap total allocation (risk management)
    total_alloc = sum(p.position_pct for p in positions)
    if total_alloc > max_total_alloc:
        scale = max_total_alloc / total_alloc
        for p in positions:
            p.position_pct *= scale
            p.capital_allocated = p.position_pct * total_capital

    log.info(
        "Kelly sizing: %d positions, total alloc=%.1f%%, top=%s@%.1f%%",
        len(positions),
        sum(p.position_pct for p in positions) * 100,
        positions[0].ticker if positions else "none",
        positions[0].position_pct * 100 if positions else 0,
    )

    return positions
