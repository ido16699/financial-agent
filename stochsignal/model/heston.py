"""Heston stochastic volatility model.

The Heston model adds a mean-reverting variance process:

  dS = μ S dt + √v S dW₁
  dv = κ(θ − v)dt + ξ√v dW₂
  corr(dW₁, dW₂) = ρ

Parameters:
  κ (kappa)  : variance mean-reversion speed
  θ (theta)  : long-run variance
  ξ (xi)     : vol-of-vol (volatility of the variance)
  ρ (rho)    : correlation between price and variance (leverage effect)
  v₀         : initial variance (current estimate)

Calibration:
  We estimate parameters from historical returns using method-of-moments
  on realized variance time series. Not as precise as option-calibration
  but works with price data only.

Usage:
  params = calibrate_heston(log_returns)
  paths = simulate_heston(params, horizon_days, n_paths)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stochsignal.model.gbm import TRADING_DAYS_PER_YEAR
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class HestonParams:
    """Calibrated Heston model parameters."""
    ticker: str
    mu: float          # annualised drift
    kappa: float       # variance mean-reversion speed
    theta: float       # long-run variance (annualised)
    xi: float          # vol-of-vol
    rho: float         # price-variance correlation
    v0: float          # current variance estimate


def calibrate_heston(
    ticker: str,
    log_returns: pd.Series,
    rv_window: int = 21,
) -> HestonParams:
    """Calibrate Heston params from daily log-returns using method-of-moments.

    Steps:
    1. Compute rolling realized variance (RV) over `rv_window` days.
    2. Estimate κ and θ from the RV series using AR(1) regression:
       RV(t+1) = a + b·RV(t) → κ = -log(b)·252, θ = a/(1-b)
    3. Estimate ξ from the std of RV changes.
    4. Estimate ρ from correlation between returns and RV changes.
    5. v₀ = most recent RV value.
    """
    r = log_returns.dropna().values
    if len(r) < rv_window + 20:
        raise ValueError(f"Too few returns for Heston calibration: {len(r)}")

    T = TRADING_DAYS_PER_YEAR

    # Drift
    mu = float(np.mean(r) * T)

    # Rolling realized variance (annualised)
    rv_series = pd.Series(r).rolling(rv_window).var().dropna().values * T
    if len(rv_series) < 20:
        raise ValueError(f"Too few RV observations for Heston: {len(rv_series)}")

    # AR(1) on RV: RV(t+1) = a + b*RV(t)
    rv_x = rv_series[:-1]
    rv_y = rv_series[1:]
    # OLS: b = cov(x,y)/var(x), a = mean(y) - b*mean(x)
    cov_xy = np.cov(rv_x, rv_y)[0, 1]
    var_x = np.var(rv_x)
    if var_x < 1e-15:
        # Variance is constant — degenerate to GBM
        return HestonParams(
            ticker=ticker, mu=mu, kappa=1.0,
            theta=float(np.mean(rv_series)),
            xi=0.01, rho=0.0, v0=float(rv_series[-1]),
        )

    b = cov_xy / var_x
    b = np.clip(b, 0.001, 0.999)  # ensure mean-reversion
    a = np.mean(rv_y) - b * np.mean(rv_x)

    # Convert AR(1) to continuous-time
    # b = exp(-κ/T) → κ = -ln(b) * T
    kappa = float(-np.log(b) * T)
    kappa = max(kappa, 0.1)  # floor to avoid pathological cases
    theta = float(a / (1 - b))
    theta = max(theta, 0.001)

    # Vol-of-vol: std of RV changes, scaled
    rv_changes = np.diff(rv_series)
    xi = float(np.std(rv_changes) * np.sqrt(T))
    xi = np.clip(xi, 0.01, 5.0)

    # Leverage effect: correlation between returns and variance changes
    min_len = min(len(r[rv_window:]), len(rv_changes))
    returns_aligned = r[rv_window:rv_window + min_len]
    rv_changes_aligned = rv_changes[:min_len]
    if len(returns_aligned) > 10:
        rho = float(np.corrcoef(returns_aligned, rv_changes_aligned)[0, 1])
        rho = np.clip(rho, -0.99, 0.99)
    else:
        rho = -0.7  # typical equity leverage effect

    v0 = float(rv_series[-1])

    # Feller condition check: 2κθ > ξ² (ensures v stays positive)
    feller = 2 * kappa * theta / (xi ** 2) if xi > 0 else float("inf")

    log.info(
        "Heston %s: κ=%.2f θ=%.4f ξ=%.2f ρ=%.2f v₀=%.4f σ₀=%.1f%% Feller=%.2f",
        ticker, kappa, theta, xi, rho, v0, np.sqrt(v0) * 100, feller,
    )

    return HestonParams(
        ticker=ticker, mu=mu, kappa=kappa, theta=theta,
        xi=xi, rho=rho, v0=v0,
    )


def simulate_heston(
    params: HestonParams,
    horizon_days: int,
    n_paths: int,
    mu_override: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate Heston model paths via Euler-Maruyama discretisation.

    Returns
    -------
    1-D array of shape (n_paths,) — total log-return over the horizon.
    """
    rng = rng or np.random.default_rng()
    mu = mu_override if mu_override is not None else params.mu
    dt = 1.0 / TRADING_DAYS_PER_YEAR

    # Correlated Brownian increments
    z1 = rng.standard_normal((n_paths, horizon_days))
    z2 = rng.standard_normal((n_paths, horizon_days))
    w1 = z1
    w2 = params.rho * z1 + np.sqrt(1 - params.rho ** 2) * z2

    log_S = np.zeros(n_paths)
    v = np.full(n_paths, params.v0)

    for t in range(horizon_days):
        v_pos = np.maximum(v, 0)  # reflection scheme for positivity
        sqrt_v = np.sqrt(v_pos)

        # Log-price SDE
        log_S += (mu - 0.5 * v_pos) * dt + sqrt_v * np.sqrt(dt) * w1[:, t]

        # Variance SDE
        v += params.kappa * (params.theta - v_pos) * dt + params.xi * sqrt_v * np.sqrt(dt) * w2[:, t]

    return log_S


def heston_prob_up(
    params: HestonParams,
    horizon_days: int,
    n_paths: int,
    mu_override: float | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """P(S_T > S_0) via Heston Monte Carlo."""
    log_returns = simulate_heston(params, horizon_days, n_paths, mu_override=mu_override, rng=rng)
    return float(np.mean(log_returns > 0))


def current_vol_regime(params: HestonParams) -> str:
    """Classify current volatility regime relative to long-run mean.

    Returns 'low', 'normal', or 'high'.
    """
    ratio = params.v0 / params.theta if params.theta > 0 else 1.0
    if ratio < 0.7:
        return "low"
    elif ratio > 1.5:
        return "high"
    return "normal"
