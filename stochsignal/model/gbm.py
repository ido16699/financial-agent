"""Geometric Brownian Motion calibration and forecasting.

GBM: dS = μ S dt + σ S dW

Calibration: rolling MLE from daily log-returns.
  μ̂ = mean(r) * T          (annualised)
  σ̂ = std(r)  * sqrt(T)    (annualised), T = 252

Closed-form P(S_T > S_0) for the unperturbed model:
  log-return over horizon h is N((μ - σ²/2)·h, σ²·h)
  P(up) = Φ( (μ - σ²/2)·h / (σ·√h) )

MC paths are reserved for the perturbed model (see perturbation.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS_PER_YEAR = 252


@dataclass
class GBMParams:
    """Calibrated GBM parameters for one ticker."""
    ticker: str
    mu: float       # annualised drift
    sigma: float    # annualised volatility
    n_obs: int      # number of log-return observations used


def calibrate(ticker: str, log_returns: pd.Series) -> GBMParams:
    """Calibrate μ and σ from daily log-returns.

    Parameters
    ----------
    ticker:
        Symbol (for labelling only).
    log_returns:
        Daily log-return series (already computed by ingest.prices).

    Returns
    -------
    GBMParams with annualised μ and σ.
    """
    r = log_returns.dropna().values
    if len(r) < 20:
        raise ValueError(f"Too few log-returns for {ticker}: {len(r)} < 20")

    T = TRADING_DAYS_PER_YEAR
    mu = float(np.mean(r) * T)
    sigma = float(np.std(r, ddof=1) * np.sqrt(T))
    return GBMParams(ticker=ticker, mu=mu, sigma=sigma, n_obs=len(r))


def prob_up_closed_form(params: GBMParams, horizon_days: int) -> float:
    """P(S_T > S_0) under the unperturbed GBM using the closed-form formula.

    Parameters
    ----------
    params:
        Calibrated GBMParams.
    horizon_days:
        Forecast horizon in trading days.

    Returns
    -------
    Probability in [0, 1].
    """
    h = horizon_days / TRADING_DAYS_PER_YEAR
    mu_adj = params.mu - 0.5 * params.sigma ** 2
    if params.sigma < 1e-9:
        # Degenerate: deterministic drift
        return 1.0 if mu_adj > 0 else 0.0
    z = mu_adj * h / (params.sigma * np.sqrt(h))
    return float(norm.cdf(z))


def simulate_paths(
    params: GBMParams,
    horizon_days: int,
    n_paths: int,
    mu_override: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate GBM log-return over `horizon_days` for `n_paths` paths.

    Returns
    -------
    1-D array of shape (n_paths,) — total log-return over the horizon.
    Each element is log(S_T / S_0).
    """
    rng = rng or np.random.default_rng()
    mu = mu_override if mu_override is not None else params.mu
    h = horizon_days / TRADING_DAYS_PER_YEAR
    # log(S_T/S_0) = (μ - σ²/2)·h + σ·√h·Z,  Z ~ N(0,1)
    drift = (mu - 0.5 * params.sigma ** 2) * h
    diffusion = params.sigma * np.sqrt(h)
    z = rng.standard_normal(n_paths)
    return drift + diffusion * z


def prob_up_mc(
    params: GBMParams,
    horizon_days: int,
    n_paths: int,
    mu_override: float | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """P(S_T > S_0) via Monte Carlo — used for the perturbed model."""
    log_returns = simulate_paths(params, horizon_days, n_paths, mu_override=mu_override, rng=rng)
    return float(np.mean(log_returns > 0))
