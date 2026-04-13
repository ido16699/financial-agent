"""Perturbation series on the drift, now powered by Heston + chaos regime.

Conceptual model
----------------
μ(ε) = μ₀ + Δ₁ + Δ₂

Order-1 (linear response to signals):
  Δ₁ = ε_n·sentiment + ε_t·trends + ε_s·sector

Order-2 (saturation + cross-term reinforcement):
  Δ₂ = −ε_n²·s² − ε_t²·t² − ε_s²·x²
       + ε_n·ε_t·s·t + ε_n·ε_s·s·x + ε_t·ε_s·t·x

Confidence (series convergence + regime):
  base_conf = clip(1 − |Δ₂|/|Δ₁|, 0, 1)
  confidence = base_conf × regime_multiplier

Price range from MC simulation (Heston if available, GBM fallback).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stochsignal.config import settings
from stochsignal.model.gbm import GBMParams, simulate_paths
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class PerturbedForecast:
    """Full result of the perturbed model for one ticker."""
    ticker: str
    mu_0: float
    delta_1: float
    delta_2: float
    mu_perturbed: float
    prob_up: float
    confidence: float
    sentiment: float
    trends_zscore: float
    sector_zscore: float = 0.0
    regime: str = "unknown"
    lyapunov: float = 0.0
    regime_multiplier: float = 1.0
    vol_regime: str = "normal"
    current_sigma: float = 0.0
    # Price range from MC
    range_floor_pct: float = 0.0
    range_ceil_pct: float = 0.0
    range_median_pct: float = 0.0


def compute(
    params: GBMParams,
    sentiment: float,
    trends_zscore: float,
    sector_zscore: float = 0.0,
    regime_multiplier: float = 1.0,
    regime: str = "unknown",
    lyapunov: float = 0.0,
    vol_regime: str = "normal",
    current_sigma: float = 0.0,
    heston_params=None,
    horizon_days: int | None = None,
    n_paths: int | None = None,
    rng: np.random.Generator | None = None,
) -> PerturbedForecast:
    """Run the perturbation series and return a PerturbedForecast.

    If heston_params is provided, uses Heston MC. Otherwise falls back to GBM MC.
    """
    horizon_days = horizon_days or settings.forecast_horizon_days
    n_paths = n_paths or settings.mc_paths

    eps_n = settings.epsilon_news
    eps_t = settings.epsilon_trends
    eps_s = settings.epsilon_sector

    s, t, x = sentiment, trends_zscore, sector_zscore

    # --- First-order perturbation ---
    delta_1 = eps_n * s + eps_t * t + eps_s * x

    # --- Second-order perturbation ---
    delta_2 = (
        -(eps_n ** 2) * s ** 2
        - (eps_t ** 2) * t ** 2
        - (eps_s ** 2) * x ** 2
        + (eps_n * eps_t) * s * t
        + (eps_n * eps_s) * s * x
        + (eps_t * eps_s) * t * x
    )

    mu_perturbed = params.mu + delta_1 + delta_2

    # --- Confidence: series convergence × regime ---
    abs_d1 = abs(delta_1)
    abs_d2 = abs(delta_2)
    if abs_d1 < 1e-9:
        base_conf = 0.0
    else:
        base_conf = float(np.clip(1.0 - abs_d2 / abs_d1, 0.0, 1.0))
    confidence = float(np.clip(base_conf * regime_multiplier, 0.0, 1.0))

    # --- MC simulation: Heston or GBM ---
    if heston_params is not None:
        from stochsignal.model.heston import simulate_heston
        mc_log_returns = simulate_heston(
            heston_params, horizon_days=horizon_days, n_paths=n_paths,
            mu_override=mu_perturbed, rng=rng,
        )
    else:
        mc_log_returns = simulate_paths(
            params, horizon_days=horizon_days, n_paths=n_paths,
            mu_override=mu_perturbed, rng=rng,
        )

    prob = float(np.mean(mc_log_returns > 0))

    mc_pct_changes = (np.exp(mc_log_returns) - 1) * 100
    range_floor = float(np.percentile(mc_pct_changes, 5))
    range_ceil = float(np.percentile(mc_pct_changes, 95))
    range_median = float(np.percentile(mc_pct_changes, 50))

    log.info(
        "%s  μ*=%.4f  P(up)=%.3f  conf=%.2f  regime=%s  "
        "range=[%.1f%%, %.1f%%]  sent=%.2f sect=%.2f",
        params.ticker, mu_perturbed, prob, confidence, regime,
        range_floor, range_ceil, sentiment, sector_zscore,
    )

    return PerturbedForecast(
        ticker=params.ticker,
        mu_0=params.mu,
        delta_1=delta_1,
        delta_2=delta_2,
        mu_perturbed=mu_perturbed,
        prob_up=prob,
        confidence=confidence,
        sentiment=sentiment,
        trends_zscore=trends_zscore,
        sector_zscore=sector_zscore,
        regime=regime,
        lyapunov=lyapunov,
        regime_multiplier=regime_multiplier,
        vol_regime=vol_regime,
        current_sigma=current_sigma,
        range_floor_pct=range_floor,
        range_ceil_pct=range_ceil,
        range_median_pct=range_median,
    )
