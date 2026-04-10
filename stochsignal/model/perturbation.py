"""Perturbation series on the GBM drift.

Conceptual model
----------------
μ(ε) = μ₀ + ε·μ₁ + ε²·μ₂

where ε is a small expansion parameter (implicitly 1 — the ε factors are
absorbed into the coupling constants from settings).

Order-0 (μ₀): raw GBM drift from calibration.

Order-1 (μ₁): linear response to external signals.
  Δ₁ = ε_n · sentiment + ε_t · trends_zscore + ε_s · sector_zscore

Order-2 (μ₂): quadratic correction (saturation + cross-signal interaction).
  Sign conventions:
    - Diagonal terms are negative (saturation: extreme signals are less reliable)
    - Cross terms are positive (signals reinforce each other)
  Δ₂ = −ε_n² · sentiment²  − ε_t² · trends²  − ε_s² · sector²
       + ε_n·ε_t · sentiment·trends
       + ε_n·ε_s · sentiment·sector
       + ε_t·ε_s · trends·sector

Confidence (series convergence proxy):
  confidence = clip(1 − |Δ₂| / |Δ₁|, 0, 1)
  When |Δ₁| ≈ 0, the first-order signal is negligible → confidence = 0.

Perturbed drift:
  mu_perturbed = mu_0 + Δ₁ + Δ₂

The perturbed drift is passed to gbm.simulate_paths() via mu_override.
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
    mu_0: float           # raw GBM drift
    delta_1: float        # first-order perturbation
    delta_2: float        # second-order perturbation
    mu_perturbed: float   # mu_0 + delta_1 + delta_2
    prob_up: float        # MC estimate of P(S_T > S_0)
    confidence: float     # series-convergence confidence in [0, 1]
    sentiment: float      # raw VADER score
    trends_zscore: float  # raw Google Trends z-score
    sector_zscore: float = 0.0  # sector/index momentum z-score
    # Price range: expected % change at 5th and 95th percentile
    range_floor_pct: float = 0.0   # 5th percentile weekly return (%)
    range_ceil_pct: float = 0.0    # 95th percentile weekly return (%)
    range_median_pct: float = 0.0  # 50th percentile weekly return (%)


def compute(
    params: GBMParams,
    sentiment: float,
    trends_zscore: float,
    sector_zscore: float = 0.0,
    horizon_days: int | None = None,
    n_paths: int | None = None,
    rng: np.random.Generator | None = None,
) -> PerturbedForecast:
    """Run the perturbation series and return a PerturbedForecast.

    Parameters
    ----------
    params:
        Calibrated GBMParams for the ticker.
    sentiment:
        VADER compound score in [-1, 1] from ingest.news.
    trends_zscore:
        Google Trends z-score from ingest.trends.
    sector_zscore:
        Sector/index momentum z-score from ingest.sector.
    horizon_days:
        Forecast horizon (trading days). Defaults to settings value.
    n_paths:
        MC paths. Defaults to settings value.
    rng:
        Optional seeded RNG for reproducibility.
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
    # ε² diagonals (saturation), ε·ε cross terms (reinforcement)
    delta_2 = (
        -(eps_n ** 2) * s ** 2
        - (eps_t ** 2) * t ** 2
        - (eps_s ** 2) * x ** 2
        + (eps_n * eps_t) * s * t
        + (eps_n * eps_s) * s * x
        + (eps_t * eps_s) * t * x
    )

    mu_perturbed = params.mu + delta_1 + delta_2

    # --- Confidence: series convergence ---
    abs_d1 = abs(delta_1)
    abs_d2 = abs(delta_2)
    if abs_d1 < 1e-9:
        confidence = 0.0
    else:
        confidence = float(np.clip(1.0 - abs_d2 / abs_d1, 0.0, 1.0))

    # --- MC simulation: probability + price range ---
    mc_log_returns = simulate_paths(
        params,
        horizon_days=horizon_days,
        n_paths=n_paths,
        mu_override=mu_perturbed,
        rng=rng,
    )
    prob = float(np.mean(mc_log_returns > 0))

    # Convert log-returns to % price changes for range
    mc_pct_changes = (np.exp(mc_log_returns) - 1) * 100
    range_floor = float(np.percentile(mc_pct_changes, 5))
    range_ceil = float(np.percentile(mc_pct_changes, 95))
    range_median = float(np.percentile(mc_pct_changes, 50))

    log.info(
        "%s  μ₀=%.4f Δ₁=%.4f Δ₂=%.4f μ*=%.4f  P(up)=%.3f  conf=%.2f  "
        "range=[%.1f%%, %.1f%%]  sector_z=%.2f",
        params.ticker, params.mu, delta_1, delta_2, mu_perturbed, prob, confidence,
        range_floor, range_ceil, sector_zscore,
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
        range_floor_pct=range_floor,
        range_ceil_pct=range_ceil,
        range_median_pct=range_median,
    )
