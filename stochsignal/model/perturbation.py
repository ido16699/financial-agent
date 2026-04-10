"""Perturbation series on the GBM drift.

Conceptual model
----------------
μ(ε) = μ₀ + ε·μ₁ + ε²·μ₂

where ε is a small expansion parameter (implicitly 1 — the ε factors are
absorbed into the coupling constants epsilon_news and epsilon_trends from
settings).

Order-0 (μ₀): raw GBM drift from calibration.

Order-1 (μ₁): linear response to external signals.
  Δ₁ = epsilon_news · sentiment + epsilon_trends · trends_zscore

Order-2 (μ₂): quadratic correction (saturation + cross-signal interaction).
  Sign conventions:
    - Diagonal terms are negative (saturation: extreme signals are less reliable)
    - Cross term is positive (sentiment and trends reinforce each other)
  Δ₂ = −|epsilon_news|   · sentiment²
       −|epsilon_trends| · trends_zscore²
       +|epsilon_news · epsilon_trends| · sentiment · trends_zscore

Confidence (series convergence proxy):
  confidence = clip(1 − |Δ₂| / |Δ₁|, 0, 1)
  When |Δ₁| ≈ 0, the first-order signal is negligible → confidence = 0.

Perturbed drift:
  mu_perturbed = mu_0 + Δ₁ + Δ₂

The perturbed drift is passed to gbm.prob_up_mc() via mu_override.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stochsignal.config import settings
from stochsignal.model.gbm import GBMParams, prob_up_mc
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


def compute(
    params: GBMParams,
    sentiment: float,
    trends_zscore: float,
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

    # --- First-order perturbation ---
    delta_1 = eps_n * sentiment + eps_t * trends_zscore

    # --- Second-order perturbation ---
    # Saturation on diagonals (negative), reinforcement on cross (positive)
    delta_2 = (
        -abs(eps_n) * sentiment ** 2
        - abs(eps_t) * trends_zscore ** 2
        + abs(eps_n * eps_t) * sentiment * trends_zscore
    )

    mu_perturbed = params.mu + delta_1 + delta_2

    # --- Confidence: series convergence ---
    abs_d1 = abs(delta_1)
    abs_d2 = abs(delta_2)
    if abs_d1 < 1e-9:
        # First-order signal is negligible — series may not be converging
        confidence = 0.0
    else:
        confidence = float(np.clip(1.0 - abs_d2 / abs_d1, 0.0, 1.0))

    # --- MC probability ---
    prob = prob_up_mc(
        params,
        horizon_days=horizon_days,
        n_paths=n_paths,
        mu_override=mu_perturbed,
        rng=rng,
    )

    log.info(
        "%s  μ₀=%.4f Δ₁=%.4f Δ₂=%.4f μ*=%.4f  P(up)=%.3f  conf=%.2f",
        params.ticker, params.mu, delta_1, delta_2, mu_perturbed, prob, confidence,
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
    )
