"""Chaos theory tools: Lyapunov exponents and regime detection.

Maximal Lyapunov Exponent (MLE)
-------------------------------
Measures how fast nearby trajectories diverge in phase space.

  λ₁ > 0  → chaotic, hard to predict beyond horizon ~1/λ₁
  λ₁ ≈ 0  → edge of chaos
  λ₁ < 0  → stable/mean-reverting, predictable

Implementation: Rosenstein et al. (1993) algorithm.
  1. Embed the return series in d-dimensional phase space using delay τ
  2. For each point, find nearest neighbour (excluding temporal neighbours)
  3. Track divergence of nearest-neighbour pairs over time
  4. MLE = slope of mean log-divergence vs time

Tradeable signal:
  - "predictable" regime (λ₁ low/negative) → model forecasts are trustworthy
  - "chaotic" regime (λ₁ high positive) → reduce position size or sit out

Takens Embedding
----------------
Optimal embedding dimension d and delay τ are estimated from:
  - τ: first minimum of the auto-mutual information
  - d: false nearest neighbours (FNN) method (simplified)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class RegimeInfo:
    """Output of the regime detector."""
    lyapunov_exponent: float   # maximal Lyapunov exponent
    regime: str                # 'predictable', 'transitional', 'chaotic'
    predictability_horizon: float  # estimated horizon in trading days (1/λ₁)
    confidence_multiplier: float   # scale factor for position sizing [0.0, 1.5]
    embedding_dim: int
    delay: int


def _auto_mutual_info(x: np.ndarray, max_lag: int = 50) -> int:
    """Estimate optimal delay τ as first minimum of auto-mutual information.

    Simplified version using binned MI estimation.
    """
    n = len(x)
    n_bins = max(10, int(np.sqrt(n / 5)))
    best_tau = 1
    prev_mi = float("inf")

    for tau in range(1, min(max_lag, n // 4)):
        x1 = x[:-tau]
        x2 = x[tau:]
        # Binned MI estimation
        hist_2d, _, _ = np.histogram2d(x1, x2, bins=n_bins)
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        # MI = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

        if mi > prev_mi:
            best_tau = tau - 1
            break
        prev_mi = mi

    return max(1, best_tau)


def _estimate_embedding_dim(x: np.ndarray, tau: int, max_dim: int = 10) -> int:
    """Estimate embedding dimension using false nearest neighbours (simplified).

    Increases d until the fraction of false neighbours drops below 5%.
    """
    n = len(x)

    for d in range(2, max_dim + 1):
        m = n - (d - 1) * tau
        if m < 50:
            return d - 1

        # Build embedded vectors
        embedded = np.array([x[i:i + d * tau:tau] for i in range(m)])
        embedded_lower = embedded[:, :-1]  # d-1 dimensional

        # Find nearest neighbours in d-1 dimensions
        tree = KDTree(embedded_lower)
        dists, indices = tree.query(embedded_lower, k=2)
        nn_dists = dists[:, 1]  # nearest neighbour distance (skip self)
        nn_indices = indices[:, 1]

        # Check if they're still neighbours in d dimensions
        false_nn = 0
        total = 0
        for i in range(m):
            j = nn_indices[i]
            if j >= m:
                continue
            d_lower = nn_dists[i]
            if d_lower < 1e-10:
                continue
            d_higher = np.linalg.norm(embedded[i] - embedded[j])
            ratio = abs(d_higher - d_lower) / d_lower
            total += 1
            if ratio > 2.0:
                false_nn += 1

        if total > 0 and false_nn / total < 0.05:
            return d

    return max_dim


def _lyapunov_rosenstein(
    x: np.ndarray,
    d: int,
    tau: int,
    max_iter: int = 20,
    min_temporal_sep: int = 10,
) -> float:
    """Compute maximal Lyapunov exponent using Rosenstein et al. algorithm.

    Parameters
    ----------
    x : 1-D array of log-returns
    d : embedding dimension
    tau : delay
    max_iter : how many steps to track divergence
    min_temporal_sep : minimum index separation for nearest neighbours

    Returns
    -------
    Estimated λ₁ (per trading day). Annualise by multiplying by 252.
    """
    n = len(x)
    m = n - (d - 1) * tau
    if m < 50:
        return 0.0

    # Embed
    embedded = np.array([x[i:i + d * tau:tau] for i in range(m)])

    # Find nearest neighbours with temporal separation
    tree = KDTree(embedded)
    divergence = np.zeros((m, max_iter))

    for i in range(m):
        # Query for several neighbours, pick the closest with temporal separation
        k = min(20, m)
        dists, indices = tree.query(embedded[i], k=k)
        nn_idx = -1
        for idx_j in range(1, k):
            j = indices[idx_j]
            if abs(i - j) > min_temporal_sep:
                nn_idx = j
                break
        if nn_idx < 0:
            continue

        # Track divergence over time
        for step in range(max_iter):
            i2 = i + step
            j2 = nn_idx + step
            if i2 >= m or j2 >= m:
                break
            dist = np.linalg.norm(embedded[i2] - embedded[j2])
            divergence[i, step] = np.log(max(dist, 1e-15))

    # Average log-divergence at each step
    mean_div = []
    for step in range(max_iter):
        col = divergence[:, step]
        valid = col[col != 0]
        if len(valid) > 10:
            mean_div.append(np.mean(valid))

    if len(mean_div) < 5:
        return 0.0

    # λ₁ = slope of mean log-divergence vs time
    steps_arr = np.arange(len(mean_div), dtype=float)
    # Linear regression
    A = np.vstack([steps_arr, np.ones(len(steps_arr))]).T
    slope, _ = np.linalg.lstsq(A, np.array(mean_div), rcond=None)[0]

    return float(slope)  # per day


def detect_regime(
    log_returns: np.ndarray,
    window: int | None = None,
) -> RegimeInfo:
    """Detect the current market regime for a return series.

    Parameters
    ----------
    log_returns : 1-D array of daily log-returns
    window : use only the last N returns. Default: 252.

    Returns
    -------
    RegimeInfo with classification and confidence multiplier.
    """
    if window is not None:
        x = log_returns[-window:]
    else:
        x = log_returns[-252:]

    if len(x) < 60:
        log.warning("Too few returns for regime detection: %d", len(x))
        return RegimeInfo(
            lyapunov_exponent=0.0, regime="unknown",
            predictability_horizon=float("inf"),
            confidence_multiplier=0.5,
            embedding_dim=3, delay=1,
        )

    # Estimate embedding parameters
    tau = _auto_mutual_info(x)
    d = _estimate_embedding_dim(x, tau)

    # Compute Lyapunov exponent
    lam = _lyapunov_rosenstein(x, d, tau)

    # Annualise
    lam_annual = lam * 252

    # Classify regime — thresholds calibrated for financial time series
    # Financial data is inherently noisy; λ₁ ≈ 0.02-0.04/day is normal.
    # Only extreme values indicate truly unpredictable regimes.
    if lam < 0.02:
        regime = "predictable"
        conf_mult = 1.3
    elif lam < 0.05:
        regime = "transitional"
        conf_mult = 1.0
    else:
        regime = "chaotic"
        conf_mult = 0.6

    # Predictability horizon (in trading days)
    pred_horizon = 1.0 / max(lam, 0.001)

    log.info(
        "Regime: %s  λ₁=%.4f/day (%.2f/yr)  horizon=%.0f days  conf_mult=%.1f  d=%d τ=%d",
        regime, lam, lam_annual, pred_horizon, conf_mult, d, tau,
    )

    return RegimeInfo(
        lyapunov_exponent=lam,
        regime=regime,
        predictability_horizon=pred_horizon,
        confidence_multiplier=conf_mult,
        embedding_dim=d,
        delay=tau,
    )
