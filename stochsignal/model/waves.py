"""Wave and pattern analysis using Fourier decomposition.

Extracts periodic structure from price returns that GBM ignores:

1. FFT Spectral Analysis
   - Decompose return series into frequency components
   - Identify dominant cycles (e.g., monthly, quarterly)
   - Extrapolate the dominant waves forward to generate a wave forecast

2. Mean-Reversion Detection (Ornstein-Uhlenbeck)
   - Fit OU process: dx = θ(μ - x)dt + σ dW
   - If the stock is far from its mean and θ is strong → expect reversion
   - Half-life = ln(2)/θ tells you how fast it reverts

3. Momentum Score
   - Short-term (5d) vs medium-term (21d) vs long-term (63d) returns
   - Weighted combination detects trend vs mean-reversion

The wave_signal is a composite z-score from all three,
fed into the perturbation series as an additional first-order term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class WaveAnalysis:
    """Output of the wave/pattern detector."""
    # FFT
    dominant_period_days: float    # strongest cycle length in trading days
    fft_forecast_signal: float    # extrapolated wave value at horizon (normalised)
    spectral_strength: float      # how strong the dominant cycle is (0-1)

    # Mean reversion (OU)
    ou_theta: float               # mean-reversion speed
    ou_mu: float                  # long-run mean (log-price level)
    ou_half_life_days: float      # half-life in trading days
    reversion_signal: float       # z-score: how far from mean (negative = expect up)

    # Momentum
    momentum_5d: float
    momentum_21d: float
    momentum_63d: float
    momentum_composite: float     # weighted blend

    # Combined wave signal (used in perturbation Δ₁)
    wave_signal: float


def _fft_analysis(
    log_returns: np.ndarray,
    horizon_days: int = 5,
) -> tuple[float, float, float]:
    """Extract dominant cycle and forecast from FFT.

    Returns (dominant_period, forecast_signal, spectral_strength).
    """
    n = len(log_returns)
    if n < 30:
        return 0.0, 0.0, 0.0

    # Detrend
    detrended = log_returns - np.mean(log_returns)

    # FFT
    fft_vals = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(n)
    magnitudes = np.abs(fft_vals)

    # Skip DC component (index 0) and very low frequencies
    if len(magnitudes) < 3:
        return 0.0, 0.0, 0.0

    magnitudes[0] = 0
    if len(magnitudes) > 1:
        magnitudes[1] = 0  # skip near-DC

    # Find dominant frequency
    dominant_idx = np.argmax(magnitudes)
    if dominant_idx == 0 or freqs[dominant_idx] < 1e-10:
        return 0.0, 0.0, 0.0

    dominant_period = 1.0 / freqs[dominant_idx]
    dominant_magnitude = magnitudes[dominant_idx]
    total_energy = np.sum(magnitudes ** 2)
    spectral_strength = float(dominant_magnitude ** 2 / total_energy) if total_energy > 0 else 0.0

    # Forecast: extrapolate dominant wave forward
    phase = np.angle(fft_vals[dominant_idx])
    # Value at t = n + horizon_days
    forecast = float(dominant_magnitude * np.cos(
        2 * np.pi * freqs[dominant_idx] * (n + horizon_days) + phase
    ))
    # Normalise by std of returns
    std_r = np.std(log_returns)
    forecast_signal = forecast / std_r if std_r > 1e-9 else 0.0

    return float(dominant_period), float(np.clip(forecast_signal, -3, 3)), float(spectral_strength)


def _ou_analysis(closes: np.ndarray) -> tuple[float, float, float, float]:
    """Fit Ornstein-Uhlenbeck process to log-prices.

    OU: dx = θ(μ - x)dt + σ dW
    Discretised: x(t+1) - x(t) = θ(μ - x(t))·Δt + noise

    Returns (theta, mu, half_life_days, reversion_signal).
    """
    log_prices = np.log(closes)
    n = len(log_prices)
    if n < 30:
        return 0.0, 0.0, float("inf"), 0.0

    x = log_prices[:-1]
    dx = np.diff(log_prices)

    # OLS: dx = a + b·x → θ = -b, μ = -a/b
    A = np.vstack([np.ones(len(x)), x]).T
    result = np.linalg.lstsq(A, dx, rcond=None)
    a, b = result[0]

    theta = -b  # mean-reversion speed (per day)

    if theta <= 0:
        # Not mean-reverting — trending
        return float(theta), 0.0, float("inf"), 0.0

    mu = -a / b  # long-run mean log-price
    half_life = np.log(2) / theta  # in trading days

    # Reversion signal: how far is current price from OU mean?
    # Negative = below mean (expect reversion UP)
    residuals = dx - (a + b * x)
    ou_sigma = np.std(residuals) if len(residuals) > 1 else 1.0
    current_deviation = log_prices[-1] - mu
    # Normalise by the OU stationary std = σ / sqrt(2θ)
    stationary_std = ou_sigma / np.sqrt(2 * theta) if theta > 0 else 1.0
    reversion_signal = -current_deviation / stationary_std if stationary_std > 1e-9 else 0.0
    # Negative deviation → positive signal (expect up)
    reversion_signal = float(np.clip(reversion_signal, -3, 3))

    return float(theta), float(mu), float(half_life), reversion_signal


def _momentum_analysis(closes: np.ndarray) -> tuple[float, float, float, float]:
    """Compute multi-timeframe momentum scores.

    Returns (mom_5d, mom_21d, mom_63d, composite).
    """
    n = len(closes)
    if n < 65:
        return 0.0, 0.0, 0.0, 0.0

    # Momentum = current price / price N days ago - 1
    mom_5d = closes[-1] / closes[-6] - 1 if n >= 6 else 0.0
    mom_21d = closes[-1] / closes[-22] - 1 if n >= 22 else 0.0
    mom_63d = closes[-1] / closes[-64] - 1 if n >= 64 else 0.0

    # Normalise by recent volatility (20-day)
    recent_returns = np.diff(np.log(closes[-22:]))
    vol = np.std(recent_returns) * np.sqrt(252) if len(recent_returns) > 5 else 0.2

    mom_5d_z = (mom_5d / vol) if vol > 1e-9 else 0.0
    mom_21d_z = (mom_21d / vol) if vol > 1e-9 else 0.0
    mom_63d_z = (mom_63d / vol) if vol > 1e-9 else 0.0

    # Composite: short-term gets less weight (noise), medium-term gets most
    composite = 0.2 * mom_5d_z + 0.5 * mom_21d_z + 0.3 * mom_63d_z
    composite = float(np.clip(composite, -3, 3))

    return float(mom_5d_z), float(mom_21d_z), float(mom_63d_z), composite


def analyse_waves(
    log_returns: np.ndarray,
    closes: np.ndarray,
    horizon_days: int = 5,
) -> WaveAnalysis:
    """Run full wave/pattern analysis.

    Parameters
    ----------
    log_returns : daily log-returns array
    closes : daily close prices array (same period)
    horizon_days : forecast horizon in trading days

    Returns
    -------
    WaveAnalysis with all component signals and combined wave_signal.
    """
    # FFT
    dominant_period, fft_signal, spectral_strength = _fft_analysis(log_returns, horizon_days)

    # OU mean reversion
    ou_theta, ou_mu, half_life, reversion_signal = _ou_analysis(closes)

    # Momentum
    mom_5d, mom_21d, mom_63d, mom_composite = _momentum_analysis(closes)

    # Combined wave signal: weighted blend
    # FFT only contributes if spectral strength is meaningful
    fft_weight = min(spectral_strength * 5, 1.0)  # scale up small strengths
    ou_weight = 1.0 if ou_theta > 0 else 0.0       # only if mean-reverting

    wave_signal = (
        0.3 * fft_signal * fft_weight
        + 0.4 * reversion_signal * ou_weight
        + 0.3 * mom_composite
    )
    wave_signal = float(np.clip(wave_signal, -3, 3))

    log.info(
        "Waves: period=%.0fd fft=%.2f(str=%.2f) OU θ=%.3f hl=%.0fd rev=%.2f "
        "mom=%.2f → wave=%.2f",
        dominant_period, fft_signal, spectral_strength,
        ou_theta, half_life, reversion_signal,
        mom_composite, wave_signal,
    )

    return WaveAnalysis(
        dominant_period_days=dominant_period,
        fft_forecast_signal=fft_signal,
        spectral_strength=spectral_strength,
        ou_theta=ou_theta,
        ou_mu=ou_mu,
        ou_half_life_days=half_life,
        reversion_signal=reversion_signal,
        momentum_5d=mom_5d,
        momentum_21d=mom_21d,
        momentum_63d=mom_63d,
        momentum_composite=mom_composite,
        wave_signal=wave_signal,
    )
