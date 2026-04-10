"""Tests: perturbation series on known inputs.

Verify Δ₁, Δ₂, confidence, and prob_up behave as expected.
"""

import numpy as np
import pytest

from stochsignal.model.gbm import GBMParams
from stochsignal.model.perturbation import compute


_BASE_PARAMS = GBMParams(ticker="TEST", mu=0.0, sigma=0.20, n_obs=252)


def _forecast(sentiment: float, trends_z: float, mu: float = 0.0, seed: int = 42):
    params = GBMParams(ticker="TEST", mu=mu, sigma=0.20, n_obs=252)
    rng = np.random.default_rng(seed)
    return compute(params, sentiment=sentiment, trends_zscore=trends_z, n_paths=50_000, rng=rng)


class TestDelta1:
    def test_zero_signals_give_zero_delta1(self):
        fc = _forecast(0.0, 0.0)
        assert abs(fc.delta_1) < 1e-12

    def test_positive_sentiment_positive_delta1(self):
        fc = _forecast(1.0, 0.0)
        assert fc.delta_1 > 0

    def test_negative_sentiment_negative_delta1(self):
        fc = _forecast(-1.0, 0.0)
        assert fc.delta_1 < 0

    def test_linearity_in_sentiment(self):
        fc1 = _forecast(0.5, 0.0)
        fc2 = _forecast(1.0, 0.0)
        # delta_1 should scale linearly with sentiment
        assert abs(fc2.delta_1 / fc1.delta_1 - 2.0) < 1e-9


class TestDelta2:
    def test_saturation_negative_diagonal(self):
        """Δ₂ for pure sentiment (no trends) should be negative."""
        fc = _forecast(1.0, 0.0)
        assert fc.delta_2 < 0, "Diagonal saturation: Δ₂ should be negative"

    def test_cross_term_positive_reinforcement(self):
        """When both sentiment and trends are positive, cross term reinforces."""
        fc_both = _forecast(1.0, 1.0)
        fc_sent = _forecast(1.0, 0.0)
        # Including trends (positive) should increase delta_2 relative to sentiment-only
        assert fc_both.delta_2 > fc_sent.delta_2


class TestConfidence:
    def test_zero_signals_zero_confidence(self):
        """Zero signals → Δ₁ ≈ 0 → confidence = 0."""
        fc = _forecast(0.0, 0.0)
        assert fc.confidence == 0.0

    def test_small_delta2_high_confidence(self):
        """When |Δ₂| << |Δ₁|, confidence should be close to 1."""
        # Use very small epsilon values: achieved by small sentiment
        fc = _forecast(0.01, 0.0)  # Δ₁ small, Δ₂ much smaller (quadratic)
        # With sentiment=0.01: Δ₁ ~ 0.003, Δ₂ ~ -0.000009 → ratio very small
        assert fc.confidence > 0.9

    def test_confidence_in_unit_interval(self):
        for s, t in [(1.0, 1.0), (-1.0, -1.0), (0.5, -0.5), (0.0, 1.0)]:
            fc = _forecast(s, t)
            assert 0.0 <= fc.confidence <= 1.0


class TestProbUp:
    def test_strong_positive_perturbed_mu_up(self):
        """Very positive perturbed drift → P(up) > 0.5."""
        fc = _forecast(1.0, 1.0, mu=0.5)
        assert fc.prob_up > 0.5

    def test_strong_negative_perturbed_mu_down(self):
        """Very negative perturbed drift → P(up) < 0.5."""
        fc = _forecast(-1.0, -1.0, mu=-0.5)
        assert fc.prob_up < 0.5
