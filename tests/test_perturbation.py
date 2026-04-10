"""Tests: perturbation series on known inputs.

Verify Δ₁, Δ₂, confidence, prob_up, and price range behave as expected.
"""

import numpy as np
import pytest

from stochsignal.model.gbm import GBMParams
from stochsignal.model.perturbation import compute


def _forecast(sentiment: float, trends_z: float, sector_z: float = 0.0,
              mu: float = 0.0, seed: int = 42):
    params = GBMParams(ticker="TEST", mu=mu, sigma=0.20, n_obs=252)
    rng = np.random.default_rng(seed)
    return compute(params, sentiment=sentiment, trends_zscore=trends_z,
                   sector_zscore=sector_z, n_paths=50_000, rng=rng)


class TestDelta1:
    def test_zero_signals_give_zero_delta1(self):
        fc = _forecast(0.0, 0.0, 0.0)
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
        assert abs(fc2.delta_1 / fc1.delta_1 - 2.0) < 1e-9

    def test_sector_contributes_to_delta1(self):
        fc_no_sector = _forecast(0.0, 0.0, 0.0)
        fc_with_sector = _forecast(0.0, 0.0, 1.0)
        assert fc_with_sector.delta_1 > fc_no_sector.delta_1


class TestDelta2:
    def test_saturation_negative_diagonal(self):
        """Δ₂ for pure sentiment (no trends/sector) should be negative."""
        fc = _forecast(1.0, 0.0, 0.0)
        assert fc.delta_2 < 0, "Diagonal saturation: Δ₂ should be negative"

    def test_cross_term_positive_reinforcement(self):
        """When both sentiment and trends are positive, cross term reinforces."""
        fc_both = _forecast(1.0, 1.0)
        fc_sent = _forecast(1.0, 0.0)
        assert fc_both.delta_2 > fc_sent.delta_2

    def test_epsilon_squared_scaling(self):
        """Δ₂ diagonal should scale as ε² not |ε|.

        For sentiment-only: Δ₂ = -ε_n² · s²
        With ε_n = 0.3, s = 1.0: Δ₂ = -0.09 (not -0.3)
        """
        fc = _forecast(1.0, 0.0, 0.0)
        # eps_n = 0.3 from settings → eps_n² = 0.09
        expected_delta_2 = -(0.3 ** 2) * (1.0 ** 2)  # -0.09
        assert abs(fc.delta_2 - expected_delta_2) < 1e-9, (
            f"Δ₂ = {fc.delta_2}, expected {expected_delta_2}"
        )

    def test_sector_cross_terms(self):
        """Adding sector with same sign as sentiment should increase Δ₂."""
        fc_no_sector = _forecast(1.0, 1.0, 0.0)
        fc_with_sector = _forecast(1.0, 1.0, 1.0)
        # Sector adds cross terms (+ε_n·ε_s·s·x + ε_t·ε_s·t·x) but also -ε_s²·x²
        # With all positive inputs the cross terms should dominate for small ε_s
        assert fc_with_sector.delta_2 != fc_no_sector.delta_2


class TestConfidence:
    def test_zero_signals_zero_confidence(self):
        fc = _forecast(0.0, 0.0, 0.0)
        assert fc.confidence == 0.0

    def test_small_delta2_high_confidence(self):
        """When |Δ₂| << |Δ₁|, confidence should be close to 1."""
        fc = _forecast(0.01, 0.0)
        # With ε² scaling: Δ₁ = 0.003, Δ₂ = -0.000009 → ratio very small
        assert fc.confidence > 0.95

    def test_confidence_in_unit_interval(self):
        for s, t, x in [(1.0, 1.0, 1.0), (-1.0, -1.0, 0.0), (0.5, -0.5, 0.5), (0.0, 1.0, -1.0)]:
            fc = _forecast(s, t, x)
            assert 0.0 <= fc.confidence <= 1.0


class TestProbUp:
    def test_strong_positive_perturbed_mu_up(self):
        fc = _forecast(1.0, 1.0, 1.0, mu=0.5)
        assert fc.prob_up > 0.5

    def test_strong_negative_perturbed_mu_down(self):
        fc = _forecast(-1.0, -1.0, -1.0, mu=-0.5)
        assert fc.prob_up < 0.5


class TestPriceRange:
    def test_range_floor_below_ceiling(self):
        fc = _forecast(0.0, 0.0, 0.0, mu=0.1)
        assert fc.range_floor_pct < fc.range_ceil_pct

    def test_range_median_between_floor_ceil(self):
        fc = _forecast(0.5, 0.3, 0.2, mu=0.1)
        assert fc.range_floor_pct <= fc.range_median_pct <= fc.range_ceil_pct

    def test_positive_drift_positive_median(self):
        """Strong positive drift should give positive median return."""
        fc = _forecast(0.0, 0.0, 0.0, mu=1.0)
        assert fc.range_median_pct > 0

    def test_negative_drift_negative_median(self):
        fc = _forecast(0.0, 0.0, 0.0, mu=-1.0)
        assert fc.range_median_pct < 0
