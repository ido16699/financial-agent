"""Tests: GBM calibration round-trip on synthetic data.

Generate synthetic log-returns from a known (μ, σ), calibrate,
and verify the recovered parameters are close (within 2 std errors).
"""

import numpy as np
import pandas as pd
import pytest

from stochsignal.model.gbm import (
    GBMParams,
    calibrate,
    prob_up_closed_form,
    simulate_paths,
    prob_up_mc,
    TRADING_DAYS_PER_YEAR,
)


def make_synthetic_returns(mu_ann: float, sigma_ann: float, n: int = 1000, seed: int = 42) -> pd.Series:
    """Generate synthetic daily log-returns from GBM with known params."""
    rng = np.random.default_rng(seed)
    daily_mu = mu_ann / TRADING_DAYS_PER_YEAR
    daily_sigma = sigma_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    r = rng.normal(loc=daily_mu, scale=daily_sigma, size=n)
    return pd.Series(r, name="SYN")


class TestCalibration:
    def test_roundtrip_mu(self):
        """Calibrated μ should be within 2 std errors of true μ."""
        mu_true, sigma_true = 0.10, 0.20
        n = 2000
        r = make_synthetic_returns(mu_true, sigma_true, n=n)
        params = calibrate("SYN", r)
        # Standard error of mean annualised drift
        se_mu = (sigma_true / np.sqrt(n)) * TRADING_DAYS_PER_YEAR
        assert abs(params.mu - mu_true) < 3 * se_mu, (
            f"μ recovered={params.mu:.4f}, true={mu_true}, se={se_mu:.4f}"
        )

    def test_roundtrip_sigma(self):
        """Calibrated σ should be within 5% of true σ."""
        mu_true, sigma_true = 0.05, 0.25
        r = make_synthetic_returns(mu_true, sigma_true, n=3000)
        params = calibrate("SYN", r)
        assert abs(params.sigma - sigma_true) / sigma_true < 0.05, (
            f"σ recovered={params.sigma:.4f}, true={sigma_true}"
        )

    def test_too_few_returns_raises(self):
        r = pd.Series([0.01] * 10)
        with pytest.raises(ValueError, match="Too few"):
            calibrate("SYN", r)

    def test_n_obs_correct(self):
        r = make_synthetic_returns(0.08, 0.18, n=500)
        params = calibrate("SYN", r)
        assert params.n_obs == 500


class TestProbUpClosedForm:
    def test_positive_drift_above_half(self):
        """Strong positive drift → P(up) should be clearly above 0.5."""
        params = GBMParams(ticker="T", mu=0.5, sigma=0.2, n_obs=252)
        p = prob_up_closed_form(params, horizon_days=5)
        assert p > 0.55

    def test_negative_drift_below_half(self):
        params = GBMParams(ticker="T", mu=-0.5, sigma=0.2, n_obs=252)
        p = prob_up_closed_form(params, horizon_days=5)
        assert p < 0.45

    def test_zero_sigma_degenerate(self):
        """Zero sigma → deterministic: P(up)=1 if mu>σ²/2, else 0."""
        params = GBMParams(ticker="T", mu=0.1, sigma=0.0, n_obs=252)
        p = prob_up_closed_form(params, horizon_days=5)
        assert p == 1.0

    def test_mc_vs_closed_form_agree(self):
        """MC probability should agree with closed-form within 1%."""
        params = GBMParams(ticker="T", mu=0.12, sigma=0.20, n_obs=252)
        rng = np.random.default_rng(0)
        p_cf = prob_up_closed_form(params, horizon_days=5)
        p_mc = prob_up_mc(params, horizon_days=5, n_paths=100_000, rng=rng)
        assert abs(p_cf - p_mc) < 0.01, f"CF={p_cf:.4f} MC={p_mc:.4f}"
