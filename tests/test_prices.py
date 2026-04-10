"""Tests: point-in-time slice never leaks future data."""

import pandas as pd
import numpy as np
import pytest

from stochsignal.ingest.prices import get_price_history, get_log_returns


def make_fake_prices(start: str, end: str) -> pd.DataFrame:
    """Build a synthetic price DataFrame mimicking yfinance output."""
    idx = pd.date_range(start=start, end=end, freq="B")
    np.random.seed(0)
    closes = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, len(idx)))
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1_000_000},
        index=idx,
    )


class TestPointInTime:
    def test_no_future_leak(self, monkeypatch):
        """Prices returned must have index <= as_of."""
        import stochsignal.ingest.prices as price_mod
        import stochsignal.ingest.cache as cache_mod

        fake = make_fake_prices("2020-01-01", "2025-12-31")
        monkeypatch.setattr(cache_mod, "get", lambda key: None)
        monkeypatch.setattr(cache_mod, "set", lambda key, val, ttl_seconds: None)

        import yfinance as yf
        monkeypatch.setattr(
            yf, "download",
            lambda *args, **kwargs: fake,
        )

        as_of = pd.Timestamp("2023-06-15")
        result = price_mod.get_price_history("FAKE", as_of=as_of, window_days=252)

        assert (result.index <= as_of).all(), "Future prices leaked past as_of boundary"

    def test_window_respected(self, monkeypatch):
        """Result should have at most `window_days` rows."""
        import stochsignal.ingest.prices as price_mod
        import stochsignal.ingest.cache as cache_mod
        import yfinance as yf

        fake = make_fake_prices("2018-01-01", "2025-12-31")
        monkeypatch.setattr(cache_mod, "get", lambda key: None)
        monkeypatch.setattr(cache_mod, "set", lambda key, val, ttl_seconds: None)
        monkeypatch.setattr(yf, "download", lambda *a, **kw: fake)

        result = price_mod.get_price_history("FAKE", as_of=pd.Timestamp("2025-01-01"), window_days=100)
        assert len(result) <= 100

    def test_log_returns_no_future_leak(self, monkeypatch):
        """Log-returns must not contain dates after as_of."""
        import stochsignal.ingest.prices as price_mod
        import stochsignal.ingest.cache as cache_mod
        import yfinance as yf

        fake = make_fake_prices("2020-01-01", "2025-12-31")
        monkeypatch.setattr(cache_mod, "get", lambda key: None)
        monkeypatch.setattr(cache_mod, "set", lambda key, val, ttl_seconds: None)
        monkeypatch.setattr(yf, "download", lambda *a, **kw: fake)

        as_of = pd.Timestamp("2022-12-31")
        lr = price_mod.get_log_returns("FAKE", as_of=as_of)
        assert (lr.index <= as_of).all()
