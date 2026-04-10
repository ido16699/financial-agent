"""Tests: cache TTL behavior."""

import time
import tempfile
from pathlib import Path

import pytest


class TestCacheTTL:
    def _make_cache_with_dir(self, tmp_path: Path):
        """Return cache module with its cache_dir pointing to tmp_path."""
        import stochsignal.ingest.cache as cache_mod
        import stochsignal.config as config_mod

        # Patch settings.cache_dir to use the temp directory
        original = config_mod.settings.cache_dir
        config_mod.settings.cache_dir = tmp_path
        yield cache_mod
        config_mod.settings.cache_dir = original

    def test_set_and_get(self, tmp_path):
        import stochsignal.config as config_mod
        import stochsignal.ingest.cache as cache_mod

        config_mod.settings.cache_dir = tmp_path
        cache_mod.set("test_key", {"value": 42}, ttl_seconds=60)
        result = cache_mod.get("test_key")
        assert result == {"value": 42}

    def test_expired_returns_none(self, tmp_path):
        import stochsignal.config as config_mod
        import stochsignal.ingest.cache as cache_mod

        config_mod.settings.cache_dir = tmp_path
        cache_mod.set("expiring_key", "hello", ttl_seconds=1)
        time.sleep(1.1)
        result = cache_mod.get("expiring_key")
        assert result is None

    def test_missing_key_returns_none(self, tmp_path):
        import stochsignal.config as config_mod
        import stochsignal.ingest.cache as cache_mod

        config_mod.settings.cache_dir = tmp_path
        result = cache_mod.get("does_not_exist")
        assert result is None

    def test_invalidate(self, tmp_path):
        import stochsignal.config as config_mod
        import stochsignal.ingest.cache as cache_mod

        config_mod.settings.cache_dir = tmp_path
        cache_mod.set("to_delete", 99, ttl_seconds=3600)
        cache_mod.invalidate("to_delete")
        assert cache_mod.get("to_delete") is None
