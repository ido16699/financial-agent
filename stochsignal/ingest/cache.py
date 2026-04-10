"""Simple file-based TTL cache using pickle.

Each cached item is stored as <cache_dir>/<key>.pkl alongside a
<cache_dir>/<key>.meta file that records the expiry timestamp.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

from stochsignal.config import settings
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)


def _meta_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.meta"


def _data_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.pkl"


def _safe_key(raw: str) -> str:
    """Sanitize a raw key string to a safe filename."""
    return raw.replace("/", "_").replace(".", "_").replace(" ", "_")


def get(key: str) -> Any | None:
    """Return cached value if it exists and has not expired, else None."""
    cache_dir = settings.cache_dir
    key = _safe_key(key)
    meta = _meta_path(cache_dir, key)
    data = _data_path(cache_dir, key)

    if not meta.exists() or not data.exists():
        return None

    expiry = float(meta.read_text().strip())
    if time.time() > expiry:
        log.debug("Cache expired for key=%s", key)
        return None

    with open(data, "rb") as f:
        return pickle.load(f)


def set(key: str, value: Any, ttl_seconds: int) -> None:
    """Persist value under key with a TTL."""
    cache_dir = settings.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _safe_key(key)

    expiry = time.time() + ttl_seconds
    _meta_path(cache_dir, key).write_text(str(expiry))
    with open(_data_path(cache_dir, key), "wb") as f:
        pickle.dump(value, f)
    log.debug("Cached key=%s ttl=%ds", key, ttl_seconds)


def invalidate(key: str) -> None:
    """Remove a cached entry."""
    cache_dir = settings.cache_dir
    key = _safe_key(key)
    for path in [_meta_path(cache_dir, key), _data_path(cache_dir, key)]:
        if path.exists():
            path.unlink()
