"""Load and expose project configuration from YAML files and environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent.parent
_SETTINGS_PATH = _ROOT / "config" / "settings.yaml"
_WATCHLIST_PATH = _ROOT / "config" / "watchlist.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


class Settings:
    """Thin wrapper around settings.yaml — access via dot notation."""

    def __init__(self) -> None:
        raw = _load_yaml(_SETTINGS_PATH)
        self._raw = raw

        m = raw["model"]
        self.calibration_window_days: int = m["calibration_window_days"]
        self.forecast_horizon_days: int = m["forecast_horizon_days"]
        self.mc_paths: int = m["mc_paths"]
        self.epsilon_news: float = m["epsilon_news"]
        self.epsilon_trends: float = m["epsilon_trends"]

        s = raw["signal"]
        self.min_confidence_to_report: float = s["min_confidence_to_report"]

        b = raw["backtest"]
        self.backtest_start: str = b["start_date"]
        self.backtest_end: str = b["end_date"]
        self.transaction_cost_bps: int = b["transaction_cost_bps"]

        d = raw["digest"]
        self.send_day: int = d["send_day"]
        self.send_hour_utc: int = d["send_hour_utc"]
        self.disclaimer: str = d["disclaimer"].strip()

        i = raw["ingest"]
        self.price_ttl: int = i["price_ttl_seconds"]
        self.news_ttl: int = i["news_ttl_seconds"]
        self.trends_ttl: int = i["trends_ttl_seconds"]
        self.news_article_limit: int = i["news_article_limit"]
        self.trends_lookback_days: int = i["trends_lookback_days"]
        self.cache_dir: Path = _ROOT / i["cache_dir"]

    # Secrets from environment
    @property
    def news_api_key(self) -> str:
        key = os.environ.get("NEWS_API_KEY", "")
        return key

    @property
    def smtp_host(self) -> str:
        return os.environ.get("SMTP_HOST", "smtp.gmail.com")

    @property
    def smtp_port(self) -> int:
        return int(os.environ.get("SMTP_PORT", "587"))

    @property
    def smtp_user(self) -> str:
        return os.environ.get("SMTP_USER", "")

    @property
    def smtp_password(self) -> str:
        return os.environ.get("SMTP_PASSWORD", "")

    @property
    def digest_recipients(self) -> list[str]:
        raw = os.environ.get("DIGEST_RECIPIENTS", "")
        return [r.strip() for r in raw.split(",") if r.strip()]


class Watchlist:
    """Flat list of tickers from watchlist.yaml, with group metadata."""

    def __init__(self) -> None:
        raw = _load_yaml(_WATCHLIST_PATH)
        self.us_large_cap: list[str] = raw.get("us_large_cap", [])
        self.nasdaq_tech: list[str] = raw.get("nasdaq_tech", [])
        self.tase: list[str] = raw.get("tase", [])

    @property
    def all_tickers(self) -> list[str]:
        return self.us_large_cap + self.nasdaq_tech + self.tase

    def group_of(self, ticker: str) -> str:
        if ticker in self.us_large_cap:
            return "US Large-Cap"
        if ticker in self.nasdaq_tech:
            return "Nasdaq Tech"
        if ticker in self.tase:
            return "TASE"
        return "Unknown"


# Module-level singletons
settings = Settings()
watchlist = Watchlist()
