"""Fetch and cache news sentiment scores per ticker.

Primary source: NewsAPI.  Fallback: GDELT REST API (no key needed).
Sentiment: VADER compound score averaged over recent article titles/descriptions.

Point-in-time note: NewsAPI free tier only goes back ~30 days. For forward
runs this is fine. Backtest MUST NOT call this with as_of far in the past;
the backtest harness skips news for historical windows.
"""

from __future__ import annotations

import datetime
import statistics
from typing import Any

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from stochsignal.config import settings
from stochsignal.ingest import cache
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)

_vader = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------

def _fetch_newsapi(ticker: str, limit: int) -> list[str]:
    """Return list of text snippets (title + description) from NewsAPI."""
    key = settings.news_api_key
    if not key:
        return []

    url = "https://newsapi.org/v2/everything"
    params: dict[str, Any] = {
        "q": ticker,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": limit,
        "apiKey": key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        texts = []
        for a in articles:
            text = " ".join(filter(None, [a.get("title"), a.get("description")]))
            if text.strip():
                texts.append(text)
        return texts
    except Exception as exc:
        log.warning("NewsAPI failed for %s: %s", ticker, exc)
        return []


# ---------------------------------------------------------------------------
# GDELT fallback
# ---------------------------------------------------------------------------

def _fetch_gdelt(ticker: str, limit: int) -> list[str]:
    """Return list of article titles from GDELT ArticleSearch API."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    # GDELT query: company name derived from ticker (rough heuristic)
    query = ticker.split(".")[0]  # strip exchange suffix
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": limit,
        "format": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a.get("title", "") for a in articles if a.get("title")]
    except Exception as exc:
        log.warning("GDELT fallback failed for %s: %s", ticker, exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news_sentiment(ticker: str) -> float:
    """Return VADER compound sentiment score in [-1, 1] for `ticker`.

    Tries NewsAPI first, falls back to GDELT. Returns 0.0 if both fail
    or no articles are found.
    """
    cache_key = f"news_{ticker}_{datetime.date.today()}"
    cached = cache.get(cache_key)
    if cached is not None:
        log.debug("News cache hit: %s", ticker)
        return cached

    limit = settings.news_article_limit
    texts = _fetch_newsapi(ticker, limit)
    if not texts:
        log.info("Falling back to GDELT for %s", ticker)
        texts = _fetch_gdelt(ticker, limit)

    if not texts:
        log.warning("No news found for %s — sentiment=0.0", ticker)
        score = 0.0
    else:
        scores = [_vader.polarity_scores(t)["compound"] for t in texts]
        score = statistics.mean(scores)
        log.info("News sentiment %s: %.3f  (n=%d articles)", ticker, score, len(scores))

    cache.set(cache_key, score, ttl_seconds=settings.news_ttl)
    return score
