"""Fetch and cache news sentiment scores per ticker using FinBERT.

Primary source: NewsAPI.  Fallback: GDELT REST API (no key needed).
Sentiment: FinBERT financial sentiment model (ProsusAI/finbert).

FinBERT returns {positive, negative, neutral} with confidence scores.
We convert to a single score in [-1, 1]:
  score = P(positive) - P(negative)
Then average across all articles, weighted by recency.
"""

from __future__ import annotations

import datetime
import math
from typing import Any

import requests

from stochsignal.config import settings
from stochsignal.ingest import cache
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)

# Lazy-load FinBERT to avoid slow import at startup
_finbert_pipeline = None


def _get_finbert():
    """Lazy-load the FinBERT pipeline (first call downloads ~400MB)."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        log.info("Loading FinBERT model (first time may download ~400MB)...")
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            top_k=None,  # return all labels with scores
            truncation=True,
            max_length=512,
        )
        log.info("FinBERT loaded.")
    return _finbert_pipeline


def _finbert_score(text: str) -> float:
    """Run FinBERT on a single text, return score in [-1, 1].

    score = P(positive) - P(negative)
    """
    pipe = _get_finbert()
    results = pipe(text[:512])[0]  # list of {label, score} dicts
    scores = {r["label"]: r["score"] for r in results}
    return scores.get("positive", 0.0) - scores.get("negative", 0.0)


def _finbert_batch_scores(texts: list[str]) -> list[float]:
    """Run FinBERT on a batch of texts, return scores in [-1, 1]."""
    if not texts:
        return []
    pipe = _get_finbert()
    truncated = [t[:512] for t in texts]
    all_results = pipe(truncated, batch_size=8)
    scores = []
    for result in all_results:
        label_scores = {r["label"]: r["score"] for r in result}
        scores.append(label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0))
    return scores


# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------

def _fetch_newsapi(ticker: str, limit: int) -> list[dict[str, Any]]:
    """Return list of articles with text and published date from NewsAPI."""
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
        result = []
        for a in articles:
            text = " ".join(filter(None, [a.get("title"), a.get("description")]))
            if text.strip():
                result.append({
                    "text": text,
                    "published": a.get("publishedAt", ""),
                })
        return result
    except Exception as exc:
        log.warning("NewsAPI failed for %s: %s", ticker, exc)
        return []


# ---------------------------------------------------------------------------
# GDELT fallback
# ---------------------------------------------------------------------------

def _fetch_gdelt(ticker: str, limit: int) -> list[dict[str, Any]]:
    """Return list of articles from GDELT ArticleSearch API."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    query = ticker.split(".")[0]
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
        return [
            {"text": a.get("title", ""), "published": a.get("seendate", "")}
            for a in articles if a.get("title")
        ]
    except Exception as exc:
        log.warning("GDELT fallback failed for %s: %s", ticker, exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news_sentiment(ticker: str) -> float:
    """Return FinBERT sentiment score in [-1, 1] for `ticker`.

    Scores are recency-weighted: more recent articles count more.
    Returns 0.0 if no articles are found.
    """
    cache_key = f"news_finbert_{ticker}_{datetime.date.today()}"
    cached = cache.get(cache_key)
    if cached is not None:
        log.debug("News cache hit: %s", ticker)
        return cached

    limit = settings.news_article_limit
    articles = _fetch_newsapi(ticker, limit)
    if not articles:
        log.info("Falling back to GDELT for %s", ticker)
        articles = _fetch_gdelt(ticker, limit)

    if not articles:
        log.warning("No news found for %s — sentiment=0.0", ticker)
        score = 0.0
    else:
        texts = [a["text"] for a in articles]
        raw_scores = _finbert_batch_scores(texts)

        # Recency weighting: exponential decay, most recent article = weight 1.0
        # Oldest article = weight ~0.3
        n = len(raw_scores)
        decay_rate = 1.2  # higher = more weight on recent
        weights = [math.exp(-decay_rate * i / max(n - 1, 1)) for i in range(n)]
        total_weight = sum(weights)

        score = sum(s * w for s, w in zip(raw_scores, weights)) / total_weight
        log.info(
            "FinBERT sentiment %s: %.3f  (n=%d articles, top=%.3f)",
            ticker, score, n, raw_scores[0] if raw_scores else 0.0,
        )

    cache.set(cache_key, score, ttl_seconds=settings.news_ttl)
    return score
