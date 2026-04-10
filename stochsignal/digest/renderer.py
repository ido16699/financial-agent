"""Render the weekly digest HTML and text from Jinja2 templates."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from stochsignal.config import settings, watchlist
from stochsignal.model.perturbation import PerturbedForecast

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)


@dataclass
class SignalRow:
    """One row in the digest table."""
    ticker: str
    group: str
    direction: str        # "UP" or "DOWN"
    prob_up: float
    confidence: float
    sentiment: float
    trends_zscore: float
    mu_perturbed: float


def _to_signal_row(forecast: PerturbedForecast) -> SignalRow:
    return SignalRow(
        ticker=forecast.ticker,
        group=watchlist.group_of(forecast.ticker),
        direction="UP" if forecast.prob_up >= 0.5 else "DOWN",
        prob_up=forecast.prob_up,
        confidence=forecast.confidence,
        sentiment=forecast.sentiment,
        trends_zscore=forecast.trends_zscore,
        mu_perturbed=forecast.mu_perturbed,
    )


def render(
    forecasts: list[PerturbedForecast],
    run_date: datetime.date | None = None,
) -> tuple[str, str, str]:
    """Render digest from a list of PerturbedForecast objects.

    Parameters
    ----------
    forecasts:
        All forecasts (filtered and unfiltered).
    run_date:
        Date to display in the digest. Defaults to today.

    Returns
    -------
    (subject, html, text) triple ready to pass to a Notifier.
    """
    run_date = run_date or datetime.date.today()
    min_conf = settings.min_confidence_to_report

    above = [f for f in forecasts if f.confidence >= min_conf]
    below = [f.ticker for f in forecasts if f.confidence < min_conf]

    # Sort by confidence descending
    above.sort(key=lambda f: f.confidence, reverse=True)

    signals = [_to_signal_row(f) for f in above]

    ctx = {
        "run_date": run_date.strftime("%Y-%m-%d"),
        "signals": signals,
        "skipped": below,
        "min_confidence": min_conf,
        "disclaimer": settings.disclaimer,
    }

    html = _env.get_template("email.html").render(**ctx)
    text = _env.get_template("email.txt").render(**ctx)
    subject = f"StochSignal Weekly Digest — {run_date} ({len(signals)} signals)"

    return subject, html, text
