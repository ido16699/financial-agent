# StochSignal

Personal research agent that predicts weekly directional movement (up/down) for a watchlist of stocks and emails high-confidence picks Sunday evenings.

**This is a research project, not investment advice.**

## Model

Baseline: Geometric Brownian Motion (GBM) calibrated from rolling daily log-returns.

Drift perturbation series:

```
μ(ε) = μ₀ + ε·μ₁ + ε²·μ₂
```

- **Order 1** (`μ₁`): linear in VADER news sentiment + Google Trends z-score
- **Order 2** (`μ₂`): quadratic + cross terms (saturation on diagonals, reinforcement on cross)
- **Confidence** = `clip(1 − |Δ₂|/|Δ₁|, 0, 1)` — series convergence proxy

Coefficients (`epsilon_news`, `epsilon_trends`) are config-driven, not fit. Data-driven coefficients are v2.

## Stack

Python 3.11, uv, numpy/scipy/pandas, yfinance, pytrends, NewsAPI (GDELT fallback), VADER, Jinja2, click.

## Setup

```bash
cp .env.example .env
# Fill in NEWS_API_KEY and SMTP credentials

uv venv && uv pip install -e ".[dev]"
```

## Usage

```bash
# Weekly run (Sunday evening)
stochsignal --dry-run              # preview digest, no email
stochsignal                        # send email

# Debug a single ticker
calibrate AAPL

# 2026 day-by-day replay
replay --start 2026-01-06 --end 2026-04-01
```

## Backtest

```bash
pytest tests/
```

Price-only GBM backtest covers 2020–2025.
Full perturbed model runs forward from first real execution (no look-ahead leakage on news/trends).

## Goals

- Calibration (reliability diagram) before anything else
- Hit rate > 55% on high-confidence picks

---
*Not investment advice. For personal research only.*
