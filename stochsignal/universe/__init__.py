"""Adaptive stock-universe selection (point-in-time, no look-ahead)."""
from stochsignal.universe.adaptive import select_universe
from stochsignal.universe.seed_pool import SEED_POOL

__all__ = ["select_universe", "SEED_POOL"]
