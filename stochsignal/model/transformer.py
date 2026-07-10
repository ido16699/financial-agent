"""Many-to-one Transformer that 'guesses the next move' of the market.

Reads a sequence of the last L trading days (each day = a feature vector /
"word", produced by sequence_dataset.py) and predicts the next-`horizon`-day
return as a GAUSSIAN: a mean (the guess) and a log-variance (its uncertainty).
The probabilistic head gives native confidence — no ad-hoc calibration hacks.

Architecture:
    x (B, L, F)
      -> Linear input projection to d_model
      -> + learned positional embedding
      -> N x TransformerEncoderLayer (multi-head self-attention, pre-norm)
      -> attention pooling (a learned query attends over the L timesteps) -> (B, d_model)
      -> MLP head -> (mu, log_var)   both (B,)

Requires torch (installed on Colab / GPU box, not in the base env).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_features: int
    seq_len: int = 64
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.2          # markets are low signal — regularise hard
    horizon: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


class AttentionPool(nn.Module):
    """Collapse (B, L, d) -> (B, d) via a single learned query attention."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # scores: (B, L) = h · query
        scores = torch.einsum("bld,d->bl", h, self.query) * self.scale
        weights = torch.softmax(scores, dim=1)            # (B, L)
        return torch.einsum("bl,bld->bd", weights, h)     # (B, d)


class MarketTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.seq_len, cfg.d_model) * 0.02)
        self.in_drop = nn.Dropout(cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # pre-norm: stabler training
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.pool = AttentionPool(cfg.d_model)

        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 2),      # -> (mu, log_var)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, F)
        h = self.input_proj(x) + self.pos_emb[:, : x.size(1)]
        h = self.in_drop(h)
        h = self.encoder(h)                 # (B, L, d_model)
        pooled = self.pool(h)               # (B, d_model)
        out = self.head(pooled)             # (B, 2)
        mu = out[:, 0]
        log_var = out[:, 1].clamp(-8.0, 4.0)   # keep variance numerically sane
        return mu, log_var


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of y under N(mu, exp(log_var)). Mean over batch."""
    inv_var = torch.exp(-log_var)
    return 0.5 * (log_var + (y - mu) ** 2 * inv_var).mean()


def prob_up(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """P(return > 0) = Phi(mu / sigma) under the predicted Gaussian."""
    sigma = torch.exp(0.5 * log_var).clamp_min(1e-6)
    # standard normal CDF via erf
    return 0.5 * (1.0 + torch.erf(mu / (sigma * math.sqrt(2.0))))
