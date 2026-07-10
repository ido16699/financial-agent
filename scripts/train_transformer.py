"""Train the many-to-one MarketTransformer on the daily sequence dataset.

Consumes the tensors written by ``stochsignal.model.sequence_dataset`` and
trains with Gaussian-NLL (predict next-horizon return + uncertainty). Designed
to run on a Colab GPU but falls back to CPU/MPS.

Usage (Colab):
    python -m scripts.train_transformer --data data/sequences \
        --epochs 40 --batch-size 512 --d-model 128 --layers 4
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from stochsignal.model.transformer import (
    MarketTransformer, ModelConfig, gaussian_nll, prob_up,
)
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)
CHECKPOINT = Path("config/transformer_model.pt")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SeqDataset(Dataset):
    """Loads one split. X is memory-mapped so we never hold two copies."""

    def __init__(self, data_dir: Path, split: str):
        self.X = np.load(data_dir / "X.npy", mmap_mode="r")
        y = np.load(data_dir / "y.npy")
        s = np.load(data_dir / "split.npy")
        self.idx = np.where(s == split)[0]
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        x = torch.from_numpy(np.ascontiguousarray(self.X[j])).float()
        return x, torch.tensor(self.y[j])


def cosine_warmup(optimizer, warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    nll_sum = n = correct = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mu, log_var = model(x)
        nll_sum += gaussian_nll(mu, log_var, y).item() * len(y)
        # directional hit-rate: predicted up (p>0.5) vs actual up (y>0)
        pred_up = prob_up(mu, log_var) > 0.5
        correct += (pred_up == (y > 0)).sum().item()
        n += len(y)
    return {"nll": nll_sum / n, "hit_rate": correct / n}


@click.command()
@click.option("--data", default="data/sequences", show_default=True)
@click.option("--epochs", default=40, type=int, show_default=True)
@click.option("--batch-size", default=512, type=int, show_default=True)
@click.option("--lr", default=3e-4, type=float, show_default=True)
@click.option("--weight-decay", default=1e-2, type=float, show_default=True)
@click.option("--d-model", default=128, type=int, show_default=True)
@click.option("--layers", default=4, type=int, show_default=True)
@click.option("--heads", default=8, type=int, show_default=True)
@click.option("--dropout", default=0.2, type=float, show_default=True)
@click.option("--patience", default=6, type=int, show_default=True,
              help="Early-stop after this many epochs with no val improvement.")
@click.option("--num-workers", default=2, type=int, show_default=True)
def main(data, epochs, batch_size, lr, weight_decay, d_model, layers, heads,
         dropout, patience, num_workers):
    data_dir = Path(data)
    meta = json.loads((data_dir / "meta.json").read_text())
    device = pick_device()
    log.info("Device: %s | features=%d seq_len=%d horizon=%d",
             device, meta["n_features"], meta["seq_len"], meta["horizon"])

    train_ds = SeqDataset(data_dir, "train")
    val_ds = SeqDataset(data_dir, "val")
    log.info("train=%d val=%d sequences", len(train_ds), len(val_ds))

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)

    cfg = ModelConfig(
        n_features=meta["n_features"], seq_len=meta["seq_len"],
        d_model=d_model, n_heads=heads, n_layers=layers,
        d_ff=d_model * 2, dropout=dropout, horizon=meta["horizon"],
    )
    model = MarketTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %.2fM params", n_params / 1e6)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    sched = cosine_warmup(optim, warmup_steps=int(0.05 * total_steps), total_steps=total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_nll = float("inf")
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                mu, log_var = model(x)
                loss = gaussian_nll(mu, log_var, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item()

        val = evaluate(model, val_loader, device)
        log.info("epoch %2d | train_nll %.4f | val_nll %.4f | val_hit %.3f | lr %.2e",
                 epoch, running / len(train_loader), val["nll"], val["hit_rate"],
                 sched.get_last_lr()[0])

        if val["nll"] < best_nll - 1e-4:
            best_nll = val["nll"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("Early stopping at epoch %d (best val_nll %.4f)", epoch, best_nll)
                break

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state or model.state_dict(),
        "config": cfg.to_dict(),
        "meta": meta,               # feature names + scaler mean/std + splits
        "best_val_nll": best_nll,
    }, CHECKPOINT)
    log.info("Saved checkpoint -> %s (best val_nll %.4f)", CHECKPOINT, best_nll)


if __name__ == "__main__":
    main()
