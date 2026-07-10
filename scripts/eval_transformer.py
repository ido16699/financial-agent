"""Honest evaluation of the trained MarketTransformer on the HOLDOUT split.

Loads config/transformer_model.pt and the sequence dataset, runs inference on the
2017-2019 test split (never seen during training), and reports the metrics that
actually tell you whether the net has edge:

  * NLL / MAE           — probabilistic fit
  * directional hit-rate — % of next-5d moves called correctly
  * Information Coefficient (Spearman corr of predicted mu vs realised return)
  * decile calibration   — predicted P(up) vs empirical up-rate
  * long-minus-short spread — mean realised return of the top vs bottom predicted decile

IC and the long/short spread are the honest "is there alpha?" numbers. A net that
overfits will show great val NLL but ~0 IC on this holdout.

Usage:
    python -m scripts.eval_transformer --data data/sequences --split test
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import torch
from scipy.stats import spearmanr

from stochsignal.model.transformer import MarketTransformer, ModelConfig, prob_up, gaussian_nll
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)
CHECKPOINT = Path("config/transformer_model.pt")


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**ckpt["config"])
    model = MarketTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["meta"]


@torch.no_grad()
def predict(model, X: np.ndarray, device, batch: int = 1024):
    mus, lvs = [], []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).float().to(device)
        mu, lv = model(xb)
        mus.append(mu.cpu().numpy())
        lvs.append(lv.cpu().numpy())
    return np.concatenate(mus), np.concatenate(lvs)


@click.command()
@click.option("--data", default="data/sequences", show_default=True)
@click.option("--split", default="test", show_default=True,
              type=click.Choice(["val", "test"]))
@click.option("--checkpoint", default=str(CHECKPOINT), show_default=True)
def main(data, split, checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data)

    model, meta = load_model(Path(checkpoint), device)
    X = np.load(data_dir / "X.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy")
    s = np.load(data_dir / "split.npy")
    idx = np.where(s == split)[0]
    Xs, ys = np.ascontiguousarray(X[idx]), y[idx]
    log.info("Evaluating %s split: %d sequences", split, len(ys))

    mu, log_var = predict(model, Xs, device)
    p_up = prob_up(torch.tensor(mu), torch.tensor(log_var)).numpy()

    # --- core metrics ------------------------------------------------------- #
    nll = gaussian_nll(torch.tensor(mu), torch.tensor(log_var), torch.tensor(ys)).item()
    mae = float(np.mean(np.abs(mu - ys)))
    hit = float(np.mean((p_up > 0.5) == (ys > 0)))
    ic, _ = spearmanr(mu, ys)

    log.info("NLL        : %.4f", nll)
    log.info("MAE        : %.5f", mae)
    log.info("Hit-rate   : %.4f   (0.50 = coin flip)", hit)
    log.info("IC (Spearman mu vs realised): %.4f   (>0.03 is a real signal)", ic)

    # --- decile calibration + long/short spread ----------------------------- #
    order = np.argsort(mu)
    deciles = np.array_split(order, 10)
    log.info("Decile |  pred P(up)  emp up-rate  mean realised")
    d_rets = []
    for d, grp in enumerate(deciles):
        d_rets.append(ys[grp].mean())
        log.info("  %2d   |    %.3f        %.3f       %+.4f",
                 d, p_up[grp].mean(), (ys[grp] > 0).mean(), ys[grp].mean())
    spread = d_rets[-1] - d_rets[0]
    log.info("Long(top decile) - Short(bottom decile) 5d spread: %+.4f", spread)
    log.info("  (annualised ~ %.1f%% if traded weekly, before costs)",
             spread * 52 * 100)

    out = {
        "split": split, "n": int(len(ys)), "nll": nll, "mae": mae,
        "hit_rate": hit, "ic": float(ic), "ls_spread_5d": float(spread),
    }
    (data_dir / f"eval_{split}.json").write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", data_dir / f"eval_{split}.json")


if __name__ == "__main__":
    main()
