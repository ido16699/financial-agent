"""Build daily sliding-window sequences for the Transformer market model.

The Transformer reads a sequence of the last ``L`` trading days (each day = a
feature "word") and predicts the next ``horizon``-day return (many-to-one).

Feature vector per day (HYBRID):
  * cheap raw/technical features computed EVERY day (vectorised, causal)
  * physics/DSP signals (GBM prob, wave signal, chaos regime) recomputed on a
    weekly cadence and forward-filled (they are expensive per-point)

Everything is built OFFLINE from the frozen price snapshot + the quarterly
adaptive-universe manifest, so it is fully reproducible and needs no network.

Leakage discipline:
  * every feature at day t uses ONLY data <= t
  * the standardisation scaler is fit on the TRAIN split only
  * an embargo gap (= horizon) is dropped around each chronological split cut so
    a train target never overlaps the validation/test window

Usage:
    python -m stochsignal.model.sequence_dataset \
        --out data/sequences --seq-len 64 --horizon 5
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd

from stochsignal.ingest.snapshot_loader import load_all_prices
from stochsignal.model.gbm import calibrate, prob_up_closed_form
from stochsignal.model.waves import analyse_waves
from stochsignal.model.chaos import detect_regime
from stochsignal.logging_utils import get_logger

log = get_logger(__name__)

RAW_FEATURES = [
    "r1", "r5", "r10", "r21", "r63",       # multi-lag log returns
    "vol21", "vol63",                       # rolling realised vol
    "volume_z",                             # volume vs its own rolling mean
    "range_pct",                            # daily high-low range / close
    "ma20_gap", "ma50_gap", "ma200_gap",    # distance from moving averages
]
PHYSICS_FEATURES = ["gbm_prob", "wave_signal", "regime_mult"]
FEATURE_NAMES = RAW_FEATURES + PHYSICS_FEATURES

# Default chronological splits (by target date). Embargo handled separately.
DEFAULT_SPLITS = {
    "train": ("2000-01-01", "2013-12-31"),
    "val": ("2014-01-01", "2016-12-31"),
    "test": ("2017-01-01", "2019-12-31"),
}

PHYSICS_WINDOW = 252  # trailing days fed to each physics computation


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised cheap features. Every column at row t uses only data <= t."""
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    logc = np.log(close)
    r1 = logc.diff(1)

    feats = pd.DataFrame(index=df.index)
    feats["r1"] = r1
    feats["r5"] = logc.diff(5)
    feats["r10"] = logc.diff(10)
    feats["r21"] = logc.diff(21)
    feats["r63"] = logc.diff(63)
    feats["vol21"] = r1.rolling(21).std()
    feats["vol63"] = r1.rolling(63).std()
    vmean = volume.rolling(63).mean()
    vstd = volume.rolling(63).std()
    feats["volume_z"] = ((volume - vmean) / vstd).clip(-5, 5)
    feats["range_pct"] = (high - low) / close
    feats["ma20_gap"] = close / close.rolling(20).mean() - 1.0
    feats["ma50_gap"] = close / close.rolling(50).mean() - 1.0
    feats["ma200_gap"] = close / close.rolling(200).mean() - 1.0
    return feats


def _physics_features(df: pd.DataFrame, ticker: str,
                      every: int, horizon: int) -> pd.DataFrame:
    """Expensive physics signals, computed on a coarse cadence and forward-filled.

    At each cadence day we slice the trailing ``PHYSICS_WINDOW`` closes (data <=
    that day) and run the GBM / wave / chaos models on it.
    """
    close = df["Close"].astype(float)
    idx = df.index
    rows: dict[pd.Timestamp, list[float]] = {}

    # cadence anchors: every `every` trading days once we have enough history
    for i in range(PHYSICS_WINDOW, len(idx), every):
        window_close = close.iloc[i - PHYSICS_WINDOW: i + 1].values
        if len(window_close) < 80 or np.any(~np.isfinite(window_close)):
            continue
        log_ret = np.log(window_close[1:] / window_close[:-1])
        log_ret_series = pd.Series(log_ret, index=idx[i - PHYSICS_WINDOW + 1: i + 1])

        try:
            params = calibrate(ticker, log_ret_series)
            gbm_prob = prob_up_closed_form(params, horizon)
        except ValueError:
            gbm_prob = 0.5
        try:
            wave_signal = analyse_waves(log_ret, window_close, horizon_days=horizon).wave_signal
        except Exception:
            wave_signal = 0.0
        try:
            regime_mult = detect_regime(log_ret).confidence_multiplier
        except Exception:
            regime_mult = 1.0

        rows[idx[i]] = [gbm_prob, wave_signal, regime_mult]

    if not rows:
        return pd.DataFrame(index=idx, columns=PHYSICS_FEATURES, dtype=float)

    phys = pd.DataFrame.from_dict(rows, orient="index", columns=PHYSICS_FEATURES)
    # forward-fill onto the daily index (only past anchors influence each day)
    phys = phys.reindex(idx).ffill()
    return phys


def _ticker_frame(df: pd.DataFrame, ticker: str,
                  every: int, horizon: int) -> pd.DataFrame:
    """Full per-day feature frame + forward target for one ticker."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    raw = _raw_features(df)
    phys = _physics_features(df, ticker, every=every, horizon=horizon)
    feats = pd.concat([raw, phys], axis=1)[FEATURE_NAMES]

    close = df["Close"].astype(float)
    # forward `horizon`-day log return (the label). Uses FUTURE data by design;
    # only ever attached to the sample whose decision date is t.
    feats["target"] = np.log(close.shift(-horizon) / close)
    return feats


# --------------------------------------------------------------------------- #
# Universe (point-in-time membership)
# --------------------------------------------------------------------------- #
def _load_manifest(path: Path) -> tuple[list[pd.Timestamp], dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Universe manifest not found at {path}. Run: "
            "python -m scripts.build_universe_snapshots --start 2000-01-01 --size 100"
        )
    manifest = json.loads(path.read_text())
    dates = sorted(pd.Timestamp(d) for d in manifest.keys())
    return dates, {pd.Timestamp(k): set(v) for k, v in manifest.items()}


def _active_on(day: pd.Timestamp, snap_dates: list[pd.Timestamp],
               manifest: dict) -> set[str]:
    """Universe as known strictly on-or-before `day` (last snapshot <= day)."""
    eligible = [d for d in snap_dates if d <= day]
    if not eligible:
        return set()
    return manifest[eligible[-1]]


# --------------------------------------------------------------------------- #
# Split assignment with embargo
# --------------------------------------------------------------------------- #
def _assign_split(dates: np.ndarray, splits: dict, horizon: int) -> np.ndarray:
    """Return an array of split labels; '' means dropped (embargo / out-of-range)."""
    out = np.full(len(dates), "", dtype=object)
    ts = pd.to_datetime(dates)
    for name, (lo, hi) in splits.items():
        lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
        # embargo: pull the upper edge in by `horizon` calendar-ish days so a
        # sample's forward-looking target cannot bleed into the next split.
        hi_embargo = hi_ts - pd.Timedelta(days=horizon + 2)
        mask = (ts >= lo_ts) & (ts <= hi_embargo)
        out[mask.values] = name
    return out


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
@dataclass
class BuildStats:
    n_sequences: int
    n_tickers: int
    per_split: dict


def build(out_dir: Path, seq_len: int, horizon: int, physics_every: int,
          snapshot: str | None, manifest_path: Path,
          splits: dict) -> BuildStats:
    # The physics models log one INFO line per computation; at ~300k points that
    # would drown the build. Silence them for the duration.
    import logging
    for name in ("stochsignal.model.waves", "stochsignal.model.chaos"):
        logging.getLogger(name).setLevel(logging.WARNING)

    prices = load_all_prices(snapshot)
    snap_dates, manifest = _load_manifest(manifest_path)

    # only bother with tickers that ever enter the universe
    universe_tickers = set().union(*manifest.values()) if manifest else set(prices)
    tickers = sorted(t for t in universe_tickers if t in prices)
    log.info("Building sequences for %d universe tickers (L=%d, horizon=%d)",
             len(tickers), seq_len, horizon)

    X_list, y_list, date_list, tk_list = [], [], [], []

    for n, ticker in enumerate(tickers, 1):
        feats = _ticker_frame(prices[ticker], ticker,
                              every=physics_every, horizon=horizon)
        mat = feats[FEATURE_NAMES].to_numpy(dtype=np.float64)
        target = feats["target"].to_numpy(dtype=np.float64)
        idx = feats.index

        valid = np.all(np.isfinite(mat), axis=1)  # full feature row present
        for i in range(seq_len - 1, len(idx)):
            if not np.isfinite(target[i]):
                continue
            day = idx[i]
            if ticker not in _active_on(day, snap_dates, manifest):
                continue
            window = valid[i - seq_len + 1: i + 1]
            if not window.all():
                continue
            X_list.append(mat[i - seq_len + 1: i + 1])
            y_list.append(target[i])
            date_list.append(day)
            tk_list.append(ticker)

        if n % 25 == 0:
            log.info("  ...%d/%d tickers, %d sequences so far", n, len(tickers), len(y_list))

    if not y_list:
        raise RuntimeError("No sequences produced — check snapshot/manifest coverage.")

    X = np.asarray(X_list, dtype=np.float32)           # (N, L, F)
    y = np.asarray(y_list, dtype=np.float32)           # (N,)
    dates = np.array([d.strftime("%Y-%m-%d") for d in date_list])
    tks = np.array(tk_list)
    split = _assign_split(dates, splits, horizon)

    # --- fit scaler on TRAIN rows only, then apply to everything ------------- #
    train_mask = split == "train"
    if not train_mask.any():
        raise RuntimeError("No training sequences — check split date ranges.")
    train_flat = X[train_mask].reshape(-1, X.shape[-1])
    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std[std < 1e-8] = 1.0
    X = (X - mean) / std

    keep = split != ""
    X, y, dates, tks, split = X[keep], y[keep], dates[keep], tks[keep], split[keep]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "dates.npy", dates)
    np.save(out_dir / "tickers.npy", tks)
    np.save(out_dir / "split.npy", split.astype(str))
    (out_dir / "meta.json").write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "seq_len": seq_len,
        "horizon": horizon,
        "physics_every": physics_every,
        "n_features": X.shape[-1],
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "splits": splits,
        "n_sequences": int(len(y)),
    }, indent=2))

    per_split = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    log.info("Saved %d sequences to %s  |  splits=%s", len(y), out_dir, per_split)
    return BuildStats(n_sequences=len(y), n_tickers=len(tickers), per_split=per_split)


@click.command()
@click.option("--out", default="data/sequences", show_default=True)
@click.option("--seq-len", default=64, type=int, show_default=True,
              help="Lookback window (number of trading days per sequence).")
@click.option("--horizon", default=5, type=int, show_default=True,
              help="Forward horizon in trading days for the target return.")
@click.option("--physics-every", default=5, type=int, show_default=True,
              help="Recompute expensive physics signals every N trading days.")
@click.option("--snapshot", default=None,
              help="Path to price snapshot parquet (default data/prices_snapshot.parquet).")
@click.option("--manifest", "manifest_path",
              default="config/universe_snapshots/manifest.json", show_default=True)
def main(out, seq_len, horizon, physics_every, snapshot, manifest_path):
    stats = build(
        out_dir=Path(out), seq_len=seq_len, horizon=horizon,
        physics_every=physics_every, snapshot=snapshot,
        manifest_path=Path(manifest_path), splits=DEFAULT_SPLITS,
    )
    log.info("Done: %d sequences across %d tickers. %s",
             stats.n_sequences, stats.n_tickers, stats.per_split)


if __name__ == "__main__":
    main()
