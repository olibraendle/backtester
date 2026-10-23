"""
Persistent log of every backtest configuration run (results/trial_log.csv).

The Deflated Sharpe Ratio needs the number of independent strategy trials and the
variance of their Sharpe ratios, so every sweep point, walk-forward fit, jitter draw,
etc. is appended here. Null/permuted datasets are logged under their own `dataset`
label and do not count as trials of the real strategy.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

import config as C

COLUMNS = ["timestamp", "dataset", "source", "period", "lookback_bars", "z_entry", "z_exit", "hold_bars",
           "extra", "n_trades", "mean_net_bps", "sharpe"]
PARAM_COLS = ["lookback_bars", "z_entry", "z_exit", "hold_bars"]


def log(rows: list[dict], dataset: str, source: str, period: str) -> None:
    """Append rows with Params fields + n_trades/mean_net_bps/sharpe (+ optional `extra` dict)."""
    if not rows:
        return
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    recs = []
    for r in rows:
        extra = r.get("extra")
        recs.append({
            "timestamp": stamp, "dataset": dataset, "source": source, "period": period,
            **{k: r.get(k) for k in PARAM_COLS},
            "extra": json.dumps(extra, sort_keys=True) if extra else "",
            "n_trades": r.get("n_trades", r.get("trades")),
            "mean_net_bps": r.get("mean_net_bps"), "sharpe": r.get("sharpe"),
        })
    df = pd.DataFrame(recs, columns=COLUMNS)
    C.TRIAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TRIAL_LOG_PATH, mode="a", header=not C.TRIAL_LOG_PATH.exists(), index=False)


def read() -> pd.DataFrame:
    if not C.TRIAL_LOG_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(C.TRIAL_LOG_PATH)


# Sources whose runs were used to CHOOSE parameters (in-sample sweeps, walk-forward fits, heatmaps).
SELECTION_SOURCES = ("backtest_sweep", "permutation_real_sweep", "walk_forward_w", "heatmap")


def trial_stats(dataset: str, selection_only: bool = True) -> tuple[int, float]:
    """
    (number of distinct strategy configurations tried on `dataset`, variance of their
    annualized Sharpe ratios). Configurations are distinct parameter sets; the same
    parameters re-run under other settings count once (their mean Sharpe is used).
    selection_only: count only runs used to choose parameters (SELECTION_SOURCES), not
    robustness re-runs such as jitter draws or execution stresses.
    """
    df = read()
    df = df[(df["dataset"] == dataset) & np.isfinite(pd.to_numeric(df["sharpe"], errors="coerce"))]
    if selection_only:
        df = df[df["source"].astype(str).str.startswith(SELECTION_SOURCES)]
    if df.empty:
        return 0, float("nan")
    per_cfg = df.groupby(PARAM_COLS, dropna=False)["sharpe"].mean()
    return int(len(per_cfg)), float(per_cfg.var(ddof=1)) if len(per_cfg) > 1 else float("nan")
