"""
Shared plumbing for the validation suite: loading data (with the holdout lock),
period definitions, a feature cache, reproducible RNG streams, a fork-based
parallel map, and small output helpers.

Every validation module calls the strategy through backtest.run_strategy /
simulate_trades / run_sweep and signals.compute_features - never a copy.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import multiprocessing as mp
import pickle
import time
import warnings
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import config as C
import signals as sg


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    panel: sg.Panel
    feats: sg.Features
    news: pd.DataFrame
    periods: dict[str, pd.DatetimeIndex]       # dev, wf_oos, all (dev+wf_oos), [holdout]
    synthetic: bool
    workers: int
    seed: int
    out_root: Path
    holdout_unlocked: bool = False
    truth: dict | None = None
    opts: dict = field(default_factory=dict)  # CLI overrides (n permutations, ...)

    @property
    def dataset(self) -> str:
        return "synthetic" if self.synthetic else "real"

    @property
    def headline(self) -> str:
        return "wf_oos"

    def out(self, sub: str) -> Path:
        p = self.out_root / sub
        p.mkdir(parents=True, exist_ok=True)
        return p

    def mask(self, period: str) -> np.ndarray:
        return self.panel.days.isin(self.periods[period])

    def rng(self, *keys) -> np.random.Generator:
        return rng_for(self.seed, *keys)

    def opt(self, name: str, default):
        v = self.opts.get(name)
        return default if v is None else v


def rng_for(seed: int, *keys) -> np.random.Generator:
    """Independent, reproducible stream per (seed, key...) regardless of worker scheduling."""
    ints = [k if isinstance(k, (int, np.integer)) else zlib.crc32(str(k).encode()) for k in keys]
    return np.random.default_rng([seed, *ints])


# ---------------------------------------------------------------------------
# Periods and the holdout lock
# ---------------------------------------------------------------------------
def holdout_start(days: pd.DatetimeIndex, synthetic: bool) -> pd.Timestamp:
    if synthetic:
        dev, wf, _ = C.VALIDATION_SYNTH_SPLITS
        return days[int(round(len(days) * (dev + wf)))]
    return pd.Timestamp(C.HOLDOUT_START)


def make_periods(days: pd.DatetimeIndex, tradable_any: np.ndarray, synthetic: bool,
                 unlocked: bool) -> dict[str, pd.DatetimeIndex]:
    """dev / wf_oos / all (= dev + wf_oos) / holdout (only when unlocked) as tradable-day indexes."""
    ev = days[tradable_any]
    if synthetic:
        cal = synthetic_calendar()
        dev_end = cal[int(round(len(cal) * C.VALIDATION_SYNTH_SPLITS[0]))]
        wf_start = dev_end
    else:
        dev_end = pd.Timestamp(C.DEV_END) + pd.Timedelta(days=1)
        wf_start = pd.Timestamp(C.WF_OOS_START)
    h0 = holdout_start(synthetic_calendar() if synthetic else days, synthetic)
    periods = {"dev": ev[ev < dev_end], "wf_oos": ev[(ev >= wf_start) & (ev < h0)]}
    periods["all"] = periods["dev"].append(periods["wf_oos"])
    if unlocked:
        periods["holdout"] = ev[ev >= h0]
    return periods


def synthetic_calendar() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range(C.VALIDATION_SYNTH_START, periods=C.VALIDATION_SYNTH_DAYS))


def log_holdout_unlock(synthetic: bool, command: str) -> None:
    C.HOLDOUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with open(C.HOLDOUT_LOG_PATH, "a") as f:
        f.write(f"{stamp}\t{'synthetic' if synthetic else 'REAL'}\t{command}\n")
    msg = ("HOLDOUT UNLOCKED - this run sees the last 12 months. The holdout is now SPENT: any "
           "decision made after looking at it is no longer an out-of-sample test. Logged to "
           f"{C.HOLDOUT_LOG_PATH}.")
    warnings.warn(msg)
    print("!" * 80 + "\n" + msg + "\n" + "!" * 80, flush=True)


# ---------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------
def _cache_key(parts: dict) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _feature_config() -> dict:
    keys = ["BETA_LOOKBACK_DAYS", "MIN_LOOKBACK_DAYS", "MIN_BARS_PER_DAY", "MAX_NEWS_SYMBOLS", "MIN_PRICE",
            "MIN_AVG_DOLLAR_VOLUME", "DOLLAR_VOLUME_LOOKBACK_DAYS", "SKIP_EARLY_CLOSE_DAYS", "LOOKBACK_BARS"]
    return {k: getattr(C, k) for k in keys}


def load_real(stocks: list[str], sector_map: dict[str, str], unlocked: bool, label: str
              ) -> tuple[sg.Panel, sg.Features, pd.DataFrame]:
    """Panel + features from the Alpaca cache, cached on disk keyed by the data manifest."""
    import fetch_data
    if not C.MANIFEST_PATH.exists():
        raise SystemExit("No cached data. Run `python fetch_data.py` first (or use --synthetic).")
    key = _cache_key({"manifest": C.MANIFEST_PATH.stat().st_mtime_ns, "stocks": stocks, "map": sector_map,
                      "unlocked": unlocked, "holdout": C.HOLDOUT_START, "cfg": _feature_config(), "v": 2})
    path = C.CACHE_DIR / f"{label}_{key}.pkl"
    if path.exists():
        t0 = time.time()
        with open(path, "rb") as f:
            panel, feats, news = pickle.load(f)
        print(f"loaded cached features {path.name} [{time.time() - t0:.1f}s]", flush=True)
        return panel, feats, news
    t0 = time.time()
    news = fetch_data.load_news()
    h0 = None
    if not unlocked:                       # the holdout never enters memory
        h0 = pd.Timestamp(C.HOLDOUT_START).tz_localize(C.TIMEZONE)
        news = news[pd.to_datetime(news["created_at"], utc=True) < h0]
    panel = sg.build_panel_streaming(lambda s: fetch_data.load_symbol_bars(s, before=h0), stocks, sector_map,
                                     fetch_data.cached_symbols())
    feats = sg.compute_features(panel, news, C.LOOKBACK_BARS)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for old in C.CACHE_DIR.glob(f"{label}_*.pkl"):
        old.unlink()
    with open(path, "wb") as f:
        pickle.dump((panel, feats, news), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"built features for {label} [{time.time() - t0:.1f}s], cached to {path}", flush=True)
    return panel, feats, news


def load_synthetic(unlocked: bool, sector_map: dict[str, str] = C.SECTOR_MAP, seed: int = C.SYNTH_SEED):
    import synthetic
    bars, news, truth = synthetic.generate(n_days=C.VALIDATION_SYNTH_DAYS, sector_map=sector_map,
                                           seed=seed, start=C.VALIDATION_SYNTH_START)
    if not unlocked:
        h0 = holdout_start(synthetic_calendar(), True).tz_localize(C.TIMEZONE)
        bars = bars[bars["timestamp"] < h0]
        news = news[news["created_at"] < h0]
    panel = sg.build_panel(bars, list(sector_map), sector_map)
    del bars
    feats = sg.compute_features(panel, news, C.LOOKBACK_BARS)
    return panel, feats, news, truth


def load_context(synthetic: bool, unlocked: bool, workers: int, opts: dict, command: str) -> Ctx:
    if unlocked:
        log_holdout_unlock(synthetic, command)
    t0 = time.time()
    if synthetic:
        panel, feats, news, truth = load_synthetic(unlocked)
    else:
        panel, feats, news = load_real(C.STOCKS, C.SECTOR_MAP, unlocked, "main")
        truth = None
    periods = make_periods(panel.days, feats.tradable.any(axis=0), synthetic, unlocked)
    out_root = C.VALIDATION_SYNTH_DIR if synthetic else C.VALIDATION_DIR
    print(f"context: {'synthetic' if synthetic else 'real'} data, {panel.n_stocks} stocks, "
          f"{len(panel.days)} days; " + ", ".join(f"{k}={len(v)}d" for k, v in periods.items())
          + f" [{time.time() - t0:.1f}s]", flush=True)
    return Ctx(panel=panel, feats=feats, news=news, periods=periods, synthetic=synthetic, workers=workers,
               seed=C.VALIDATION_SEED, out_root=out_root, holdout_unlocked=unlocked, truth=truth, opts=opts)


# ---------------------------------------------------------------------------
# Memory-aware parallel map (fork: workers share the parent's read-only arrays copy-on-write)
# ---------------------------------------------------------------------------
SHARED: dict = {}


def mem_available_mb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return float("inf")


def _rss_mb(field: str) -> float:
    for line in open("/proc/self/status"):
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024
    return 0.0


def _measure(fn, task):
    """Run fn(task) in this process; return (result, peak extra memory in MB)."""
    try:
        with open("/proc/self/clear_refs", "w") as f:   # resets VmHWM (peak RSS) to the current RSS
            f.write("5")
    except OSError:
        pass
    before = _rss_mb("VmRSS")
    out = fn(task)
    return out, max(_rss_mb("VmHWM") - before, 20.0)


def imap_guarded(fn, tasks: list, workers: int, desc: str = ""):
    """
    Yield (index, result) as tasks finish. The first task runs in this process to measure its
    memory; the worker count is then capped so that (MemAvailable - MEM_RESERVE_MB) covers one
    task per worker, and no new task starts while free memory is below reserve + one task.
    """
    if not tasks:
        return
    t0 = time.time()
    # Probe in a fresh forked child: its peak-RSS growth is exactly the memory a worker adds
    # (in this process, previously freed heap would hide part of it).
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("fork")) as probe:
        first, per_task = probe.submit(_measure, fn, tasks[0]).result()
    yield 0, first
    rest = list(enumerate(tasks))[1:]
    budget = mem_available_mb() - C.MEM_RESERVE_MB
    n = int(max(1, min(workers, budget // (per_task * C.MEM_SAFETY_FACTOR)))) if workers > 1 else 1
    if desc:
        print(f"  {desc}: ~{per_task:.0f} MB per task, {mem_available_mb():.0f} MB available -> "
              f"{n} worker(s) (requested {workers})", flush=True)
    if n <= 1 or not rest:
        for k, t in rest:
            yield k, fn(t)
    else:
        with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("fork")) as ex:
            running = {}
            while rest or running:
                while rest and len(running) < n and (
                        not running or mem_available_mb() > C.MEM_RESERVE_MB + per_task * C.MEM_SAFETY_FACTOR):
                    k, t = rest.pop(0)
                    running[ex.submit(fn, t)] = k
                done, _ = wait(list(running), timeout=5, return_when=FIRST_COMPLETED)
                for fut in done:
                    yield running.pop(fut), fut.result()
    if desc:
        print(f"  {desc}: {len(tasks)} tasks [{time.time() - t0:.1f}s]", flush=True)


def pmap(fn, tasks: list, workers: int, desc: str = "", chunksize: int = 1) -> list:
    """Ordered map over tasks with memory-aware parallelism (see imap_guarded). fn reads big data from SHARED."""
    if chunksize > 1:     # group small tasks so per-task overhead and the memory probe stay meaningful
        groups = [tasks[s:s + chunksize] for s in range(0, len(tasks), chunksize)]
        nested = pmap(_apply_chunk, [(fn, g) for g in groups], workers, desc)
        return [x for g in nested for x in g]
    out = [None] * len(tasks)
    for k, r in imap_guarded(fn, tasks, workers, desc):
        out[k] = r
    return out


def _apply_chunk(arg):
    fn, group = arg
    return [fn(t) for t in group]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def summarize(port: pd.DataFrame, days: pd.DatetimeIndex, cost_mult: float = 1.0) -> dict:
    """Headline numbers for a sized portfolio over `days`."""
    pt = port[port["date"].isin(days)]
    m = bt.portfolio_metrics(pt, days, cost_mult)
    st = bt.trade_stats(pt, cost_mult)
    return {"n_trades": m["trades"], "mean_net_bps": m["mean_net_bps"], "mean_gross_bps": st["mean_gross_bps"],
            "t_stat": st["t_stat_day_clustered"], "win_rate": m["win_rate"], "sharpe": m["sharpe"],
            "ann_return": m["ann_return"], "max_dd_pct": m["max_drawdown_pct"], "total_pnl": m["total_pnl"],
            "avg_hold_min": m["avg_hold_min"]}


def daily_returns(port: pd.DataFrame, days: pd.DatetimeIndex, cost_mult: float = 1.0) -> pd.Series:
    return bt.daily_pnl(port[port["date"].isin(days)], days, f"pnl_{cost_mult:g}x") / C.CAPITAL


def percentile_of(value: float, dist: np.ndarray) -> float:
    dist = np.asarray(dist, float)
    dist = dist[np.isfinite(dist)]
    return float((dist < value).mean() * 100) if len(dist) else float("nan")


def block_bootstrap_indices(rng: np.random.Generator, n: int, length: int, block: int, reps: int) -> np.ndarray:
    """(reps, length) indices from a moving-block bootstrap of a length-n series."""
    n_blocks = -(-length // block)
    starts = rng.integers(0, max(n - block + 1, 1), size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :length]
    return np.minimum(idx, n - 1)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def write_summary(out: Path, title: str, lines: list[str], results: dict) -> None:
    text = "\n".join([title, "=" * len(title), *lines]) + "\n"
    (out / "summary.txt").write_text(text)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    print(text)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fmt_table(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f"{x:,.3f}")


def load_results(out_root: Path, sub: str) -> dict | None:
    p = out_root / sub / "results.json"
    return json.loads(p.read_text()) if p.exists() else None
