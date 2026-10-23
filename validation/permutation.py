"""
`permutation`: Monte Carlo permutation test of the full pipeline, including the in-sample sweep.

  1. Real development-period data -> features -> 108-point sweep -> best in-sample result S
     (best Sharpe and best mean net bps among points with >= PERM_MIN_TRADES trades).
  2. Permuted dataset: within each stock and day, shuffle the order of the 5-min residual
     returns and rebuild prices from (shuffled residuals + original ETF returns x beta). This
     destroys any intraday reversal pattern but keeps each day's volatility, fat tails and
     sector co-movement. News timestamps move to random trading days of the same stock
     (time of day kept).
  3. Same pipeline incl. the sweep on the permuted data -> best result.
  4. Repeat N times; p = (1 + #{permuted best >= S}) / (N + 1).

Results are appended to permutations.csv as they finish, so an interrupted run resumes.
"""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as bt
import config as C
import report as rp
import signals as sg
from validation import common as cm
from validation import trial_log
from validation.common import Ctx, write_summary

S = C.BARS_PER_DAY


# ---------------------------------------------------------------------------
# Permuted data
# ---------------------------------------------------------------------------
PERMUTE_CHUNK = 6     # stocks processed at a time (bounds the float64 temporaries; part of the RNG stream)


def permute_panel(panel: sg.Panel, feats: sg.Features, rng: np.random.Generator) -> sg.Panel:
    """Shuffle each stock-day's residual returns across its 5-min slots and rebuild OHLC."""
    N = panel.n_stocks
    out = [a.copy() for a in (panel.open, panel.high, panel.low, panel.close)]   # float32; ETF rows stay as is
    for a in range(0, N, PERMUTE_CHUNK):
        rows = slice(a, min(a + PERMUTE_CHUNK, N))
        new = _permute_rows(panel, feats, rows, rng)
        for arr, v in zip(out, new):
            arr[rows] = v
    return sg.make_panel(panel.symbols, N, panel.etf_idx, panel.days, *out, panel.daily_dollar_vol)


def _permute_rows(panel: sg.Panel, feats: sg.Features, rows: slice, rng) -> tuple[np.ndarray, ...]:
    rs = feats.ret[rows].astype(np.float64)
    re = np.nan_to_num(feats.ret[panel.etf_idx[rows]].astype(np.float64))
    beta = np.nan_to_num(feats.beta[rows])[..., None]     # warm-up days: beta 0 -> shuffle raw returns
    eps = rs - beta * re
    order = np.argsort(rng.random(eps.shape), axis=-1)
    r_new = beta * re + np.take_along_axis(eps, order, axis=-1)
    del eps, order, re

    o, h, lo, c = (a[rows].astype(np.float64) for a in (panel.open, panel.high, panel.low, panel.close))
    has = panel.has_bar[rows]
    log_o = np.log(o)
    first_open = sg._ffill_last_axis(log_o[..., ::-1])[..., ::-1][..., 0:1]   # first traded open of the day
    log_c = first_open + np.cumsum(r_new, axis=-1)
    del r_new
    # Keep each bar's open-vs-previous-close gap where it was (microstructure stays in place).
    prev_c = np.log(panel.close_ff[rows].astype(np.float64))
    log_o_new = np.empty_like(log_c)
    log_o_new[..., 0] = log_o[..., 0]
    log_o_new[..., 1:] = log_c[..., :-1] + np.nan_to_num(log_o[..., 1:] - prev_c[..., :-1])
    del prev_c, log_o
    o_new, c_new = np.exp(log_o_new), np.exp(log_c)
    del log_o_new, log_c
    with np.errstate(invalid="ignore", divide="ignore"):
        h_new = np.fmax(o_new, c_new) * (h / np.fmax(o, c))
        l_new = np.fmin(o_new, c_new) * (lo / np.fmin(o, c))

    keep_orig = np.isnan(rs).all(axis=-1)[..., None] | np.isnan(first_open)   # invalid stock-days
    result = []
    for new, orig in zip((o_new, h_new, l_new, c_new), (o, h, lo, c)):
        arr = np.where(keep_orig, orig, new).astype(sg.F)
        arr[~has] = np.nan
        result.append(arr)
    return tuple(result)


def permute_news(news: pd.DataFrame, panel: sg.Panel, rng: np.random.Generator) -> pd.DataFrame:
    """Company-specific news moved to random trading days of the same stock (time of day kept)."""
    n = news[["created_at", "symbols"]].copy()
    n["symbols"] = n["symbols"].apply(lambda x: list(x) if x is not None else [])
    n = n[n["symbols"].apply(len).between(1, C.MAX_NEWS_SYMBOLS)].explode("symbols")
    n = n[n["symbols"].isin(panel.stocks)]
    local = pd.to_datetime(n["created_at"], utc=True).dt.tz_convert(C.TIMEZONE).dt.tz_localize(None)
    tod = local - local.dt.normalize()
    new_day = panel.days.values[rng.integers(0, len(panel.days), size=len(n))]
    created = pd.DatetimeIndex(new_day + tod.to_numpy()).tz_localize(
        C.TIMEZONE, nonexistent="shift_forward", ambiguous="NaT").tz_convert("UTC")
    out = pd.DataFrame({"created_at": created, "symbols": n["symbols"].apply(lambda s: [s]).to_numpy()})
    return out.dropna(subset=["created_at"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline: features -> in-sample sweep -> best result
# ---------------------------------------------------------------------------
def best_of(sweep: pd.DataFrame) -> dict:
    ok = sweep[sweep["trades"] >= C.PERM_MIN_TRADES]
    if ok.empty:
        return {"best_sharpe": np.nan, "best_net_bps": np.nan, "params_sharpe": None}
    top = ok.loc[ok["sharpe"].idxmax()]
    return {"best_sharpe": float(ok["sharpe"].max()), "best_net_bps": float(ok["mean_net_bps"].max()),
            "params_sharpe": {k: top[k] for k in bt.Params().as_dict()}}


def pipeline(panel: sg.Panel, news: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    feats = sg.compute_features(panel, news, C.LOOKBACK_BARS)
    days = panel.days[feats.tradable.any(axis=0)]
    sweep = bt.run_sweep(panel, feats, days, verbose=False)
    return best_of(sweep), sweep


def _one_permutation(k: int) -> dict:
    sh = cm.SHARED
    rng = cm.rng_for(sh["seed"], "permutation", k)
    t0 = time.time()
    panel = permute_panel(sh["panel"], sh["feats"], rng)
    news = permute_news(sh["news"], sh["panel"], rng)
    best, _ = pipeline(panel, news)
    return {"perm": k, **{k2: v for k2, v in best.items() if k2 != "params_sharpe"},
            "params_sharpe": json.dumps(best["params_sharpe"], default=float), "seconds": time.time() - t0}


def dev_panel(ctx: Ctx) -> sg.Panel:
    last = ctx.panel.days.get_loc(ctx.periods["dev"][-1])
    return sg.panel_subset(ctx.panel, 0, last + 1)


def run(ctx: Ctx) -> dict:
    out = ctx.out("permutation")
    n_perm = ctx.opt("permutations", C.PERMUTATION_N)
    workers = ctx.opt("heavy_workers", C.N_WORKERS_HEAVY)

    panel = dev_panel(ctx)
    print(f"real pipeline on the development period ({len(panel.days)} days) ...", flush=True)
    t0 = time.time()
    real_best, real_sweep = pipeline(panel, ctx.news)
    real_sweep.to_csv(out / "real_sweep.csv", index=False)
    trial_log.log(real_sweep.to_dict("records"), ctx.dataset, "permutation_real_sweep", "dev")
    print(f"  S: best Sharpe {real_best['best_sharpe']:.2f}, best net {real_best['best_net_bps']:+.2f} bps "
          f"[{time.time() - t0:.0f}s]", flush=True)

    # Resume support: keep finished permutations if the setup is unchanged.
    sig = {"dataset": ctx.dataset, "days": [str(panel.days[0].date()), str(panel.days[-1].date())],
           "n_stocks": panel.n_stocks, "grid": C.SWEEP_GRID, "seed": ctx.seed, "min_trades": C.PERM_MIN_TRADES,
           "real_best_sharpe": round(real_best["best_sharpe"], 6), "permute_chunk": PERMUTE_CHUNK}
    csv, meta = out / "permutations.csv", out / "permutations_meta.json"
    done = pd.DataFrame()
    if csv.exists() and meta.exists() and json.loads(meta.read_text()) == json.loads(json.dumps(sig)):
        done = pd.read_csv(csv)
    else:
        csv.unlink(missing_ok=True)
        meta.write_text(json.dumps(sig))
    todo = [k for k in range(n_perm) if k not in set(done.get("perm", []))]
    print(f"permutations: {len(done)} done, {len(todo)} to run on {workers} workers", flush=True)

    feats_dev = sg.compute_features(panel, ctx.news, C.LOOKBACK_BARS)
    cm.SHARED.update(panel=panel, feats=feats_dev, news=ctx.news, seed=ctx.seed)
    t0 = time.time()
    for n, (_, r) in enumerate(cm.imap_guarded(_one_permutation, todo, workers, "permutations"), 1):
        pd.DataFrame([r]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        if n % 5 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n}/{len(todo)} permutations [{el:.0f}s, ~{el / n * (len(todo) - n) / 60:.0f} min left, "
                  f"{cm.mem_available_mb():.0f} MB free]", flush=True)
    cm.SHARED.clear()

    perms = pd.read_csv(csv).sort_values("perm").head(n_perm)
    N = len(perms)
    res = {"dataset": ctx.dataset, "n_permutations": N, "real": real_best}
    for metric in ("best_sharpe", "best_net_bps"):
        s = real_best[metric]
        exceed = int((perms[metric] >= s).sum())
        res[metric] = {"S": s, "n_ge_S": exceed, "p_value": (1 + exceed) / (N + 1),
                       "perm_median": float(perms[metric].median()), "perm_p95": float(perms[metric].quantile(0.95))}
    res["pass"] = bool(res["best_sharpe"]["p_value"] < 0.01 and res["best_net_bps"]["p_value"] < 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=rp.SURFACE)
    for ax, metric, label in zip(axes, ("best_sharpe", "best_net_bps"), ("best in-sample Sharpe", "best in-sample net bps")):
        rp._style(ax)
        ax.hist(perms[metric].dropna(), bins=40, color=rp.SERIES[0])
        s = real_best[metric]
        ax.axvline(s, color=rp.SERIES[1], linewidth=2)
        ax.annotate(f"real S = {s:.2f}\np = {res[metric]['p_value']:.4f}", (s, ax.get_ylim()[1] * 0.85),
                    xytext=(-4, 0), textcoords="offset points", ha="right", fontsize=8, color=rp.INK_2)
        ax.set_title(f"{label}: {N} permutations vs real", loc="left", fontsize=9, color=rp.INK)
    fig.tight_layout()
    fig.savefig(out / "permutation_hist.png", dpi=130, facecolor=rp.SURFACE)
    plt.close(fig)

    lines = [f"data: {ctx.dataset}; development period {sig['days'][0]} .. {sig['days'][1]}; "
             f"{N} permutations; sweep of {len(real_sweep)} points per run (min {C.PERM_MIN_TRADES} trades)",
             f"real best params (by Sharpe): {real_best['params_sharpe']}", ""]
    for metric in ("best_sharpe", "best_net_bps"):
        r = res[metric]
        lines.append(f"{metric:13s}: S = {r['S']:.3f}; permuted median {r['perm_median']:.3f}, 95th pct "
                     f"{r['perm_p95']:.3f}; {r['n_ge_S']} permutations >= S -> p = {r['p_value']:.4f}")
    lines.append(f"\n{'PASS' if res['pass'] else 'FAIL'}: p < 0.01 required for both metrics"
                 + ("" if N >= 99 else f"  (with only {N} permutations the smallest attainable p is {1 / (N + 1):.3f})"))
    write_summary(out, "PERMUTATION - Monte Carlo permutation test (full pipeline incl. sweep)", lines, res)
    return res
