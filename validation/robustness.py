"""
`robustness`
  * walk-forward: tune on rolling 2-year windows (max in-sample Sharpe over the sweep grid,
    >= WF_MIN_TRADES trades), trade the next 6 months, slide by 6 months; chained OOS equity
  * parameter heatmaps (development period): L x z_entry and z_exit x H; isolated peaks flagged
  * parameter jitter: JITTER_N draws within +-20% of the chosen parameters
  * breakdowns: year, sector, stock, entry hour, long/short, |z| bucket, VIX-proxy and
    realized-volatility regimes
  * concentration: without the top 1% of trades; leave-one-stock/sector/year-out
  * out-of-universe: unchanged parameters on a second list of large/mid caps
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as bt
import config as C
import report as rp
from validation import common as cm
from validation import trial_log
from validation.common import Ctx, fmt_table, summarize, write_summary

DEFAULT = bt.Params()


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def wf_windows(days: pd.DatetimeIndex) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    out, s = [], 0
    while s + C.WF_TRAIN_DAYS < len(days):
        out.append((days[s:s + C.WF_TRAIN_DAYS], days[s + C.WF_TRAIN_DAYS:s + C.WF_TRAIN_DAYS + C.WF_TEST_DAYS]))
        s += C.WF_TEST_DAYS
    return out


def _wf_window(task):
    k, train, test = task
    sh = cm.SHARED
    panel, feats = sh["panel"], sh["feats"]
    sweep = bt.run_sweep(panel, feats, train, verbose=False)
    ok = sweep[sweep["trades"] >= C.WF_MIN_TRADES]
    if ok.empty or not np.isfinite(ok["sharpe"]).any():
        return k, None, sweep, pd.DataFrame()
    best = ok.loc[ok["sharpe"].idxmax()]
    params = bt.Params(int(best.lookback_bars), float(best.z_entry), float(best.z_exit), int(best.hold_bars))
    port = bt.run_strategy(panel, feats, params, panel.days.isin(test))
    port["window"] = k
    return k, {**params.as_dict(), "is_sharpe": float(best.sharpe), "is_net_bps": float(best.mean_net_bps)}, sweep, port


def walk_forward(ctx: Ctx, out) -> dict:
    windows = wf_windows(ctx.periods["all"])
    cm.SHARED.update(panel=ctx.panel, feats=ctx.feats)
    results = cm.pmap(_wf_window, [(k, tr, te) for k, (tr, te) in enumerate(windows)], ctx.workers, "walk-forward windows")
    cm.SHARED.clear()
    chosen, ports = [], []
    for k, params, sweep, port in results:
        trial_log.log(sweep.to_dict("records"), ctx.dataset, f"walk_forward_w{k}", "train")
        tr, te = windows[k]
        rec = {"window": k, "train_start": tr[0].date(), "train_end": tr[-1].date(),
               "test_start": te[0].date(), "test_end": te[-1].date(), **(params or {})}
        if params is not None:
            m = summarize(port, te)
            rec.update(oos_trades=m["n_trades"], oos_net_bps=m["mean_net_bps"], oos_sharpe=m["sharpe"])
            ports.append(port)
        chosen.append(rec)
    chosen = pd.DataFrame(chosen)
    chosen.to_csv(out / "walk_forward_windows.csv", index=False)
    port = pd.concat(ports, ignore_index=True) if ports else bt._empty_trades()
    test_days = pd.DatetimeIndex(sorted(set().union(*[set(te) for _, te in windows]))) if windows else pd.DatetimeIndex([])
    m1, m2 = summarize(port, test_days, 1.0), summarize(port, test_days, 2.0)
    wf_days = test_days[test_days.isin(ctx.periods[ctx.headline])]
    m2_h = summarize(port, wf_days, 2.0)
    port.to_csv(out / "walk_forward_trades.csv", index=False)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), facecolor=rp.SURFACE,
                                 gridspec_kw={"height_ratios": [3, 2]}, sharex=True)
    rp._style(a1)
    for color, m in zip(rp.SERIES, (1.0, 2.0)):
        eq = C.CAPITAL + bt.daily_pnl(port, test_days, f"pnl_{m:g}x").cumsum()
        a1.plot(eq.index, eq.values, color=color, linewidth=2, label=f"{m:g}x cost")
    for _, te in windows:
        a1.axvline(te[0], color=rp.GRID, linewidth=0.8)
    a1.axhline(C.CAPITAL, color="#c3c2b7", linewidth=0.8)
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("Walk-forward: chained out-of-sample equity ($), re-tuned every 6 months", loc="left",
                 fontsize=10, color=rp.INK)
    rp._style(a2)
    if "z_entry" in chosen:
        for col, color in zip(["z_entry", "z_exit"], rp.SERIES):
            a2.step(pd.to_datetime(chosen["test_start"]), chosen[col], where="post", color=color, linewidth=2, label=col)
        a2b = a2.twinx()
        a2b.step(pd.to_datetime(chosen["test_start"]), chosen["lookback_bars"], where="post", color="#1baf7a",
                 linewidth=1.5, linestyle="--", label="L (right)")
        a2b.step(pd.to_datetime(chosen["test_start"]), chosen["hold_bars"], where="post", color="#6250d6",
                 linewidth=1.5, linestyle=":", label="H (right)")
        a2b.tick_params(colors=rp.MUTED, labelsize=8)
        a2.legend(frameon=False, fontsize=7, loc="upper left")
        a2b.legend(frameon=False, fontsize=7, loc="upper right")
    a2.set_title("Parameters chosen per window (z thresholds left; L and H in bars right)", loc="left",
                 fontsize=9, color=rp.INK)
    fig.tight_layout()
    fig.savefig(out / "walk_forward.png", dpi=130, facecolor=rp.SURFACE)
    plt.close(fig)

    n_distinct = int(chosen[list(DEFAULT.as_dict())].drop_duplicates().shape[0]) if "z_entry" in chosen else 0
    return {"windows": len(windows), "oos_1x": m1, "oos_2x": m2, "oos_2x_in_wf_oos_period": m2_h,
            "distinct_param_sets": n_distinct, "chosen": chosen.to_dict("records"),
            "positive_at_2x": bool(m2["mean_net_bps"] > 0 and m2["total_pnl"] > 0)}


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------
def _grid_cell(task):
    params, period = task
    sh = cm.SHARED
    port = bt.run_strategy(sh["panel"], sh["feats"], params, sh["panel"].days.isin(sh["periods"][period]))
    return params, summarize(port, sh["periods"][period])


def isolated_peak(grid: np.ndarray) -> dict:
    """Peak is 'isolated' if its 4-neighbours average less than half of it."""
    g = np.where(np.isfinite(grid), grid, -np.inf)
    y, x = np.unravel_index(np.argmax(g), g.shape)
    peak = grid[y, x]
    nb = [grid[y + dy, x + dx] for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
          if 0 <= y + dy < grid.shape[0] and 0 <= x + dx < grid.shape[1]]
    nb = np.array([v for v in nb if np.isfinite(v)])
    nmean = float(nb.mean()) if len(nb) else np.nan
    return {"peak": float(peak), "neighbour_mean": nmean, "at": (int(y), int(x)),
            "isolated": bool(peak > 0 and (not np.isfinite(nmean) or nmean < 0.5 * peak)),
            "share_positive": float(np.mean(grid[np.isfinite(grid)] > 0))}


def heatmaps(ctx: Ctx, out) -> dict:
    period = "dev"
    ga, gb = C.HEATMAP_GRID_A, C.HEATMAP_GRID_B
    tasks = [(bt.Params(L, ze, DEFAULT.z_exit, DEFAULT.hold_bars), period)
             for L in ga["lookback_bars"] for ze in ga["z_entry"]]
    tasks += [(bt.Params(DEFAULT.lookback_bars, DEFAULT.z_entry, zx, H), period)
              for zx in gb["z_exit"] for H in gb["hold_bars"]]
    cm.SHARED.update(panel=ctx.panel, feats=ctx.feats, periods=ctx.periods)
    res = cm.pmap(_grid_cell, tasks, ctx.workers, "heatmap cells", chunksize=4)
    cm.SHARED.clear()
    rows = [{**p.as_dict(), **m} for p, m in res]
    trial_log.log(rows, ctx.dataset, "heatmap", period)
    df = pd.DataFrame(rows)
    df.to_csv(out / "heatmap_cells.csv", index=False)

    flags = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=rp.SURFACE)
    specs = [("A", "lookback_bars", "z_entry", (df.z_exit == DEFAULT.z_exit) & (df.hold_bars == DEFAULT.hold_bars)),
             ("B", "z_exit", "hold_bars", (df.lookback_bars == DEFAULT.lookback_bars) & (df.z_entry == DEFAULT.z_entry))]
    for r, (name, ycol, xcol, sel) in enumerate(specs):
        sub = df[sel].drop_duplicates([ycol, xcol])
        for c, metric in enumerate(("mean_net_bps", "sharpe")):
            grid = sub.pivot(index=ycol, columns=xcol, values=metric)
            flags[f"{name}_{metric}"] = isolated_peak(grid.to_numpy(float))
            ax = axes[r, c]
            vals = grid.to_numpy(float)
            lim = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
            im = ax.imshow(vals, cmap=rp.DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
            ax.set_xticks(range(grid.shape[1]), [f"{v:g}" for v in grid.columns], fontsize=7, color=rp.MUTED)
            ax.set_yticks(range(grid.shape[0]), [f"{v:g}" for v in grid.index], fontsize=7, color=rp.MUTED)
            ax.set_xlabel(xcol, fontsize=8, color=rp.MUTED)
            ax.set_ylabel(ycol, fontsize=8, color=rp.MUTED)
            for (y, x), v in np.ndenumerate(vals):
                ax.text(x, y, "-" if not np.isfinite(v) else f"{v:.1f}", ha="center", va="center", fontsize=7, color=rp.INK)
            f = flags[f"{name}_{metric}"]
            ax.set_title(f"{metric} ({period})" + ("  ISOLATED PEAK" if f["isolated"] else ""), loc="left",
                         fontsize=9, color=rp.INK)
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out / "heatmaps.png", dpi=120, facecolor=rp.SURFACE)
    plt.close(fig)
    return {"flags": flags, "any_isolated": any(f["isolated"] for f in flags.values()),
            "min_share_positive": min(f["share_positive"] for f in flags.values())}


# ---------------------------------------------------------------------------
# Jitter
# ---------------------------------------------------------------------------
def _jitter_chunk(draws):
    sh = cm.SHARED
    out = []
    for p in draws:
        port = bt.run_strategy(sh["panel"], sh["feats"], p, sh["mask"])
        out.append({**p.as_dict(), **{f"{per}_{k}": v for per in ("dev", "wf_oos")
                                      for k, v in summarize(port, sh["periods"][per]).items()}})
    return out


def jitter(ctx: Ctx, out) -> dict:
    n = ctx.opt("jitter", C.JITTER_N)
    rng = ctx.rng("jitter")
    j = C.JITTER_FRAC
    u = lambda v: v * rng.uniform(1 - j, 1 + j, n)  # noqa: E731
    draws = [bt.Params(max(2, int(round(L))), float(ze), float(zx), max(1, int(round(H))))
             for L, ze, zx, H in zip(u(DEFAULT.lookback_bars), u(DEFAULT.z_entry), u(DEFAULT.z_exit), u(DEFAULT.hold_bars))]
    draws.sort(key=lambda p: p.lookback_bars)        # chunks share L -> z is computed once per chunk
    chunks = [draws[s:s + 20] for s in range(0, n, 20)]
    cm.SHARED.update(panel=ctx.panel, feats=ctx.feats, periods=ctx.periods, mask=ctx.mask("all"))
    recs = [r for c in cm.pmap(_jitter_chunk, chunks, ctx.workers, "jitter runs") for r in c]
    cm.SHARED.clear()
    df = pd.DataFrame(recs)
    df.to_csv(out / "jitter_runs.csv", index=False)
    for per in ("dev", "wf_oos"):
        trial_log.log([{**{k: r[k] for k in DEFAULT.as_dict()}, "n_trades": r[f"{per}_n_trades"],
                        "mean_net_bps": r[f"{per}_mean_net_bps"], "sharpe": r[f"{per}_sharpe"]} for r in recs],
                      ctx.dataset, "jitter", per)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=rp.SURFACE)
    res = {"n": n}
    for ax, per in zip(axes, ("dev", "wf_oos")):
        rp._style(ax)
        v = df[f"{per}_mean_net_bps"]
        ax.hist(v.dropna(), bins=40, color=rp.SERIES[0])
        ax.axvline(0, color=rp.MUTED, linewidth=1)
        res[per] = {"pct_profitable": float((v > 0).mean()), "median_net_bps": float(v.median()),
                    "p5_net_bps": float(v.quantile(0.05)), "median_sharpe": float(df[f"{per}_sharpe"].median())}
        ax.set_title(f"{per}: mean net bps over {n} jittered parameter sets\n{res[per]['pct_profitable']:.0%} profitable",
                     loc="left", fontsize=9, color=rp.INK)
    fig.tight_layout()
    fig.savefig(out / "jitter.png", dpi=130, facecolor=rp.SURFACE)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# Breakdowns, regimes, concentration
# ---------------------------------------------------------------------------
def vol_regimes(ctx: Ctx) -> pd.DataFrame:
    """Per day: VIX-proxy regime and realized-vol regime, each from information before that day."""
    days = ctx.panel.days
    out = pd.DataFrame(index=days)
    # Realized volatility of the sector ETFs (mean over ETFs of the day's 5-min return std).
    etf_rows = np.unique(ctx.panel.etf_idx)
    rv = pd.Series(np.nanmean(np.nanstd(ctx.feats.ret[etf_rows], axis=-1), axis=0), index=days)
    prev = rv.shift(1)
    out["realized_vol"] = np.where(prev > prev.rolling(C.VIX_REGIME_LOOKBACK_DAYS, min_periods=60).median(),
                                   "high", "low")
    out.loc[prev.rolling(C.VIX_REGIME_LOOKBACK_DAYS, min_periods=60).median().isna(), "realized_vol"] = "n/a"
    out["vix_proxy"] = "n/a"
    if not ctx.synthetic:
        import fetch_data
        try:
            v = fetch_data.load_bars([C.VIX_PROXY])
            local = v["timestamp"].dt.tz_convert(C.TIMEZONE)
            close = v.groupby(local.dt.tz_localize(None).dt.normalize())["close"].last().reindex(days)
            pc = close.shift(1)
            med = pc.rolling(C.VIX_REGIME_LOOKBACK_DAYS, min_periods=60).median()
            out["vix_proxy"] = np.where(pc > med, "high", "low")
            out.loc[med.isna() | pc.isna(), "vix_proxy"] = "n/a"
        except SystemExit:
            pass
    return out


def breakdown_tables(ctx: Ctx, port: pd.DataFrame, regimes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = pd.to_datetime(port["date"])
    keys = {
        "year": d.dt.year, "sector": port["etf"], "stock": port["symbol"],
        "entry_hour": pd.to_datetime(port["entry_time"]).dt.hour,
        "long_short": np.where(port["direction"] > 0, "long", "short"),
        "abs_z_bucket": bt.z_bucket(port["z_signal"]),
        "vix_proxy_regime": regimes["vix_proxy"].reindex(d).to_numpy(),
        "realized_vol_regime": regimes["realized_vol"].reindex(d).to_numpy(),
    }
    return {k: bt.breakdown(port, pd.Series(v, index=port.index)) for k, v in keys.items()}


def _leave_out(task):
    kind, name = task
    sh = cm.SHARED
    panel, feats, cands = sh["panel"], sh["feats"], sh["cands"]
    if kind == "stock":
        keep = cands["i"] != panel.stocks.index(name)
    else:
        keep = ~np.isin(cands["i"], [i for i, s in enumerate(panel.stocks) if C.SECTOR_MAP.get(s) == name])
    port = bt.run_strategy(panel, feats, cands=cands[keep])
    return kind, name, {per: summarize(port, sh["periods"][per]) for per in ("dev", "wf_oos")}


def concentration(ctx: Ctx, port: pd.DataFrame) -> dict:
    res = {}
    for per in ("dev", ctx.headline):
        days = ctx.periods[per]
        pt = port[port["date"].isin(days)]
        cut = pt["net_bps"].quantile(1 - C.TOP_TRADE_FRAC)
        res[f"{per}_without_top1pct"] = summarize(pt[pt["net_bps"] < cut], days)
        res[f"{per}_base"] = summarize(pt, days)
        pnl = pt.groupby(pd.to_datetime(pt["date"]).dt.year)["pnl_1x"].sum()
        res[f"{per}_leave_one_year_out"] = {
            int(y): summarize(pt[pd.to_datetime(pt["date"]).dt.year != y], days[days.year != y]) for y in pnl.index}
    cands = bt.candidates_for(ctx.feats, DEFAULT, ctx.mask("all"))
    cm.SHARED.update(panel=ctx.panel, feats=ctx.feats, cands=cands, periods=ctx.periods)
    tasks = [("stock", s) for s in ctx.panel.stocks] + [("sector", e) for e in sorted(set(C.SECTOR_MAP.values()))]
    loo = cm.pmap(_leave_out, tasks, ctx.workers, "leave-one-out runs", chunksize=4)
    cm.SHARED.clear()
    res["leave_one_out"] = {f"{k}:{n}": m for k, n, m in loo}
    return res


def out_of_universe(ctx: Ctx) -> dict:
    if ctx.synthetic:
        panel, feats, _, _ = cm.load_synthetic(ctx.holdout_unlocked, C.OUT_OF_UNIVERSE_MAP, seed=C.SYNTH_SEED + 100)
        periods = cm.make_periods(panel.days, feats.tradable.any(axis=0), True, ctx.holdout_unlocked)
    else:
        oou = [s for s in C.OUT_OF_UNIVERSE_MAP if s not in C.SECTOR_MAP]
        panel, feats, _ = cm.load_real(oou, C.OUT_OF_UNIVERSE_MAP, ctx.holdout_unlocked, "oou")
        periods = cm.make_periods(panel.days, feats.tradable.any(axis=0), False, ctx.holdout_unlocked)
    port = bt.run_strategy(panel, feats)
    res = {per: summarize(port, periods[per]) for per in ("dev", "wf_oos")}
    res["n_stocks"] = panel.n_stocks
    trial_log.log([{**DEFAULT.as_dict(), **res[p], "extra": {"universe": "out_of_universe"}} for p in ("dev", "wf_oos")],
                  ctx.dataset, "out_of_universe", "dev+wf_oos")
    return res


# ---------------------------------------------------------------------------
def run(ctx: Ctx) -> dict:
    out = ctx.out("robustness")
    print("walk-forward ...", flush=True)
    wf = walk_forward(ctx, out)
    print("heatmaps ...", flush=True)
    hm = heatmaps(ctx, out)
    print("jitter ...", flush=True)
    jt = jitter(ctx, out)
    print("breakdowns + concentration ...", flush=True)
    port = bt.run_strategy(ctx.panel, ctx.feats, day_mask=ctx.mask("all"))
    regimes = vol_regimes(ctx)
    bd_lines = []
    for per in ("dev", ctx.headline):
        pt = port[port["date"].isin(ctx.periods[per])]
        for name, tbl in breakdown_tables(ctx, pt, regimes).items():
            tbl.to_csv(out / f"breakdown_{per}_{name}.csv")
            if name != "stock":
                bd_lines += ["", f"-- {per} by {name} --", fmt_table(tbl)]
    conc = concentration(ctx, port)
    loo = pd.DataFrame({k: {f"{per}_{m}": v[per][m] for per in ("dev", "wf_oos") for m in ("mean_net_bps", "sharpe")}
                        for k, v in conc["leave_one_out"].items()}).T
    loo.to_csv(out / "leave_one_out.csv")
    print("out-of-universe ...", flush=True)
    oou = out_of_universe(ctx)

    h = ctx.headline
    base = conc[f"{h}_base"]
    pt_h = port[port["date"].isin(ctx.periods[h])]
    share = lambda col: float(pt_h.groupby(col)["pnl_1x"].sum().max() / pt_h["pnl_1x"].sum()) if len(pt_h) and pt_h["pnl_1x"].sum() > 0 else float("nan")  # noqa: E731
    years = pd.to_datetime(pt_h["date"]).dt.year
    conc_flags = {
        "without_top1pct_positive": bool(conc[f"{h}_without_top1pct"]["mean_net_bps"] > 0),
        "all_leave_one_stock_out_positive": bool((loo.loc[loo.index.str.startswith("stock:"), f"{h}_mean_net_bps"] > 0).all()),
        "all_leave_one_sector_out_positive": bool((loo.loc[loo.index.str.startswith("sector:"), f"{h}_mean_net_bps"] > 0).all()),
        "all_leave_one_year_out_positive": bool(all(v["mean_net_bps"] > 0 for v in conc[f"{h}_leave_one_year_out"].values())),
        "max_pnl_share_one_stock": share("symbol"), "max_pnl_share_one_sector": share("etf"),
        "max_pnl_share_one_year": float(pt_h.groupby(years)["pnl_1x"].sum().max() / pt_h["pnl_1x"].sum())
        if len(pt_h) and pt_h["pnl_1x"].sum() > 0 else float("nan"),
    }
    conc_flags["pass"] = bool(all(conc_flags[k] for k in conc_flags if k.startswith(("without", "all_"))))
    # Under a profitable strategy, a single year is the most a single name/sector/year should carry.

    res = {"dataset": ctx.dataset, "walk_forward": wf, "heatmaps": hm, "jitter": jt,
           "concentration": {**conc_flags, "base": base, "without_top1pct": conc[f"{h}_without_top1pct"]},
           "out_of_universe": oou}
    lo_loo = loo.sort_values(f"{h}_mean_net_bps").head(5)
    lines = [
        f"data: {ctx.dataset}; headline period {h}",
        "", f"WALK-FORWARD ({wf['windows']} windows: {C.WF_TRAIN_DAYS}-day train, {C.WF_TEST_DAYS}-day test)",
        f"  chained OOS at 1x cost: n={wf['oos_1x']['n_trades']} net {wf['oos_1x']['mean_net_bps']:+.2f} bps "
        f"Sharpe {wf['oos_1x']['sharpe']:.2f} ann.ret {wf['oos_1x']['ann_return']:.1%} maxDD {wf['oos_1x']['max_dd_pct']:.1%}",
        f"  chained OOS at 2x cost: net {wf['oos_2x']['mean_net_bps']:+.2f} bps Sharpe {wf['oos_2x']['sharpe']:.2f} "
        f"-> {'PASS' if wf['positive_at_2x'] else 'FAIL'} (positive at 2x)",
        f"  parameter stability: {wf['distinct_param_sets']} distinct parameter sets chosen in {wf['windows']} windows",
        fmt_table(pd.DataFrame(wf["chosen"]).drop(columns=["train_start", "train_end"], errors="ignore")),
        "", "HEATMAPS (development period; others at default):",
        *[f"  {k}: peak {v['peak']:.2f} at {v['at']}, neighbour mean {v['neighbour_mean']:.2f}, "
          f"{v['share_positive']:.0%} of cells > 0{'  ISOLATED PEAK' if v['isolated'] else ''}" for k, v in hm["flags"].items()],
        "", f"JITTER ({jt['n']} draws within +-{C.JITTER_FRAC:.0%}):",
        *[f"  {per}: {jt[per]['pct_profitable']:.0%} profitable, median net {jt[per]['median_net_bps']:+.2f} bps, "
          f"5th pct {jt[per]['p5_net_bps']:+.2f}, median Sharpe {jt[per]['median_sharpe']:.2f}" for per in ("dev", "wf_oos")],
        "", f"CONCENTRATION ({h}): base net {base['mean_net_bps']:+.2f} bps; without top 1% trades "
        f"{conc[f'{h}_without_top1pct']['mean_net_bps']:+.2f} bps (Sharpe {conc[f'{h}_without_top1pct']['sharpe']:.2f})",
        f"  largest P&L share: one stock {conc_flags['max_pnl_share_one_stock']:.0%}, one sector "
        f"{conc_flags['max_pnl_share_one_sector']:.0%}, one year {conc_flags['max_pnl_share_one_year']:.0%}",
        f"  leave-one-out all positive: stocks {conc_flags['all_leave_one_stock_out_positive']}, sectors "
        f"{conc_flags['all_leave_one_sector_out_positive']}, years {conc_flags['all_leave_one_year_out_positive']}",
        "  weakest leave-one-out runs:", fmt_table(lo_loo),
        "", f"OUT-OF-UNIVERSE ({oou['n_stocks']} stocks, default params): "
        + "; ".join(f"{p}: n={oou[p]['n_trades']} net {oou[p]['mean_net_bps']:+.2f} bps Sharpe {oou[p]['sharpe']:.2f}"
                    for p in ("dev", "wf_oos")),
        "", "BREAKDOWNS (stock-level tables in breakdown_*_stock.csv)", *bd_lines,
    ]
    write_summary(out, "ROBUSTNESS", lines, res)
    return res
