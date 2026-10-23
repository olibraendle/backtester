"""
`stats`: is the edge statistically real?
  * no_news vs news groups (signal study): n, mean gross/net bps, t-stat, win rate
  * day-block bootstrap: 95% CI of mean net bps and annualized Sharpe
  * random-entry baseline: same stocks, trade count, direction mix and holding times,
    random entry times in the 10:00-15:00 window
  * placebo news filter: news flags reshuffled among signals (same news share)
  * Deflated Sharpe Ratio (Bailey & Lopez de Prado), trials from results/trial_log.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

import backtest as bt
import config as C
import report as rp
import signals as sg
from validation import common as cm
from validation import trial_log
from validation.common import Ctx, fmt_table, summarize, write_summary


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def block_bootstrap(port: pd.DataFrame, days: pd.DatetimeIndex, rng, reps: int, block: int) -> dict:
    pt = port[port["date"].isin(days)]
    per_day = pt.groupby("date").agg(net=("net_bps", "sum"), n=("net_bps", "size"), pnl=("pnl_1x", "sum"))
    per_day = per_day.reindex(days, fill_value=0.0)
    net, cnt, ret = (per_day[c].to_numpy(float) for c in ("net", "n", "pnl"))
    ret = ret / C.CAPITAL
    T = len(days)
    means, sharpes = [], []
    for start in range(0, reps, 1000):
        idx = cm.block_bootstrap_indices(rng, T, T, block, min(1000, reps - start))
        with np.errstate(invalid="ignore", divide="ignore"):
            means.append(net[idx].sum(1) / cnt[idx].sum(1))
            r = ret[idx]
            sharpes.append(r.mean(1) / r.std(1, ddof=1) * np.sqrt(252))
    means, sharpes = np.concatenate(means), np.concatenate(sharpes)
    q = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]  # noqa: E731
    return {"mean_net_bps_ci95": q(means), "sharpe_ci95": q(sharpes),
            "p_mean_net_le_0": float(np.nanmean(means <= 0)), "_means": means, "_sharpes": sharpes}


# ---------------------------------------------------------------------------
# Random-entry baseline
# ---------------------------------------------------------------------------
def _random_entry_runs(task) -> list[tuple[float, float]]:
    run_ids, period = task
    sh = cm.SHARED
    panel, feats, real, days, seed = sh["panel"], sh["feats"], sh["real"], sh["days"], sh["seed"]
    lo, hi = sg.entry_slot_range()
    dmask = panel.days.isin(days)
    out = []
    for r in run_ids:
        rng = cm.rng_for(seed, "random_entry", period, r)
        i = real["i"].to_numpy()
        # random day among the period's days on which that stock is tradable
        d = np.empty(len(i), dtype=int)
        for sym in np.unique(i):
            ok = np.nonzero(feats.tradable[sym] & dmask)[0]
            k = i == sym
            d[k] = rng.choice(ok, size=k.sum())
        cands = pd.DataFrame({"i": i, "d": d, "t": rng.integers(lo, hi + 1, size=len(i)),
                              "z": -real["direction"].to_numpy().astype(float), "has_news": False})
        hold = rng.permutation(real["bars_held"].to_numpy())
        tr = bt.simulate_trades(panel, feats, cands, exit_after_bars=hold)
        tr = bt.size_trades(tr, feats)
        m = summarize(tr, days)
        out.append((m["mean_net_bps"], m["sharpe"]))
    return out


# ---------------------------------------------------------------------------
# Placebo news filter
# ---------------------------------------------------------------------------
def _placebo_runs(task) -> list[tuple[float, float]]:
    run_ids, period = task
    sh = cm.SHARED
    study, feats, days, seed = sh["study"], sh["feats"], sh["days"], sh["seed"]
    flags = study["has_news"].to_numpy()
    out = []
    for r in run_ids:
        rng = cm.rng_for(seed, "placebo", period, r)
        fake = rng.permutation(flags)
        port = bt.size_trades(bt.select_portfolio(study[~fake]), feats)
        m = summarize(port, days)
        out.append((m["mean_net_bps"], m["sharpe"]))
    return out


def _chunks(n: int, size: int) -> list[list[int]]:
    return [list(range(s, min(s + size, n))) for s in range(0, n, size)]


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------
def deflated_sharpe(daily_ret: np.ndarray, n_trials: int, sharpe_var_annual: float) -> dict:
    """
    Bailey & Lopez de Prado (2014). SR, its variance across trials and the benchmark SR0 are
    in per-day units; DSR = P(true SR > SR0) given T days, skewness and kurtosis.
    """
    r = np.asarray(daily_ret, float)
    T = len(r)
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else np.nan
    skew, kurt = float(sps.skew(r)), float(sps.kurtosis(r, fisher=False))
    if n_trials >= 2 and np.isfinite(sharpe_var_annual):
        g = np.euler_gamma
        v = sharpe_var_annual / 252.0
        sr0 = np.sqrt(v) * ((1 - g) * sps.norm.ppf(1 - 1 / n_trials) + g * sps.norm.ppf(1 - 1 / (n_trials * np.e)))
    else:
        sr0 = 0.0
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    dsr = float(sps.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))
    return {"sharpe_annual": float(sr * np.sqrt(252)), "sr0_annual": float(sr0 * np.sqrt(252)),
            "n_trials": n_trials, "trial_sharpe_var_annual": sharpe_var_annual, "T_days": T,
            "skew": skew, "kurtosis": kurt, "dsr": dsr, "pass": bool(dsr > 0.95)}


# ---------------------------------------------------------------------------
def _hist(ax, dist, value, title, xlabel):
    rp._style(ax)
    ax.hist(dist[np.isfinite(dist)], bins=50, color=rp.SERIES[0], alpha=0.85)
    ax.axvline(value, color=rp.SERIES[1], linewidth=2)
    ax.annotate(f"strategy {value:.2f}", (value, ax.get_ylim()[1] * 0.92), xytext=(4, 0),
                textcoords="offset points", color=rp.INK_2, fontsize=8)
    ax.set_title(title, loc="left", fontsize=9, color=rp.INK)
    ax.set_xlabel(xlabel, fontsize=8, color=rp.MUTED)


def run(ctx: Ctx) -> dict:
    out = ctx.out("stats")
    reps = ctx.opt("bootstrap", C.BOOTSTRAP_N)
    res, lines = {"dataset": ctx.dataset}, [f"data: {ctx.dataset}; headline period: {ctx.headline}"]

    study_all = bt.simulate_trades(ctx.panel, ctx.feats, sg.find_candidates(ctx.feats, C.Z_ENTRY, news="any"))
    port_all = bt.run_strategy(ctx.panel, ctx.feats)
    for period in ("dev", ctx.headline):
        days = ctx.periods[period]
        st = study_all[study_all["date"].isin(days)]
        port = port_all[port_all["date"].isin(days)]
        groups = {}
        for view, frame in (("every_bar", st), ("first_per_stock_day", bt.first_per_stock_day(st))):
            for g, flag in (("no_news", False), ("news", True)):
                groups[(view, g)] = bt.trade_stats(frame[frame["has_news"] == flag])
        groups[("portfolio", "no_news")] = bt.trade_stats(port)
        gdf = pd.DataFrame(groups).T
        gdf.to_csv(out / f"groups_{period}.csv")
        summ = summarize(port, days)
        trial_log.log([{**bt.Params().as_dict(), **summ}], ctx.dataset, "stats", period)

        print(f"[{period}] bootstrap ({reps}) ...", flush=True)
        boot = block_bootstrap(port, days, ctx.rng("bootstrap", period), reps, C.BLOCK_LEN_DAYS)

        print(f"[{period}] random-entry baseline ({C.RANDOM_ENTRY_N}) ...", flush=True)
        cm.SHARED.update(panel=ctx.panel, feats=ctx.feats, real=port, days=days, seed=ctx.seed)
        rand = np.array([x for chunk in cm.pmap(_random_entry_runs, [(c, period) for c in _chunks(C.RANDOM_ENTRY_N, 50)],
                                                 ctx.workers, "random entry") for x in chunk])
        print(f"[{period}] placebo news ({C.PLACEBO_N}) ...", flush=True)
        cm.SHARED.update(study=study_all[study_all["date"].isin(days)])
        plac = np.array([x for chunk in cm.pmap(_placebo_runs, [(c, period) for c in _chunks(C.PLACEBO_N, 25)],
                                                 ctx.workers, "placebo") for x in chunk])
        cm.SHARED.clear()

        pr = {"random_entry_pct_net": cm.percentile_of(summ["mean_net_bps"], rand[:, 0]),
              "random_entry_pct_sharpe": cm.percentile_of(summ["sharpe"], rand[:, 1]),
              "random_entry_mean_net": float(np.nanmean(rand[:, 0])),
              "placebo_pct_net": cm.percentile_of(summ["mean_net_bps"], plac[:, 0]),
              "placebo_pct_sharpe": cm.percentile_of(summ["sharpe"], plac[:, 1]),
              "placebo_mean_net": float(np.nanmean(plac[:, 0])),
              "placebo_p95_net": float(np.nanpercentile(plac[:, 0], 95))}
        pd.DataFrame(rand, columns=["mean_net_bps", "sharpe"]).to_csv(out / f"random_entry_{period}.csv", index=False)
        pd.DataFrame(plac, columns=["mean_net_bps", "sharpe"]).to_csv(out / f"placebo_{period}.csv", index=False)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=rp.SURFACE)
        _hist(axes[0, 0], boot["_means"], summ["mean_net_bps"], "Block bootstrap: mean net bps", "bps/trade")
        _hist(axes[0, 1], boot["_sharpes"], summ["sharpe"], "Block bootstrap: annualized Sharpe", "Sharpe")
        _hist(axes[1, 0], rand[:, 0], summ["mean_net_bps"], "Random-entry baseline: mean net bps", "bps/trade")
        _hist(axes[1, 1], plac[:, 0], summ["mean_net_bps"], "Placebo news filter: mean net bps", "bps/trade")
        fig.suptitle(f"{period}: strategy (orange) vs. resampled / null distributions", x=0.01, ha="left",
                     fontsize=11, color=rp.INK)
        fig.tight_layout()
        fig.savefig(out / f"distributions_{period}.png", dpi=120, facecolor=rp.SURFACE)
        plt.close(fig)

        boot_out = {k: v for k, v in boot.items() if not k.startswith("_")}
        res[period] = {"groups": {f"{a}/{b}": v for (a, b), v in groups.items()}, "portfolio": summ,
                       "bootstrap": boot_out, **pr}
        lines += ["", f"--- {period} ({len(days)} days) ---", fmt_table(gdf),
                  f"portfolio: n={summ['n_trades']} mean net {summ['mean_net_bps']:+.2f} bps, Sharpe {summ['sharpe']:.2f}",
                  f"block bootstrap ({reps}x, {C.BLOCK_LEN_DAYS}-day blocks): mean net 95% CI "
                  f"[{boot_out['mean_net_bps_ci95'][0]:+.2f}, {boot_out['mean_net_bps_ci95'][1]:+.2f}] bps; "
                  f"Sharpe 95% CI [{boot_out['sharpe_ci95'][0]:.2f}, {boot_out['sharpe_ci95'][1]:.2f}]",
                  f"random-entry baseline: strategy at the {pr['random_entry_pct_net']:.1f}th percentile (net bps), "
                  f"{pr['random_entry_pct_sharpe']:.1f}th (Sharpe); baseline mean {pr['random_entry_mean_net']:+.2f} bps",
                  f"placebo news filter: strategy at the {pr['placebo_pct_net']:.1f}th percentile (net bps), "
                  f"{pr['placebo_pct_sharpe']:.1f}th (Sharpe); placebo mean {pr['placebo_mean_net']:+.2f}, "
                  f"95th pct {pr['placebo_p95_net']:+.2f} bps"]

    # Deflated Sharpe on the headline period, after logging this run's trials
    port_h = port_all[port_all["date"].isin(ctx.periods[ctx.headline])]
    r_h = cm.daily_returns(port_h, ctx.periods[ctx.headline]).to_numpy()
    dsr = deflated_sharpe(r_h, *trial_log.trial_stats(ctx.dataset, selection_only=True))
    dsr_all = deflated_sharpe(r_h, *trial_log.trial_stats(ctx.dataset, selection_only=False))
    res["deflated_sharpe"] = dsr                      # pass/fail: selection trials
    res["deflated_sharpe_all_trials"] = dsr_all       # literal: every logged configuration
    for label, d in (("selection trials (pass/fail)", dsr), ("every logged config (reference)", dsr_all)):
        lines += [f"{'' if label.startswith('every') else chr(10)}Deflated Sharpe ({ctx.headline}, {label}): "
                  f"SR {d['sharpe_annual']:.2f} vs. expected max of {d['n_trials']} trials SR0 {d['sr0_annual']:.2f} "
                  f"(annualized); skew {d['skew']:.2f}, kurtosis {d['kurtosis']:.1f}, T={d['T_days']} -> "
                  f"DSR = {d['dsr']:.3f}" + (f" ({'PASS' if d['pass'] else 'FAIL'}: > 0.95)" if d is dsr else "")]
    write_summary(out, "STATS - is the edge statistically real?", lines, res)
    return res
