"""
`risk`: Monte Carlo drawdowns, position sizing and kill criteria.
  * day-block bootstrap of the out-of-sample daily P&L: RISK_PATHS one-year paths ->
    max drawdown, longest losing streak, final return, P(losing year); 5/50/95th percentiles
  * position size such that the 95th-percentile drawdown <= MAX_DRAWDOWN_LIMIT of capital
  * kill criteria for live trading:
      (a) live mean net bps < KILL_FRACTION x backtest after N trades, with N chosen so the rule
          fires with KILL_POWER probability if the edge is gone (true mean net = 0)
      (b) live drawdown beyond the 95th-percentile bootstrap drawdown
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

import backtest as bt
import config as C
import report as rp
from validation import common as cm
from validation.common import Ctx, summarize, write_summary


def path_stats(daily_ret: np.ndarray, rng, reps: int, length: int, block: int) -> dict[str, np.ndarray]:
    """Bootstrap paths of daily returns (fraction of capital): max DD, longest losing streak, final return."""
    out = {"max_dd": [], "streak": [], "final": [], "q": []}
    T = len(daily_ret)
    for start in range(0, reps, 2000):
        k = min(2000, reps - start)
        r = daily_ret[cm.block_bootstrap_indices(rng, T, length, block, k)]
        eq = np.cumsum(r, axis=1)
        peak = np.maximum.accumulate(np.maximum(eq, 0.0), axis=1)
        out["max_dd"].append((peak - eq).max(axis=1))
        losing = r < 0
        c = np.cumsum(losing, axis=1)
        streak = c - np.maximum.accumulate(np.where(~losing, c, 0), axis=1)
        out["streak"].append(streak.max(axis=1))
        out["final"].append(eq[:, -1])
        out["q"].append(np.percentile(eq, [5, 50, 95], axis=0))
    q = np.mean(out.pop("q"), axis=0)
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["fan"] = q
    return res


def kill_n(mean_bps: float, sd_eff_bps: float) -> int:
    """
    Trades needed so that, if the edge is gone (true mean 0), the live mean falls below
    KILL_FRACTION x backtest mean with probability KILL_POWER.
    """
    if not (mean_bps > 0 and sd_eff_bps > 0):
        return -1
    z = sps.norm.ppf(C.KILL_POWER)
    return int(math.ceil((z * sd_eff_bps / (C.KILL_FRACTION * mean_bps)) ** 2))


def effective_sd(trades: pd.DataFrame) -> float:
    """Per-trade sd inflated for within-day correlation: day-clustered SE x sqrt(n)."""
    st = bt.trade_stats(trades)
    if not st["n"] or not np.isfinite(st["t_stat_day_clustered"]) or st["t_stat_day_clustered"] == 0:
        return float("nan")
    return abs(st["mean_net_bps"] / st["t_stat_day_clustered"]) * math.sqrt(st["n"])


def monitor_kill(port: pd.DataFrame, bt_days: pd.DatetimeIndex, live_days: pd.DatetimeIndex, rng) -> dict:
    """Fit both kill rules on bt_days, then walk forward through live_days and report when they fire."""
    bt_tr = port[port["date"].isin(bt_days)]
    live = port[port["date"].isin(live_days)].sort_values("entry_time")
    mu = float(bt_tr["net_bps"].mean()) if len(bt_tr) else float("nan")
    N = kill_n(mu, effective_sd(bt_tr))
    rec = {"bt_mean_net_bps": mu, "N_kill": N, "killed_mean": False, "kill_trades": np.nan,
           "kill_days_mean": np.nan, "killed_dd": False, "kill_days_dd": np.nan}
    day_pos = {d: k for k, d in enumerate(live_days)}
    if N > 0 and len(live) >= N:
        run_mean = np.cumsum(live["net_bps"].to_numpy()) / np.arange(1, len(live) + 1)
        hit = np.nonzero((np.arange(1, len(live) + 1) >= N) & (run_mean < C.KILL_FRACTION * mu))[0]
        if len(hit):
            rec.update(killed_mean=True, kill_trades=int(hit[0] + 1),
                       kill_days_mean=day_pos[live["date"].iloc[hit[0]]] + 1)
    bt_ret = cm.daily_returns(port, bt_days).to_numpy()
    if len(bt_ret) > C.BLOCK_LEN_DAYS:
        dd95 = np.percentile(path_stats(bt_ret, rng, 1000, C.RISK_PATH_DAYS, C.BLOCK_LEN_DAYS)["max_dd"], 95)
        eq = np.cumsum(cm.daily_returns(port, live_days).to_numpy())
        dd = np.maximum.accumulate(np.maximum(eq, 0.0)) - eq
        hit = np.nonzero(dd > dd95)[0]
        rec["dd95"] = float(dd95)
        if len(hit):
            rec.update(killed_dd=True, kill_days_dd=int(hit[0] + 1))
    return rec


def run(ctx: Ctx) -> dict:
    out = ctx.out("risk")
    period = ctx.headline
    days = ctx.periods[period]
    port = bt.run_strategy(ctx.panel, ctx.feats, day_mask=ctx.mask(period))
    ret = cm.daily_returns(port, days).to_numpy()
    summ = summarize(port, days)
    ps = path_stats(ret, ctx.rng("risk"), C.RISK_PATHS, C.RISK_PATH_DAYS, C.BLOCK_LEN_DAYS)

    pct = lambda a: {f"p{q}": float(np.percentile(a, q)) for q in (5, 50, 95)}  # noqa: E731
    table = pd.DataFrame({"max_drawdown_pct": pct(100 * ps["max_dd"]), "longest_losing_streak_days": pct(ps["streak"]),
                          "final_return_pct": pct(100 * ps["final"])}).T
    p_losing_year = float((ps["final"] < 0).mean())
    dd95 = float(np.percentile(ps["max_dd"], 95))
    scale = C.MAX_DRAWDOWN_LIMIT / dd95 if dd95 > 0 else float("inf")
    uncapped_frac = C.FIXED_NOTIONAL_FRAC * scale
    # Gross exposure with every slot full: MAX_POSITIONS x notional x (1 + |beta|) for the hedge leg.
    lev_cap = C.MAX_GROSS_LEVERAGE / (C.MAX_POSITIONS * (1 + float(port["beta"].abs().mean() if len(port) else 1.0)))
    rec_frac = min(uncapped_frac, lev_cap)
    sd_eff = effective_sd(port)
    N = kill_n(summ["mean_net_bps"], sd_eff)
    res = {"dataset": ctx.dataset, "period": period, "portfolio": summ,
           "percentiles": table.to_dict("index"), "p_losing_year": p_losing_year,
           "dd95_at_current_size": dd95, "size_scale": scale,
           "recommended_notional_frac": rec_frac, "uncapped_notional_frac": uncapped_frac,
           "leverage_cap_frac": lev_cap,
           "recommended_notional_usd": rec_frac * C.CAPITAL,
           "kill": {"N_trades": N, "threshold_bps": C.KILL_FRACTION * summ["mean_net_bps"],
                    "backtest_mean_bps": summ["mean_net_bps"], "effective_sd_bps": sd_eff,
                    "trades_per_day": summ["n_trades"] / max(len(days), 1),
                    "dd_limit_pct": 100 * dd95 * min(scale, 1e9),
                    "false_kill_prob_if_edge_intact": float(sps.norm.cdf(
                        -(1 - C.KILL_FRACTION) * summ["mean_net_bps"] * math.sqrt(max(N, 1)) / sd_eff))
                    if N > 0 else None}}
    table.to_csv(out / "risk_percentiles.csv")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), facecolor=rp.SURFACE)
    for ax, key, title, mult in zip(axes[:3], ("max_dd", "streak", "final"),
                                    ("max drawdown (% capital)", "longest losing streak (days)", "1-year return (% capital)"),
                                    (100, 1, 100)):
        rp._style(ax)
        ax.hist(ps[key] * mult, bins=50, color=rp.SERIES[0])
        for q in (5, 50, 95):
            ax.axvline(np.percentile(ps[key] * mult, q), color=rp.MUTED, linewidth=0.8, linestyle="--")
        ax.set_title(title, loc="left", fontsize=9, color=rp.INK)
    ax = axes[3]
    rp._style(ax)
    x = np.arange(1, C.RISK_PATH_DAYS + 1)
    ax.fill_between(x, 100 * ps["fan"][0], 100 * ps["fan"][2], color=rp.SERIES[0], alpha=0.2, linewidth=0)
    ax.plot(x, 100 * ps["fan"][1], color=rp.SERIES[0], linewidth=2)
    ax.set_title("equity paths: median, 5-95% band (% capital)", loc="left", fontsize=9, color=rp.INK)
    fig.suptitle(f"{C.RISK_PATHS:,} block-bootstrap 1-year paths of {period} daily P&L at the current size",
                 x=0.01, ha="left", fontsize=10, color=rp.INK)
    fig.tight_layout()
    fig.savefig(out / "risk.png", dpi=130, facecolor=rp.SURFACE)
    plt.close(fig)

    k = res["kill"]
    lines = [f"data: {ctx.dataset}; {period} daily P&L ({len(days)} days, {summ['n_trades']} trades, "
             f"mean net {summ['mean_net_bps']:+.2f} bps, Sharpe {summ['sharpe']:.2f}) at {C.FIXED_NOTIONAL_FRAC:.0%} "
             f"of capital per stock leg",
             "", f"{C.RISK_PATHS:,} one-year paths, {C.BLOCK_LEN_DAYS}-day blocks:", cm.fmt_table(table),
             f"P(losing year) = {p_losing_year:.1%}",
             "", f"Position sizing: 95th-pct drawdown {100 * dd95:.2f}% at the current size -> scale x{scale:.2f} "
             f"to hit {C.MAX_DRAWDOWN_LIMIT:.0%} = {100 * uncapped_frac:.1f}% of capital per stock leg; capped by "
             f"{C.MAX_GROSS_LEVERAGE:g}x gross leverage at {100 * lev_cap:.1f}% -> recommend {100 * rec_frac:.1f}% "
             f"(${rec_frac * C.CAPITAL:,.0f}) per stock leg (linear scaling; the 1%-of-bar-volume cap still applies)",
             "", "Proposed kill criteria for live trading:",
             f"  (a) after N = {k['N_trades']} live trades (~{k['N_trades'] / max(k['trades_per_day'], 1e-9):.0f} trading days "
             f"at {k['trades_per_day']:.1f} trades/day), stop if the live mean net is below {k['threshold_bps']:+.2f} bps "
             f"(= {C.KILL_FRACTION:.0%} of the backtest {k['backtest_mean_bps']:+.2f}); {C.KILL_POWER:.0%} power if the edge is gone"
             + (f", {k['false_kill_prob_if_edge_intact']:.0%} chance of a false kill if it is intact"
                if k["false_kill_prob_if_edge_intact"] is not None else ""),
             f"  (b) stop if the live drawdown exceeds the 95th-percentile bootstrap drawdown: "
             f"{C.MAX_DRAWDOWN_LIMIT:.0%} of capital at the recommended size ({100 * dd95:.2f}% at the current size)"]
    write_summary(out, "RISK - Monte Carlo drawdowns, sizing and kill criteria", lines, res)
    return res
