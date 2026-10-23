"""
`execution`: realism stress.
  * costs at 1x/2x/3x and the break-even cost per leg per side
  * pessimistic fills (worse of next-bar open and next-bar (H+L+C)/3), plus random slippage
  * missed fills: skip 5/10/20% of trades, biased toward the best (fastest-reverting) trades
  * random 0-1 bar entry delay
  * short-borrow stress: 10% of short entries blocked
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


def with_extra_cost(port: pd.DataFrame, extra_bps: np.ndarray) -> pd.DataFrame:
    """Subtract extra per-trade bps (e.g. slippage) from gross and recompute net and $ P&L."""
    tr = port.copy()
    tr["gross_bps"] = tr["gross_bps"] - extra_bps
    tr["net_bps"] = tr["gross_bps"] - tr["cost_bps"]
    for m in C.EXEC_COST_MULTIPLIERS:
        tr[f"pnl_{m:g}x"] = tr["notional"] * (tr["gross_bps"] - m * tr["cost_bps"]) / 1e4
    return tr


def slippage_bps(port: pd.DataFrame, rng) -> np.ndarray:
    """Random adverse slippage on the 4 fills (stock in/out, hedge in/out weighted by |beta|)."""
    n = len(port)
    if C.SLIPPAGE_DIST == "none" or n == 0:
        return np.zeros(n)
    draw = (lambda k: np.abs(rng.normal(0, C.SLIPPAGE_SCALE_BPS, (k, 4)))) if C.SLIPPAGE_DIST == "halfnormal" \
        else (lambda k: rng.exponential(C.SLIPPAGE_SCALE_BPS, (k, 4)))
    s = draw(n)
    hedge = np.abs(port["beta"].to_numpy()) if C.USE_HEDGE else np.zeros(n)
    return s[:, 0] + s[:, 1] + hedge * (s[:, 2] + s[:, 3])


def missed_fills(port: pd.DataFrame, rate: float, rng) -> pd.DataFrame:
    """Drop ~rate of trades; skip probability grows with the trade's rank by net bps (best missed most)."""
    n = len(port)
    if n == 0:
        return port
    rank = port["net_bps"].rank(method="first").to_numpy()
    w = rank ** C.MISSED_FILL_BIAS
    p = np.minimum(1.0, rate * n * w / w.sum())
    return port[rng.random(n) >= p]


def _rerun(task):
    kind, rep, period = task
    sh = cm.SHARED
    rng = cm.rng_for(sh["seed"], "execution", kind, period, rep)
    cands = sh["cands"][period]
    if kind == "delay":
        port = bt.run_strategy(sh["panel"], sh["feats"], cands=cands, entry_delay=rng.integers(0, 2, len(cands)))
    else:   # short-borrow: block SHORT_BLOCK_RATE of short stock-days (shorts fade up-moves: z > 0)
        short = cands["z"].to_numpy() > 0
        keys = np.unique(cands.loc[short, ["i", "d"]].to_numpy(), axis=0)
        blocked = keys[rng.random(len(keys)) < C.SHORT_BLOCK_RATE]
        bkey = set(map(tuple, blocked))
        drop = short & np.array([(a, b) in bkey for a, b in zip(cands["i"], cands["d"])])
        port = bt.run_strategy(sh["panel"], sh["feats"], cands=cands[~drop])
    return kind, period, summarize(port, sh["periods"][period])


def _dist(vals: list[dict], key: str) -> dict:
    v = np.array([x[key] for x in vals], float)
    return {"mean": float(np.nanmean(v)), "p5": float(np.nanpercentile(v, 5)), "p95": float(np.nanpercentile(v, 95))}


def run(ctx: Ctx) -> dict:
    out = ctx.out("execution")
    reps = C.EXEC_REPS
    re_reps = max(reps // 4, 10)
    res, tables = {"dataset": ctx.dataset}, {}
    cands = {per: bt.candidates_for(ctx.feats, DEFAULT, ctx.mask(per)) for per in ("dev", ctx.headline)}

    cm.SHARED.update(panel=ctx.panel, feats=ctx.feats, cands=cands, periods=ctx.periods, seed=ctx.seed)
    tasks = [(k, r, per) for k in ("delay", "short_block") for per in cands for r in range(re_reps)]
    reruns = cm.pmap(_rerun, tasks, ctx.workers, "delay / short-borrow reruns")
    cm.SHARED.clear()

    for per in ("dev", ctx.headline):
        days = ctx.periods[per]
        base = bt.run_strategy(ctx.panel, ctx.feats, cands=cands[per])
        base = with_extra_cost(base, np.zeros(len(base)))           # adds the 3x P&L column
        rows = {}
        for m in C.EXEC_COST_MULTIPLIERS:
            s = summarize(base, days, m)
            rows[f"cost {m:g}x"] = {"mean_net_bps": s["mean_net_bps"], "sharpe": s["sharpe"], "n_trades": s["n_trades"]}
        be = float(base["gross_bps"].mean() / (2 * (1 + base["beta"].abs()).mean())) if len(base) else np.nan
        pess = bt.run_strategy(ctx.panel, ctx.feats, cands=cands[per], fill="pessimistic")
        pess = with_extra_cost(pess, np.zeros(len(pess)))
        s = summarize(pess, days)
        rows["pessimistic fills"] = {"mean_net_bps": s["mean_net_bps"], "sharpe": s["sharpe"], "n_trades": s["n_trades"]}
        slip = [summarize(with_extra_cost(pess, slippage_bps(pess, ctx.rng("slip", per, r))), days) for r in range(reps)]
        rows["pessimistic + slippage"] = {"mean_net_bps": _dist(slip, "mean_net_bps")["mean"],
                                          "sharpe": _dist(slip, "sharpe")["mean"], "n_trades": s["n_trades"],
                                          "net_p5": _dist(slip, "mean_net_bps")["p5"],
                                          "net_p95": _dist(slip, "mean_net_bps")["p95"]}
        for rate in C.MISSED_FILL_RATES:
            mf = [summarize(missed_fills(base, rate, ctx.rng("miss", per, rate, r)), days) for r in range(reps)]
            rows[f"missed fills {rate:.0%} (best first)"] = {
                "mean_net_bps": _dist(mf, "mean_net_bps")["mean"], "sharpe": _dist(mf, "sharpe")["mean"],
                "n_trades": _dist(mf, "n_trades")["mean"], "net_p5": _dist(mf, "mean_net_bps")["p5"],
                "net_p95": _dist(mf, "mean_net_bps")["p95"]}
        for kind, label in (("delay", "random 0-1 bar delay"), ("short_block", f"{C.SHORT_BLOCK_RATE:.0%} shorts blocked")):
            vals = [m for k, p, m in reruns if k == kind and p == per]
            rows[label] = {"mean_net_bps": _dist(vals, "mean_net_bps")["mean"], "sharpe": _dist(vals, "sharpe")["mean"],
                           "n_trades": _dist(vals, "n_trades")["mean"], "net_p5": _dist(vals, "mean_net_bps")["p5"],
                           "net_p95": _dist(vals, "mean_net_bps")["p95"]}
        tbl = pd.DataFrame(rows).T
        tbl.to_csv(out / f"execution_{per}.csv")
        tables[per] = tbl
        trial_log.log([{**DEFAULT.as_dict(), **r, "extra": {"stress": k}} for k, r in rows.items()],
                      ctx.dataset, "execution", per)
        res[per] = {"table": tbl.to_dict("index"), "break_even_cost_bps_per_side": be,
                    "worst_case_net_bps": float(tbl["mean_net_bps"].min())}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=rp.SURFACE, sharey=True)
    for ax, per in zip(axes, ("dev", ctx.headline)):
        rp._style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=rp.GRID, linewidth=0.6)
        tbl = tables[per]
        y = np.arange(len(tbl))[::-1]
        ax.barh(y, tbl["mean_net_bps"], color=[rp.SERIES[0] if v > 0 else "#e34948" for v in tbl["mean_net_bps"]],
                height=0.6)
        if "net_p5" in tbl:
            lo, hi = tbl["net_p5"].to_numpy(float), tbl["net_p95"].to_numpy(float)
            ok = np.isfinite(lo)
            ax.errorbar(tbl["mean_net_bps"][ok], y[ok], xerr=[tbl["mean_net_bps"][ok] - lo[ok], hi[ok] - tbl["mean_net_bps"][ok]],
                        fmt="none", ecolor=rp.INK_2, elinewidth=1, capsize=2)
        ax.set_yticks(y, tbl.index, fontsize=8, color=rp.INK_2)
        ax.axvline(0, color=rp.MUTED, linewidth=1)
        ax.set_title(f"{per}: mean net bps per trade (5-95% band over {reps} reps where random)\n"
                     f"break-even cost {res[per]['break_even_cost_bps_per_side']:.2f} bps per leg per side",
                     loc="left", fontsize=9, color=rp.INK)
    fig.tight_layout()
    fig.savefig(out / "execution.png", dpi=130, facecolor=rp.SURFACE)
    plt.close(fig)

    lines = [f"data: {ctx.dataset}; default parameters; slippage {C.SLIPPAGE_DIST} scale {C.SLIPPAGE_SCALE_BPS} bps per fill"]
    for per in ("dev", ctx.headline):
        lines += ["", f"--- {per} --- break-even cost: {res[per]['break_even_cost_bps_per_side']:.2f} bps per leg per side "
                  f"(assumed {C.COST_BPS_PER_SIDE})", fmt_table(tables[per])]
    h = res[ctx.headline]
    res["pass"] = bool(h["table"]["cost 2x"]["mean_net_bps"] > 0 and h["table"]["pessimistic + slippage"]["mean_net_bps"] > 0)
    lines.append(f"\n{'PASS' if res['pass'] else 'FAIL'} ({ctx.headline}): positive at 2x cost and with pessimistic fills + slippage")
    write_summary(out, "EXECUTION - realism stress", lines, res)
    return res
