"""
Backtest the news-filtered intraday reversal strategy.

    python backtest.py                 # cached Alpaca data (run fetch_data.py first)
    python backtest.py --synthetic     # fake data with a planted effect, no API keys needed
    python backtest.py --no-sweep      # skip the in-sample parameter sweep

Fill conventions (see signals.py for the feature timing):
  * A signal is evaluated at the CLOSE of bar t; the order fills at the OPEN of
    the first bar after t (normally t+1). The signal bar's prices are never used.
  * Exits are checked at each bar's close starting with the entry bar; exits 1-4
    fill at the next bar's open, the end-of-day exit at the 15:55 bar's close.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

import config as C
import signals as sg

S = C.BARS_PER_DAY
REASON_ORDER = ["reversion", "time", "stop", "news", "eod"]


# ---------------------------------------------------------------------------
# Trade simulation (one trade per candidate signal, independent of the portfolio)
# ---------------------------------------------------------------------------
def simulate_trades(panel: sg.Panel, feats: sg.Features, cands: pd.DataFrame, *,
                    z_entry: float = C.Z_ENTRY, z_exit: float = C.Z_EXIT, hold_bars: int = C.HOLD_BARS,
                    hedge: bool = C.USE_HEDGE, stop_reference: str = C.STOP_Z_REFERENCE,
                    fill: str = "open", entry_delay=0, exit_after_bars=None) -> pd.DataFrame:
    """
    Simulate the fade trade for every candidate (i, d, t, z). Vectorised over candidates.

    fill:   "open"        - correct: next bar's open after the decision bar (default)
            "pessimistic" - worse of next bar's open and its (high+low+close)/3, per leg and side
            "signal_close"- DELIBERATE LOOKAHEAD BUG for the correctness trap: fill at the
                            close of the bar that produced the decision
    entry_delay:     bars (int or per-candidate array) between the signal bar and the order.
    exit_after_bars: optional per-candidate holding period; when given, only the time and
                     end-of-day exits apply (used by the random-entry baseline).
    """
    i, d, t = (cands[c].to_numpy() for c in ("i", "d", "t"))
    zs = cands["z"].to_numpy()
    has_news = cands["has_news"].to_numpy()
    delay = np.broadcast_to(np.asarray(entry_delay, dtype=int), i.shape)
    fixed_hold = None if exit_after_bars is None else np.broadcast_to(np.asarray(exit_after_bars, dtype=int), i.shape)
    ei = panel.etf_idx[i]

    # Entry: order sent at the close of bar t + delay -> first available open after it.
    td = t + delay
    t1 = np.minimum(td + 1, S - 1)
    e = np.where(td + 1 < S, panel.next_open_slot[i, d, t1], -1)
    etf_e = np.where(td + 1 < S, panel.next_open_slot[ei, d, t1], -1)
    ok = (e >= 0) & ((etf_e >= 0) | (not hedge))
    if fixed_hold is not None:
        fixed_hold = fixed_hold[ok]
    i, d, t, td, t1, zs, has_news, ei, e = (x[ok] for x in (i, d, t, td, t1, zs, has_news, ei, e))
    n = len(i)
    if n == 0:
        return _empty_trades()

    sgn = np.sign(zs)
    direction = -sgn                                     # fade: short after an up-move
    if fill == "signal_close":
        entry_px, etf_entry_px = panel.close_ff[i, d, td], panel.close_ff[ei, d, td]
    else:
        entry_px, etf_entry_px = panel.next_open[i, d, t1], panel.next_open[ei, d, t1]
        if fill == "pessimistic":
            entry_px = _worse_price(panel, i, d, e, entry_px, buy=direction > 0)
            etf_e2 = panel.next_open_slot[ei, d, t1]
            etf_entry_px = _worse_price(panel, ei, d, etf_e2, etf_entry_px, buy=direction < 0)

    # Exit conditions at the close of bars k = e, e+1, ..., e+H-1.
    H = int(hold_bars if fixed_hold is None else max(fixed_hold.max(), 1))
    j = np.arange(H)
    k = e[:, None] + j[None, :]
    valid = k <= S - 1
    kc = np.minimum(k, S - 1)
    ii, dd = i[:, None], d[:, None]
    zk = feats.z[ii, dd, kc]
    stop_level = (np.abs(zs) if stop_reference == "signal" else np.full(n, z_entry)) + C.STOP_Z_OFFSET
    with np.errstate(invalid="ignore"):
        if fixed_hold is None:
            cond = {
                "reversion": (np.abs(zk) < z_exit) | (np.sign(zk) != sgn[:, None]),
                "time": np.broadcast_to((j + 1 >= hold_bars)[None, :], k.shape),
                "stop": sgn[:, None] * zk >= stop_level[:, None],
                "news": feats.news_cnt[ii, dd, kc] > feats.news_cnt[i, d, t][:, None],
            }
        else:
            never = np.zeros(k.shape, bool)
            cond = {"reversion": never, "time": (j[None, :] + 1) >= fixed_hold[:, None],
                    "stop": never, "news": never}
        cond["eod"] = kc == S - 1
    fired = valid & np.logical_or.reduce(list(cond.values()))
    first = fired.argmax(axis=1)                         # every row fires by H or by 15:55
    rows = np.arange(n)
    kx = k[rows, first]                                  # exit decision bar
    reason = np.select([cond[r][rows, first] for r in REASON_ORDER[:-1]], REASON_ORDER[:-1], "eod")

    # Exit fill: next bar's open, or the 15:55 close if the decision is at the last bar
    # (or no later bar trades today).
    def exit_fill(sym, buy):
        nxt = np.minimum(kx + 1, S - 1)
        slot = np.where(kx >= S - 1, -1, panel.next_open_slot[sym, d, nxt])
        if fill == "signal_close":
            return panel.close_ff[sym, d, kx], np.where(kx >= S - 1, -1, kx + 1)
        px = np.where(slot >= 0, panel.next_open[sym, d, nxt], panel.close_ff[sym, d, S - 1])
        if fill == "pessimistic":
            px = np.where(slot >= 0, _worse_price(panel, sym, d, slot, px, buy), px)
        return px, slot

    exit_px, exit_slot = exit_fill(i, buy=direction < 0)
    etf_exit_px, _ = exit_fill(ei, buy=direction > 0)

    beta = feats.beta[i, d]
    stock_ret = direction * (exit_px / entry_px - 1.0)
    hedge_ret = -direction * beta * (etf_exit_px / etf_entry_px - 1.0) if hedge else np.zeros(n)
    gross = 1e4 * (stock_ret + hedge_ret)
    cost = 2.0 * (C.COST_BPS_PER_SIDE + C.COMMISSION_BPS) * (1.0 + (np.abs(beta) if hedge else 0.0))

    day = panel.days.values[d]
    open_ = np.timedelta64(C.SESSION_OPEN.hour * 60 + C.SESSION_OPEN.minute, "m")
    step = np.timedelta64(C.BAR_MINUTES, "m")
    entry_time = day + open_ + (td + 1 if fill == "signal_close" else e) * step
    exit_time = day + open_ + np.where(exit_slot >= 0, exit_slot, S) * step   # S*5min after open = 16:00
    stocks = np.array(panel.symbols)
    return pd.DataFrame({
        "i": i, "d": d, "t": t, "exit_bar": kx,
        "date": day, "symbol": stocks[i], "etf": stocks[ei],
        "signal_time": day + open_ + (t + 1) * step,       # close time of the signal bar
        "entry_time": entry_time, "exit_time": exit_time,
        "direction": direction.astype(int), "z_signal": zs, "z_at_exit": zk[rows, first],
        "z_sector": feats.z_sector[i, d, t], "beta": beta, "sigma": feats.sigma[i, d],
        "has_news": has_news, "entry_px": entry_px, "exit_px": exit_px,
        "etf_entry_px": etf_entry_px if hedge else np.nan, "etf_exit_px": etf_exit_px if hedge else np.nan,
        "exit_reason": reason, "bars_held": kx - e + 1,
        "hold_min": (exit_time - entry_time) / np.timedelta64(1, "m"),
        "gross_bps": gross, "cost_bps": cost, "net_bps": gross - cost,
    })


def _worse_price(panel: sg.Panel, sym, d, slot, open_px, buy) -> np.ndarray:
    """Worse of the bar's open and its typical price (H+L+C)/3 for the trade side."""
    sl = np.maximum(slot, 0)
    typical = (panel.high[sym, d, sl] + panel.low[sym, d, sl] + panel.close[sym, d, sl]) / 3.0
    typical = np.where(np.isnan(typical), open_px, typical)
    return np.where(buy, np.fmax(open_px, typical), np.fmin(open_px, typical))


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=["i", "d", "t", "exit_bar", "date", "symbol", "etf", "signal_time",
                                 "entry_time", "exit_time", "direction", "z_signal", "z_at_exit", "z_sector",
                                 "beta", "sigma", "has_news", "entry_px", "exit_px", "etf_entry_px",
                                 "etf_exit_px", "exit_reason", "bars_held", "hold_min", "gross_bps",
                                 "cost_bps", "net_bps"])


# ---------------------------------------------------------------------------
# Portfolio: position limits + sizing on top of the simulated trades
# ---------------------------------------------------------------------------
def select_portfolio(trades: pd.DataFrame, max_positions: int = C.MAX_POSITIONS) -> pd.DataFrame:
    """
    Walk candidate trades in chronological order of the signal bar (ties: largest |z|
    first) and accept a trade if the stock has no open position, wasn't traded today,
    and fewer than max_positions are open. A position counts as open at signal bar t
    while its exit decision bar is later than t.
    """
    if trades.empty:
        return trades
    tr = trades.assign(_absz=trades["z_signal"].abs()).sort_values(
        ["d", "t", "_absz"], ascending=[True, True, False], kind="stable")
    keep = np.zeros(len(tr), bool)
    cur_day, open_exit, traded = -1, [], set()
    for pos, (d, t, i, xb) in enumerate(zip(tr["d"].tolist(), tr["t"].tolist(),
                                            tr["i"].tolist(), tr["exit_bar"].tolist())):
        if d != cur_day:                                 # positions never cross days
            cur_day, open_exit, traded = d, [], set()
        if i in traded:
            continue
        open_exit = [x for x in open_exit if x > t]
        if len(open_exit) >= max_positions:
            continue
        keep[pos] = True
        traded.add(i)
        open_exit.append(xb)
    return tr[keep].drop(columns="_absz").reset_index(drop=True)


def size_trades(trades: pd.DataFrame, feats: sg.Features, capital: float = C.CAPITAL,
                sizing: str = C.SIZING) -> pd.DataFrame:
    """Add stock-leg notional and $ P&L at each cost multiplier."""
    tr = trades.copy()
    if sizing == "vol":
        notional = np.minimum(C.VOL_TARGET_RISK_USD / tr["sigma"].to_numpy(),
                              C.VOL_MAX_NOTIONAL_FRAC * capital)
    else:
        notional = np.full(len(tr), C.FIXED_NOTIONAL_FRAC * capital)
    liq_cap = C.MAX_FRAC_OF_BAR_DOLLAR_VOLUME * feats.bar_dollar_vol[tr["i"].to_numpy(), tr["d"].to_numpy()]
    tr["notional"] = np.minimum(notional, liq_cap)
    tr["hedge_notional"] = tr["notional"] * tr["beta"] if C.USE_HEDGE else 0.0
    for m in C.COST_MULTIPLIERS:
        tr[f"pnl_{m:g}x"] = tr["notional"] * (tr["gross_bps"] - m * tr["cost_bps"]) / 1e4
    return tr


# ---------------------------------------------------------------------------
# One call = one strategy configuration (used by the sweep and by validation/)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    lookback_bars: int = C.LOOKBACK_BARS
    z_entry: float = C.Z_ENTRY
    z_exit: float = C.Z_EXIT
    hold_bars: int = C.HOLD_BARS

    def as_dict(self) -> dict:
        return asdict(self)


def candidates_for(feats: sg.Features, params: Params, day_mask: np.ndarray | None = None,
                   news: str = "no_news") -> pd.DataFrame:
    """Entry candidates for `params` (switches feats to the params' lookback if needed)."""
    if feats.lookback != params.lookback_bars:
        sg.set_lookback(feats, params.lookback_bars)
    return sg.find_candidates(feats, params.z_entry, news=news, day_mask=day_mask)


def run_strategy(panel: sg.Panel, feats: sg.Features, params: Params = Params(),
                 day_mask: np.ndarray | None = None, *, cands: pd.DataFrame | None = None,
                 news: str = "no_news", max_positions: int = C.MAX_POSITIONS,
                 capital: float = C.CAPITAL, sizing: str = C.SIZING, **sim_kwargs) -> pd.DataFrame:
    """
    Full strategy for one configuration: candidates -> simulated trades -> position limits ->
    sizing. Returns the accepted, sized trades. `sim_kwargs` go to simulate_trades
    (fill, entry_delay, ...). Pass `cands` to reuse precomputed candidates.
    """
    if cands is None:
        cands = candidates_for(feats, params, day_mask, news)
    elif feats.lookback != params.lookback_bars:
        sg.set_lookback(feats, params.lookback_bars)
    tr = simulate_trades(panel, feats, cands, z_entry=params.z_entry, z_exit=params.z_exit,
                         hold_bars=params.hold_bars, **sim_kwargs)
    return size_trades(select_portfolio(tr, max_positions), feats, capital, sizing)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def trade_stats(tr: pd.DataFrame, cost_mult: float = 1.0) -> dict:
    """Per-trade stats in bps with a plain and a day-clustered t-stat."""
    if len(tr) == 0:
        return {"n": 0, "win_rate": np.nan, "mean_gross_bps": np.nan, "mean_net_bps": np.nan,
                "median_net_bps": np.nan, "t_stat": np.nan, "t_stat_day_clustered": np.nan}
    net = (tr["gross_bps"] - cost_mult * tr["cost_bps"]).to_numpy()
    n, mu = len(net), net.mean()
    sd = net.std(ddof=1) if n > 1 else np.nan
    resid_by_day = pd.Series(net - mu).groupby(tr["d"].to_numpy()).sum().to_numpy()
    g = len(resid_by_day)
    se_cl = np.sqrt(g / (g - 1) * np.sum(resid_by_day ** 2)) / n if g > 1 else np.nan
    return {"n": n, "win_rate": float((net > 0).mean()), "mean_gross_bps": float(tr["gross_bps"].mean()),
            "mean_net_bps": float(mu), "median_net_bps": float(np.median(net)),
            "t_stat": float(mu / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
            "t_stat_day_clustered": float(mu / se_cl) if se_cl and se_cl > 0 else np.nan}


def first_per_stock_day(tr: pd.DataFrame) -> pd.DataFrame:
    """De-duplicated view: the first qualifying signal per stock per day."""
    return tr.sort_values(["d", "t"]).drop_duplicates(["i", "d"], keep="first")


def daily_pnl(tr: pd.DataFrame, days: pd.DatetimeIndex, col: str) -> pd.Series:
    s = tr.groupby("date")[col].sum() if len(tr) else pd.Series(dtype=float)
    return s.reindex(days, fill_value=0.0)


def portfolio_metrics(tr: pd.DataFrame, days: pd.DatetimeIndex, cost_mult: float = 1.0,
                      capital: float = C.CAPITAL) -> dict:
    pnl = daily_pnl(tr, days, f"pnl_{cost_mult:g}x")
    equity = capital + pnl.cumsum()
    dd = equity - equity.cummax().clip(lower=capital)
    ret = pnl / capital
    sd = ret.std(ddof=1)
    out = {
        "days": len(days), "trades": len(tr),
        "total_pnl": float(pnl.sum()),
        "ann_return": float(ret.mean() * 252) if len(days) else np.nan,
        "ann_vol": float(sd * np.sqrt(252)) if len(days) > 1 else np.nan,
        "sharpe": float(ret.mean() / sd * np.sqrt(252)) if len(days) > 1 and sd > 0 else np.nan,
        "max_drawdown_usd": float(dd.min()) if len(dd) else 0.0,
        "max_drawdown_pct": float(dd.min() / capital) if len(dd) else 0.0,
        "avg_hold_min": float(tr["hold_min"].mean()) if len(tr) else np.nan,
    }
    st = trade_stats(tr, cost_mult)
    out.update({"mean_net_bps": st["mean_net_bps"], "win_rate": st["win_rate"]})
    reasons = tr["exit_reason"].value_counts(normalize=True) if len(tr) else pd.Series(dtype=float)
    out.update({f"exit_{r}": float(reasons.get(r, 0.0)) for r in REASON_ORDER})
    return out


def z_bucket(z: pd.Series) -> pd.Series:
    labels = [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+" for lo, hi in C.Z_BUCKETS]
    edges = [C.Z_BUCKETS[0][0]] + [hi for _, hi in C.Z_BUCKETS]
    return pd.cut(z.abs(), bins=edges, labels=labels, right=False)


def breakdown(tr: pd.DataFrame, key: pd.Series, cost_mult: float = 1.0) -> pd.DataFrame:
    rows = {}
    for k, g in tr.groupby(key, observed=False):
        st = trade_stats(g, cost_mult)
        rows[k] = {"n": st["n"], "win_rate": st["win_rate"], "mean_net_bps": st["mean_net_bps"],
                   "t_stat": st["t_stat"],
                   "pnl_usd": float(g[f"pnl_{cost_mult:g}x"].sum()) if f"pnl_{cost_mult:g}x" in g else np.nan}
    return pd.DataFrame(rows).T


def breakdowns(tr: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if tr.empty:
        return {}
    return {
        "year": breakdown(tr, pd.to_datetime(tr["date"]).dt.year),
        "sector": breakdown(tr, tr["etf"]),
        "entry_hour": breakdown(tr, pd.to_datetime(tr["entry_time"]).dt.hour),
        "abs_z_bucket": breakdown(tr, z_bucket(tr["z_signal"])),
    }


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
@dataclass
class Results:
    periods: dict[str, pd.DatetimeIndex]
    study: pd.DataFrame                      # every qualifying bar, news and no-news
    portfolio: pd.DataFrame                  # accepted no-news trades with sizing / $ P&L
    study_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    portfolio_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    breakdowns: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    sweep: pd.DataFrame | None = None


def evaluation_periods(feats: sg.Features, days: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    """All tradable days, split chronologically into in-sample (first 60%) / out-of-sample."""
    eval_days = days[feats.tradable.any(axis=0)]
    cut = int(round(len(eval_days) * C.IN_SAMPLE_FRAC))
    return {"all": eval_days, "in_sample": eval_days[:cut], "out_of_sample": eval_days[cut:]}


def in_period(tr: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    return tr[tr["date"].isin(days)]


def sweep_grid(grid: dict = C.SWEEP_GRID) -> list[Params]:
    return [Params(L, ze, zx, H) for L in grid["lookback_bars"] for ze in grid["z_entry"]
            for zx in grid["z_exit"] for H in grid["hold_bars"]]


def run_sweep(panel: sg.Panel, feats: sg.Features, is_days: pd.DatetimeIndex,
              grid: dict = C.SWEEP_GRID, verbose: bool = True) -> pd.DataFrame:
    """In-sample grid over L, z_entry, z_exit, H (portfolio-level, no-news trades)."""
    day_mask = panel.days.isin(is_days)
    rows, base_L = [], feats.lookback
    try:
        for L in grid["lookback_bars"]:
            for ze in grid["z_entry"]:
                cands = candidates_for(feats, Params(L, ze), day_mask)
                for zx in grid["z_exit"]:
                    for H in grid["hold_bars"]:
                        p = Params(L, ze, zx, H)
                        tr = simulate_trades(panel, feats, cands, z_entry=ze, z_exit=zx, hold_bars=H)
                        port = size_trades(select_portfolio(tr), feats)
                        m = portfolio_metrics(port, is_days)
                        st = trade_stats(first_per_stock_day(tr))
                        rows.append({**p.as_dict(),
                                     "trades": m["trades"], "mean_net_bps": m["mean_net_bps"],
                                     "win_rate": m["win_rate"], "sharpe": m["sharpe"],
                                     "ann_return": m["ann_return"], "max_drawdown_pct": m["max_drawdown_pct"],
                                     "signal_study_first_n": st["n"],
                                     "signal_study_first_net_bps": st["mean_net_bps"]})
            if verbose:
                print(f"  sweep L={L} done", flush=True)
    finally:
        sg.set_lookback(feats, base_L)
    return pd.DataFrame(rows)


def run_backtest(panel: sg.Panel, feats: sg.Features, sweep: bool = True) -> Results:
    periods = evaluation_periods(feats, panel.days)

    # 1. Signal study: every bar passing rules 1, 3, 5 - regardless of news.
    cands = sg.find_candidates(feats, C.Z_ENTRY, news="any")
    study = simulate_trades(panel, feats, cands)
    study["group"] = np.where(study["has_news"], "news", "no_news")

    # 2. Portfolio backtest on the no-news trades.
    portfolio = size_trades(select_portfolio(study[study["group"] == "no_news"]), feats)

    res = Results(periods=periods, study=study, portfolio=portfolio)
    for pname, pdays in periods.items():
        st = in_period(study, pdays)
        rows = {}
        for view, frame in (("every_bar", st), ("first_per_stock_day", first_per_stock_day(st))):
            for grp in ("no_news", "news"):
                for m in C.COST_MULTIPLIERS:
                    rows[(view, grp, f"{m:g}x")] = trade_stats(frame[frame["group"] == grp], m)
        res.study_tables[pname] = pd.DataFrame(rows).T
        pt = in_period(portfolio, pdays)
        res.portfolio_tables[pname] = pd.DataFrame(
            {f"{m:g}x cost": portfolio_metrics(pt, pdays, m) for m in C.COST_MULTIPLIERS})
        res.breakdowns[pname] = breakdowns(pt)
    res.breakdowns["signal_study_abs_z"] = {
        grp: breakdown(study[study["group"] == grp], z_bucket(study.loc[study["group"] == grp, "z_signal"]))
        for grp in ("no_news", "news")}

    if sweep:
        print("running in-sample parameter sweep ...", flush=True)
        res.sweep = run_sweep(panel, feats, periods["in_sample"])
    return res


def load_inputs(synthetic: bool, synth_days: int) -> tuple[sg.Panel, pd.DataFrame]:
    if synthetic:
        import synthetic as syn
        bars, news, _ = syn.generate(synth_days)
        return sg.build_panel(bars, C.STOCKS, C.SECTOR_MAP), news
    import fetch_data
    if not fetch_data.cached_symbols():
        raise SystemExit("No cached bars found. Run `python fetch_data.py` first (or use --synthetic).")
    panel = sg.build_panel_streaming(fetch_data.load_symbol_bars, C.STOCKS, C.SECTOR_MAP, fetch_data.cached_symbols())
    return panel, fetch_data.load_news()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use generated data with a planted effect")
    ap.add_argument("--synth-days", type=int, default=C.SYNTH_DAYS)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    args = ap.parse_args()

    import report
    t0 = time.time()
    panel, news = load_inputs(args.synthetic, args.synth_days)
    feats = sg.compute_features(panel, news, C.LOOKBACK_BARS)
    print(f"panel: {panel.n_stocks} stocks, {len(panel.symbols) - panel.n_stocks} ETFs, "
          f"{len(panel.days)} days ({panel.days[0].date()} .. {panel.days[-1].date()}), "
          f"{int(panel.early_close.sum())} early-close days  [{time.time() - t0:.1f}s]", flush=True)
    res = run_backtest(panel, feats, sweep=not args.no_sweep)
    report.write_all(res, panel, args.out, synthetic=args.synthetic)

    from validation import trial_log
    dataset = "synthetic" if args.synthetic else "real"
    for pname, tbl in res.portfolio_tables.items():
        m = tbl["1x cost"]
        trial_log.log([{**Params().as_dict(), "n_trades": m["trades"], "mean_net_bps": m["mean_net_bps"],
                        "sharpe": m["sharpe"]}], dataset, "backtest", pname)
    if res.sweep is not None:
        trial_log.log(res.sweep.to_dict("records"), dataset, "backtest_sweep", "in_sample")
    print(f"done in {time.time() - t0:.1f}s; results in {args.out}/")


if __name__ == "__main__":
    main()
