"""
`correctness`: bug and lookahead detection.
  1. Planted effect: synthetic market with known reversion -> detected at roughly the planted size.
  2. Null: random-walk residuals -> mean net bps ~= -cost (95% CI includes -cost).
  3. Lookahead traps: (a) fill at the signal bar's close, (b) news visible 5 minutes early.
     Results must change; if they barely move the timing logic is suspicious.
  4. Entry delay 0..3 bars.
  5. Random trade audit: 20 trades with every input printed for manual checking.
  6. pytest lookahead checklist.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pandas as pd

import backtest as bt
import config as C
import signals as sg
import synthetic
from validation import trial_log
from validation.common import Ctx, fmt_table, summarize, write_summary

S = C.BARS_PER_DAY


def _slot(ts: pd.Series, date: pd.Series) -> np.ndarray:
    """Slot index of a bar-start timestamp (16:00 -> 78)."""
    mins = (ts - date) / pd.Timedelta(minutes=1) - (C.SESSION_OPEN.hour * 60 + C.SESSION_OPEN.minute)
    return (mins // C.BAR_MINUTES).astype(int).to_numpy()


def planted_capture_bps(trades: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    """
    For each trade, the planted (noise-free) return the trade should capture in bps: the
    planted shock + drift returns of the bars it holds (entry open -> exit open), signed by
    the trade direction. NaN when the trade does not sit on a planted shock.
    """
    decay = np.exp(-np.arange(8) / 2.5)
    decay /= decay.sum()
    key = truth.set_index(["stock", "day"])
    out = np.full(len(trades), np.nan)
    e_slot = _slot(trades["entry_time"], trades["date"])
    x_slot = _slot(trades["exit_time"], trades["date"])
    for n, (sym, day, direction) in enumerate(zip(trades["symbol"], trades["date"], trades["direction"])):
        if (sym, day) not in key.index:
            continue
        sh = key.loc[(sym, day)]
        if isinstance(sh, pd.DataFrame):
            sh = sh.iloc[0]
        planted = np.zeros(S)
        s0, w = int(sh["slot"]), int(sh["width"])
        planted[s0:s0 + w] = sh["size"] / w
        hi = min(s0 + w + 8, S)
        planted[s0 + w:hi] += sh["drift_frac"] * sh["size"] * decay[: hi - s0 - w]
        out[n] = 1e4 * direction * planted[e_slot[n]:x_slot[n]].sum()
    return out


def planted_test(seed: int) -> dict:
    bars, news, truth = synthetic.generate(n_days=250, seed=seed)
    panel = sg.build_panel(bars, C.STOCKS, C.SECTOR_MAP)
    feats = sg.compute_features(panel, news)
    days = panel.days[feats.tradable.any(axis=0)]
    port = bt.run_strategy(panel, feats)
    planted = planted_capture_bps(port, truth["shocks"])
    matched = np.isfinite(planted)
    study = bt.simulate_trades(panel, feats, sg.find_candidates(feats, C.Z_ENTRY, news="any"))
    first = bt.first_per_stock_day(study)
    s_no, s_news = (bt.trade_stats(first[first["has_news"] == g]) for g in (False, True))
    ratio = port.loc[matched, "gross_bps"].mean() / np.mean(planted[matched]) if matched.any() else np.nan
    m = summarize(port, days)
    passed = bool(m["t_stat"] > 3 and s_news["mean_net_bps"] < 0 and 0.5 <= ratio <= 1.5)
    return {"portfolio": m, "study_no_news": s_no, "study_news": s_news,
            "matched_trades": int(matched.sum()), "planted_bps": float(np.mean(planted[matched])),
            "measured_gross_bps_on_matched": float(port.loc[matched, "gross_bps"].mean()),
            "measured_over_planted": float(ratio), "pass": passed}


def null_test(seed: int) -> dict:
    bars, news, _ = synthetic.generate(n_days=250, seed=seed, shock_prob=0.0)
    panel = sg.build_panel(bars, C.STOCKS, C.SECTOR_MAP)
    feats = sg.compute_features(panel, news)
    days = panel.days[feats.tradable.any(axis=0)]
    port = bt.run_strategy(panel, feats)
    m = summarize(port, days)
    net = port["net_bps"].to_numpy()
    se = abs(m["mean_net_bps"] / m["t_stat"]) if m["t_stat"] else np.nan     # day-clustered SE
    lo, hi = m["mean_net_bps"] - 1.96 * se, m["mean_net_bps"] + 1.96 * se
    cost = float(port["cost_bps"].mean())
    significantly_positive = lo > 0
    return {"portfolio": m, "mean_cost_bps": cost, "ci95": [lo, hi], "n": len(net),
            "ci_includes_minus_cost": bool(lo <= -cost <= hi),
            "significantly_positive": bool(significantly_positive),
            "pass": bool(lo <= -cost <= hi and not significantly_positive)}


def lookahead_traps(ctx: Ctx) -> dict:
    days = ctx.periods["all"]
    mask = ctx.mask("all")
    base = bt.run_strategy(ctx.panel, ctx.feats, day_mask=mask)
    trap_a = bt.run_strategy(ctx.panel, ctx.feats, day_mask=mask, fill="signal_close")
    # (b) news visible 5 minutes before it was created
    early = ctx.news.copy()
    early["created_at"] = pd.to_datetime(early["created_at"], utc=True) - pd.Timedelta(minutes=5)
    feats_b = dataclasses.replace(
        ctx.feats, news_cnt=sg.news_counts(ctx.panel, sg.company_news_times(early, ctx.panel.stocks)))
    trap_b = bt.run_strategy(ctx.panel, feats_b, day_mask=mask)
    flipped = int((feats_b.has_news != ctx.feats.has_news)[:, mask].sum())

    rows = {"baseline": summarize(base, days), "a_fill_at_signal_close": summarize(trap_a, days),
            "b_news_5min_early": summarize(trap_b, days)}
    b0 = rows["baseline"]["mean_net_bps"]
    res = {"table": rows, "news_flags_changed_by_b": flipped}
    for k in ("a_fill_at_signal_close", "b_news_5min_early"):
        delta = rows[k]["mean_net_bps"] - b0
        res[f"{k}_delta_bps"] = delta
        res[f"{k}_suspicious"] = bool(abs(delta) < max(0.5, 0.05 * abs(b0)))
    res["pass"] = not (res["a_fill_at_signal_close_suspicious"] and res["b_news_5min_early_suspicious"])
    return res


def delay_test(ctx: Ctx) -> pd.DataFrame:
    rows = {}
    for period in ("dev", ctx.headline):
        cands = bt.candidates_for(ctx.feats, bt.Params(), ctx.mask(period))
        for k in range(4):
            port = bt.run_strategy(ctx.panel, ctx.feats, cands=cands, entry_delay=k)
            rows[(period, k)] = summarize(port, ctx.periods[period])
    df = pd.DataFrame(rows).T
    df.index.names = ["period", "delay_bars"]
    return df


def audit_trades(ctx: Ctx, n: int = 20) -> str:
    p, f = ctx.panel, ctx.feats
    port = bt.run_strategy(p, f, day_mask=ctx.mask("all"))
    pick = port.iloc[np.sort(ctx.rng("audit").choice(len(port), size=min(n, len(port)), replace=False))]
    news = ctx.news.copy()
    news["created_at"] = pd.to_datetime(news["created_at"], utc=True)
    L = f.lookback
    blocks = []
    for k, tr in enumerate(pick.itertuples(), 1):
        i, d, t, ei = tr.i, tr.d, tr.t, p.etf_idx[tr.i]
        hdr = (f"#{k} {tr.symbol} hedge {tr.etf} {pd.Timestamp(tr.date).date()}  "
               f"{'LONG' if tr.direction > 0 else 'SHORT'}  signal close {pd.Timestamp(tr.signal_time):%H:%M}  "
               f"z={tr.z_signal:+.2f} sector_z={tr.z_sector:+.2f} beta={tr.beta:.3f} sigma={tr.sigma * 1e4:.2f}bps\n"
               f"   entry {pd.Timestamp(tr.entry_time):%H:%M} @ {tr.entry_px:.4f} (ETF {tr.etf_entry_px:.4f})  "
               f"exit {pd.Timestamp(tr.exit_time):%H:%M} @ {tr.exit_px:.4f} (ETF {tr.etf_exit_px:.4f})  "
               f"reason={tr.exit_reason}  gross={tr.gross_bps:+.2f} cost={tr.cost_bps:.2f} net={tr.net_bps:+.2f} bps")
        lo, hi = max(0, t - L + 1), min(S - 1, tr.exit_bar + 1)
        rows = []
        for s in range(lo, hi + 1):
            mark = "S" if s == t else ("X" if s == tr.exit_bar else "")
            rows.append({"bar": f"{(pd.Timestamp(tr.date) + pd.Timedelta(minutes=570 + 5 * s)):%H:%M}",
                         "": mark,
                         "o": p.open[i, d, s], "h": p.high[i, d, s], "l": p.low[i, d, s], "c": p.close[i, d, s],
                         "etf_o": p.open[ei, d, s], "etf_c": p.close[ei, d, s],
                         "r_s_bps": 1e4 * f.ret[i, d, s], "r_e_bps": 1e4 * f.ret[ei, d, s],
                         "eps_bps": 1e4 * f.eps[i, d, s], "z": f.z[i, d, s], "news_cnt": f.news_cnt[i, d, s]})
        tbl = pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}")
        t0 = pd.Timestamp(p.prev_close_ns[d], tz="UTC")
        t1 = pd.Timestamp(tr.exit_time).tz_localize(C.TIMEZONE).tz_convert("UTC")
        nw = news[(news["created_at"] > t0) & (news["created_at"] <= t1)
                  & news["symbols"].apply(lambda s: tr.symbol in list(s))]
        if len(nw):
            nl = "\n".join(f"     created {r.created_at.tz_convert(C.TIMEZONE):%Y-%m-%d %H:%M:%S}  "
                           f"updated {pd.Timestamp(r.updated_at).tz_convert(C.TIMEZONE):%H:%M:%S}  "
                           f"tags={list(r.symbols)}  {str(r.headline)[:70]}" for r in nw.itertuples())
        else:
            nl = "     (none)"
        blocks.append(f"{hdr}\n{tbl}\n   news for {tr.symbol} since previous close "
                      f"(company-specific iff <= {C.MAX_NEWS_SYMBOLS} tags):\n{nl}\n")
    return ("S = signal bar (decision at its close), X = exit decision bar; entries/exits fill at the\n"
            "NEXT bar's open (EOD at the 15:55 close). r/eps in bps, prices adjusted.\n\n" + "\n".join(blocks))


def run_pytest() -> dict:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(C.ROOT / "tests")],
                          capture_output=True, text=True, cwd=C.ROOT)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or [""]
    return {"pass": proc.returncode == 0, "summary": tail[0], "output": proc.stdout[-4000:]}


def run(ctx: Ctx) -> dict:
    out = ctx.out("correctness")
    print("planted-effect test ...", flush=True)
    planted = planted_test(ctx.seed)
    print("null test ...", flush=True)
    null = null_test(ctx.seed + 1)
    print("lookahead traps ...", flush=True)
    traps = lookahead_traps(ctx)
    print("delay test ...", flush=True)
    delay = delay_test(ctx)
    delay.to_csv(out / "delay_test.csv")
    (out / "audit_trades.txt").write_text(audit_trades(ctx))
    print("pytest ...", flush=True)
    tests = run_pytest()
    (out / "pytest_output.txt").write_text(tests["output"])

    pd.DataFrame(traps["table"]).T.to_csv(out / "lookahead_traps.csv")
    trial_log.log([{**bt.Params().as_dict(), **v, "extra": {"variant": k}} for k, v in traps["table"].items()],
                  ctx.dataset, "correctness_traps", "all")

    warn_lines = []
    for k in ("a_fill_at_signal_close", "b_news_5min_early"):
        if traps[f"{k}_suspicious"]:
            warn_lines.append(f"  WARNING: trap {k} changed mean net bps by only {traps[f'{k}_delta_bps']:+.2f} - "
                              "if the trap is real this suggests the timing logic is not doing what it should")
    lines = [
        f"data for traps/delay/audit: {ctx.dataset}; planted and null tests always use fresh synthetic markets",
        "", f"1. Planted effect: {'PASS' if planted['pass'] else 'FAIL'}",
        f"   portfolio: n={planted['portfolio']['n_trades']} mean net {planted['portfolio']['mean_net_bps']:+.2f} bps "
        f"(t={planted['portfolio']['t_stat']:.1f}); study news group net {planted['study_news']['mean_net_bps']:+.2f} bps",
        f"   on {planted['matched_trades']} trades sitting on planted shocks: planted capture "
        f"{planted['planted_bps']:+.2f} bps vs measured gross {planted['measured_gross_bps_on_matched']:+.2f} bps "
        f"(ratio {planted['measured_over_planted']:.2f}, pass band 0.5-1.5)",
        "", f"2. Null (random-walk residuals): {'PASS' if null['pass'] else 'FAIL'}",
        f"   mean net {null['portfolio']['mean_net_bps']:+.2f} bps, 95% CI [{null['ci95'][0]:+.2f}, {null['ci95'][1]:+.2f}], "
        f"-cost = {-null['mean_cost_bps']:.2f} bps, n={null['n']}"
        + ("   *** SIGNIFICANTLY POSITIVE ON PURE NOISE: THERE IS A BUG ***" if null["significantly_positive"] else ""),
        "", f"3. Lookahead traps ({'PASS' if traps['pass'] else 'FAIL'}: results must change):",
        fmt_table(pd.DataFrame(traps["table"]).T[["n_trades", "mean_net_bps", "t_stat", "sharpe"]]),
        f"   news flags changed by the 5-min-early trap: {traps['news_flags_changed_by_b']:,} stock-bars",
        *warn_lines,
        "", "4. Entry delay (bars after the signal bar's close; 0 = next bar's open):",
        fmt_table(delay[["n_trades", "mean_net_bps", "t_stat", "sharpe"]]),
        "", "5. Random trade audit: audit_trades.txt (20 trades, every input)",
        "", f"6. pytest lookahead checklist: {'PASS' if tests['pass'] else 'FAIL'} - {tests['summary']}",
    ]
    res = {"planted": planted, "null": null, "traps": traps, "delay": delay.reset_index().to_dict("records"),
           "pytest": {k: v for k, v in tests.items() if k != "output"},
           "pass": bool(planted["pass"] and null["pass"] and traps["pass"] and tests["pass"])}
    write_summary(out, "CORRECTNESS - bug and lookahead detection", lines, res)
    return res
