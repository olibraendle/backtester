"""Write results/validation[_synthetic]/REPORT.md from the saved results.json of each subcommand."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import config as C
from validation.common import load_results


def _f(x, fmt="{:+.2f}"):
    try:
        return fmt.format(x)
    except (TypeError, ValueError):
        return "n/a"


def write(out_root: Path, synthetic: bool) -> Path:
    R = {k: load_results(out_root, k) for k in
         ("coverage", "correctness", "stats", "permutation", "synthetic", "robustness", "execution", "risk")}
    h = "wf_oos"
    checks = []   # (name, status, detail)

    def add(name, ok, detail, info=False):
        status = "INFO" if info else ("n/a" if ok is None else ("PASS" if ok else "FAIL"))
        checks.append((name, status, detail))

    c = R["correctness"]
    if c:
        add("1. Correctness: planted effect detected, null ~ -cost, lookahead traps change results",
            c["pass"],
            f"planted ratio {_f(c['planted']['measured_over_planted'], '{:.2f}')} ({'ok' if c['planted']['pass'] else 'FAIL'}); "
            f"null mean {_f(c['null']['portfolio']['mean_net_bps'])} bps vs -cost {_f(-c['null']['mean_cost_bps'])} "
            f"({'ok' if c['null']['pass'] else 'FAIL'}); trap deltas: fill-at-close "
            f"{_f(c['traps']['a_fill_at_signal_close_delta_bps'])}, news-early {_f(c['traps']['b_news_5min_early_delta_bps'])} bps; "
            f"pytest {c['pytest']['summary']}")
    else:
        add("1. Correctness", None, "not run")

    s = R["stats"]
    if s:
        g = s[h]["groups"]
        no, nw = g["first_per_stock_day/no_news"], g["first_per_stock_day/news"]
        port = s[h]["portfolio"]
        add("2. no_news mean net > 0 with t > 2; news <= no_news",
            bool(port["mean_net_bps"] > 0 and port["t_stat"] > C.T_STAT_PASS and nw["mean_net_bps"] <= no["mean_net_bps"]),
            f"{h} portfolio {_f(port['mean_net_bps'])} bps (t={_f(port['t_stat'], '{:.1f}')}, n={port['n_trades']}); "
            f"study first-per-stock-day: no_news {_f(no['mean_net_bps'])}, news {_f(nw['mean_net_bps'])} bps")
        add("3. Placebo news filter clearly worse (real > 95th pct)", bool(s[h]["placebo_pct_net"] >= 95),
            f"real at the {_f(s[h]['placebo_pct_net'], '{:.1f}')}th percentile; placebo mean {_f(s[h]['placebo_mean_net'])}, "
            f"95th {_f(s[h]['placebo_p95_net'])} bps; random-entry baseline percentile {_f(s[h]['random_entry_pct_net'], '{:.1f}')}")
    else:
        add("2. Group statistics", None, "not run")
        add("3. Placebo news filter", None, "not run")

    p = R["permutation"]
    add("4. Permutation test p < 0.01", p["pass"] if p else None,
        f"{p['n_permutations']} permutations; p(best Sharpe) = {_f(p['best_sharpe']['p_value'], '{:.4f}')}, "
        f"p(best net bps) = {_f(p['best_net_bps']['p_value'], '{:.4f}')}" if p else "not run")

    r = R["robustness"]
    if r:
        wf = r["walk_forward"]
        add("5. Walk-forward OOS positive at 2x costs", wf["positive_at_2x"],
            f"{wf['windows']} windows; chained OOS 1x {_f(wf['oos_1x']['mean_net_bps'])} bps (Sharpe "
            f"{_f(wf['oos_1x']['sharpe'], '{:.2f}')}), 2x {_f(wf['oos_2x']['mean_net_bps'])} bps (Sharpe "
            f"{_f(wf['oos_2x']['sharpe'], '{:.2f}')}); {wf['distinct_param_sets']} distinct parameter sets")
        hm, jt = r["heatmaps"], r["jitter"]
        add("6. Smooth heatmaps; >= 70% of jittered runs profitable",
            bool(not hm["any_isolated"] and jt[h]["pct_profitable"] >= 0.70),
            f"isolated peak: {hm['any_isolated']}; min share of positive cells {_f(hm['min_share_positive'], '{:.0%}')}; "
            f"jitter profitable {h} {_f(jt[h]['pct_profitable'], '{:.0%}')}, dev {_f(jt['dev']['pct_profitable'], '{:.0%}')}")
        cc = r["concentration"]
        add("7. Not dependent on top 1% trades / one stock / one sector / one year", cc["pass"],
            f"without top 1% {_f(cc['without_top1pct']['mean_net_bps'])} bps; leave-one-out positive: stocks "
            f"{cc['all_leave_one_stock_out_positive']}, sectors {cc['all_leave_one_sector_out_positive']}, years "
            f"{cc['all_leave_one_year_out_positive']}; max P&L share: stock {_f(cc['max_pnl_share_one_stock'], '{:.0%}')}, "
            f"sector {_f(cc['max_pnl_share_one_sector'], '{:.0%}')}")
    else:
        for n in ("5. Walk-forward", "6. Heatmaps / jitter", "7. Concentration"):
            add(n, None, "not run")

    if s:
        d = s["deflated_sharpe"]
        add("8. Deflated Sharpe > 0 at 95% confidence", d["pass"],
            f"DSR {_f(d['dsr'], '{:.3f}')} (SR {_f(d['sharpe_annual'], '{:.2f}')} vs SR0 {_f(d['sr0_annual'], '{:.2f}')} "
            f"from {d['n_trials']} selection trials; counting every logged config: N={s['deflated_sharpe_all_trials']['n_trials']}, "
            f"DSR {_f(s['deflated_sharpe_all_trials']['dsr'], '{:.3f}')})")
    else:
        add("8. Deflated Sharpe", None, "not run")

    k = R["risk"]
    add("9. Risk-based position size and kill criteria", None,
        f"recommend {_f(100 * k['recommended_notional_frac'], '{:.1f}')}% of capital per stock leg "
        f"(95th-pct 1y drawdown {_f(100 * k['dd95_at_current_size'], '{:.2f}')}% at the current 10%); kill if mean net < "
        f"{_f(k['kill']['threshold_bps'])} bps after {k['kill']['N_trades']} trades, or drawdown > "
        f"{C.MAX_DRAWDOWN_LIMIT:.0%} at the recommended size" if k else "not run", info=True)

    graded = [x for x in checks if x[1] in ("PASS", "FAIL")]
    fails = [x for x in graded if x[1] == "FAIL"]
    missing = [x for x in checks if x[1] == "n/a"]

    extra = []
    if R["synthetic"]:
        sy = R["synthetic"]
        extra.append(f"- Controlled-truth markets ({sy['runs_per_scenario']} runs each): false-positive rate "
                     f"{_f(sy['a_false_positive_rate'], '{:.1%}')}, detection rate {_f(sy['b_detection_rate'], '{:.1%}')}; "
                     f"when the edge vanishes the mean-net kill rule fires in {_f(sy['c_killed_rate_mean_rule'], '{:.0%}')} "
                     f"of runs (median {_f(sy['c_median_days_to_kill_mean_rule'], '{:.0f}')} days), the drawdown rule in "
                     f"{_f(sy['c_killed_rate_dd_rule'], '{:.0%}')} (median {_f(sy['c_median_days_to_kill_dd_rule'], '{:.0f}')} days)")
    if R["execution"]:
        ex = R["execution"][h]
        t = ex["table"]
        extra.append(f"- Execution ({h}): break-even cost {_f(ex['break_even_cost_bps_per_side'], '{:.2f}')} bps per leg per "
                     f"side (assumed {C.COST_BPS_PER_SIDE}); 3x cost {_f(t['cost 3x']['mean_net_bps'])} bps; pessimistic "
                     f"fills + slippage {_f(t['pessimistic + slippage']['mean_net_bps'])} bps; 20% best trades missed "
                     f"{_f(t['missed fills 20% (best first)']['mean_net_bps'])} bps")
    if R["coverage"]:
        fl = R["coverage"]["flagged_years"]
        extra.append("- Data coverage: " + ("no years flagged" if not fl else "flagged years: " + ", ".join(
            f"{x['year']} ({x['vs_median']:+.0%}: {x['recommendation'].split(' (')[0]})" for x in fl))
            + f"; median missing bars {_f(R['coverage']['missing_bars_median_pct'], '{:.2f}')}%")
    if r:
        o = r["out_of_universe"]
        extra.append(f"- Out-of-universe ({o['n_stocks']} stocks, unchanged params): {h} {_f(o[h]['mean_net_bps'])} bps "
                     f"(Sharpe {_f(o[h]['sharpe'], '{:.2f}')}), dev {_f(o['dev']['mean_net_bps'])} bps")
    unlocks = C.HOLDOUT_LOG_PATH.read_text().strip().splitlines() if C.HOLDOUT_LOG_PATH.exists() else []
    extra.append(f"- Holdout: {'LOCKED, never touched' if not unlocks else f'UNLOCKED {len(unlocks)} time(s), first {unlocks[0].split(chr(9))[0]} - it is spent'}")

    # Verdict
    priority = ["4.", "5.", "1.", "8.", "3.", "2.", "7.", "6."]
    if fails:
        weakest = sorted(fails, key=lambda x: next((i for i, p in enumerate(priority) if x[0].startswith(p)), 99))[0]
        verdict = (f"{len(graded) - len(fails)} of {len(graded)} graded checks pass. The strategy should NOT be "
                   f"considered validated. The most serious failure is \"{weakest[0][3:]}\": {weakest[2]}.")
    else:
        verdict = (f"All {len(graded)} graded checks pass" + (f" ({len(missing)} not run: "
                   + ", ".join(m[0].split(". ", 1)[-1] for m in missing) + ")" if missing else "")
                   + ". The edge survives every graded test that was run. ")
        if R["execution"]:
            t = R["execution"][h]["table"]
            worst = min(t, key=lambda kk: t[kk]["mean_net_bps"])
            verdict += (f"The weakest point found is execution: under \"{worst}\" the mean drops to "
                        f"{_f(t[worst]['mean_net_bps'])} bps per trade, so realised fill quality decides whether it is tradable.")
    if synthetic:
        verdict += (" NOTE: this report is on SYNTHETIC data with a planted effect - it validates the pipeline, "
                    "not the strategy.")

    lines = [f"# Validation report ({'SYNTHETIC data' if synthetic else 'real data'})", "",
             f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Headline period: walk-forward out-of-sample "
             f"({'synthetic split' if synthetic else f'{C.WF_OOS_START} .. {C.HOLDOUT_START}'}). "
             "Details per subcommand in the sibling folders (summary.txt, CSVs, PNGs).", "",
             "| # | Check | Result | Key numbers |", "|---|---|---|---|"]
    for name, status, detail in checks:
        num, title = name.split(". ", 1) if ". " in name else ("", name)
        lines.append(f"| {num} | {title} | **{status}** | {detail} |")
    lines += ["", "## Other findings", *extra, "", "## Verdict", verdict, ""]
    path = out_root / "REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print("\n".join(lines))
    return path
