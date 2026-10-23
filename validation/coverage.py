"""
`coverage`: data audit.
  * company-specific news items per stock per year (normalised per 252 trading days)
  * share of |z| >= z_entry signals flagged as news, per year
  * years whose news coverage differs from the median year by more than COVERAGE_DEVIATION
  * % of expected 5-min bars missing, per stock per year
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import report as rp
import signals as sg
from validation.common import Ctx, fmt_table, write_summary


def news_per_stock_year(ctx: Ctx) -> pd.DataFrame:
    """(stock x year) company-specific article counts, scaled to a full 252-day year."""
    times = sg.company_news_times(ctx.news, ctx.panel.stocks)
    days = ctx.periods["all"].append(ctx.periods.get("holdout", pd.DatetimeIndex([])))
    days_per_year = pd.Series(1, index=days).groupby(days.year).sum()
    rows = {}
    for sym, ns in times.items():
        t = pd.DatetimeIndex(pd.to_datetime(ns, utc=True)).tz_convert(C.TIMEZONE)
        rows[sym] = pd.Series(1, index=t).groupby(t.year).sum() if len(t) else pd.Series(dtype=float)
    counts = pd.DataFrame(rows).T.reindex(columns=days_per_year.index).fillna(0.0)
    return counts * (252.0 / days_per_year)


def signal_news_share(ctx: Ctx) -> pd.DataFrame:
    cands = sg.find_candidates(ctx.feats, C.Z_ENTRY, news="any")
    d = ctx.panel.days[cands["d"].to_numpy()]
    cands = cands[d.isin(ctx.periods["all"].append(ctx.periods.get("holdout", pd.DatetimeIndex([]))))]
    years = ctx.panel.days[cands["d"].to_numpy()].year
    g = cands.groupby(years)["has_news"]
    return pd.DataFrame({"signals": g.size(), "news_share": g.mean()})


def missing_bars(ctx: Ctx) -> pd.DataFrame:
    """% of expected RTH bars missing per stock per year (expected = bars in that day's session)."""
    p = ctx.panel
    last_slot = np.where(p.has_bar.any(axis=0), np.arange(C.BARS_PER_DAY), -1).max(axis=1)
    expected = (last_slot + 1).astype(float)                     # per day
    have = p.has_bar[: p.n_stocks].sum(-1).astype(float)          # (N, D)
    traded_day = have > 0                                         # before listing / data start: not expected
    years = p.days.year
    out = {}
    for i, sym in enumerate(p.stocks):
        m = traded_day[i]
        miss = pd.Series(np.where(m, expected - have[i], 0.0)).groupby(years).sum()
        exp_ = pd.Series(np.where(m, expected, 0.0)).groupby(years).sum()
        out[sym] = 100.0 * miss / exp_.replace(0, np.nan)
    return pd.DataFrame(out).T


def _heatmap(df: pd.DataFrame, title: str, path, cmap="Blues", fmt="{:.1f}") -> None:
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * df.shape[1], 1.0 + 0.22 * df.shape[0]), facecolor=rp.SURFACE)
    im = ax.imshow(df.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(df.shape[1]), df.columns, fontsize=7, color=rp.MUTED)
    ax.set_yticks(range(df.shape[0]), df.index, fontsize=6, color=rp.MUTED)
    if df.size <= 1200:
        for (y, x), v in np.ndenumerate(df.values):
            if np.isfinite(v):
                ax.text(x, y, fmt.format(v), ha="center", va="center", fontsize=5, color=rp.INK)
    ax.set_title(title, loc="left", fontsize=10, color=rp.INK)
    fig.colorbar(im, ax=ax, shrink=0.5)
    fig.savefig(path, dpi=120, facecolor=rp.SURFACE, bbox_inches="tight")
    plt.close(fig)


def run(ctx: Ctx) -> dict:
    out = ctx.out("coverage")
    news = news_per_stock_year(ctx)
    share = signal_news_share(ctx)
    miss = missing_bars(ctx)
    news.to_csv(out / "news_per_stock_year.csv")
    share.to_csv(out / "signal_news_share_by_year.csv")
    miss.to_csv(out / "missing_bars_pct.csv")

    per_year = news.mean(axis=0)
    med = per_year.median()
    dev = per_year / med - 1.0
    flags = []
    for y, v in dev.items():
        if abs(v) > C.COVERAGE_DEVIATION:
            rec = ("EXCLUDE (news filter too weak: many 'no-news' signals will have had unseen news)"
                   if v < 0 else "keep, but check (unusually dense coverage, e.g. a new source was added)")
            flags.append({"year": int(y), "news_per_stock": float(per_year[y]), "vs_median": float(v),
                          "recommendation": rec})
    flag_df = pd.DataFrame(flags)
    flag_df.to_csv(out / "flagged_years.csv", index=False)

    # Charts
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=rp.SURFACE)
    for ax in (a1, a2):
        rp._style(ax)
    a1.bar(per_year.index.astype(str), per_year.values, color=rp.SERIES[0], width=0.6)
    a1.axhline(med, color=rp.MUTED, linewidth=1, linestyle="--")
    a1.axhspan(med * (1 - C.COVERAGE_DEVIATION), med * (1 + C.COVERAGE_DEVIATION), color="#f0efec", zorder=0)
    a1.set_title("Company news per stock per year (252-day scaled)\nband = median +-50%",
                 loc="left", fontsize=9, color=rp.INK)
    a2.bar(share.index.astype(str), 100 * share["news_share"], color=rp.SERIES[0], width=0.6)
    a2.set_title(f"% of |z| >= {C.Z_ENTRY} signals flagged as news", loc="left", fontsize=9, color=rp.INK)
    fig.tight_layout()
    fig.savefig(out / "news_coverage.png", dpi=130, facecolor=rp.SURFACE)
    plt.close(fig)
    _heatmap(news.round(0), "Company news per stock per year", out / "news_per_stock_year.png", fmt="{:.0f}")
    _heatmap(miss, "% of expected 5-min bars missing", out / "missing_bars.png", cmap="Reds")

    worst_miss = miss.stack().sort_values(ascending=False).head(10)
    lines = [
        f"data: {ctx.dataset}",
        "", "Company news per stock (mean over stocks, scaled to 252 days) by year:",
        fmt_table(pd.DataFrame({"news_per_stock": per_year, "vs_median": dev}).T),
        "", "Signals (|z| >= z_entry, entry rules 3 and 5) and share flagged as news by year:",
        fmt_table(share.T),
        "", f"Years with coverage more than {C.COVERAGE_DEVIATION:.0%} away from the median year:",
        fmt_table(flag_df) if len(flag_df) else "  none",
        "", f"Missing bars: median {np.nanmedian(miss.values):.2f}% of expected bars; worst stock-years:",
        fmt_table(worst_miss.to_frame("% missing")),
    ]
    res = {"news_per_stock_by_year": per_year.to_dict(), "news_share_by_year": share["news_share"].to_dict(),
           "flagged_years": flags, "missing_bars_median_pct": float(np.nanmedian(miss.values)),
           "missing_bars_max_pct": float(np.nanmax(miss.values))}
    write_summary(out, "COVERAGE - data audit", lines, res)
    return res
