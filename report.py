"""
Write backtest outputs to results/:
    trades.csv               portfolio trades (no-news strategy, with position limits)
    signal_study_trades.csv  every qualifying bar's simulated fade trade (news and no-news)
    summary.txt              all tables (signal study, portfolio, IS/OOS, breakdowns, sweep)
    equity.png               equity curve at 1x / 2x cost with the IS/OOS split, plus drawdown
    sweep.csv                in-sample parameter grid
    sweep_sharpe.png, sweep_net_bps.png   heatmaps of the grid
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import config as C  # noqa: E402
from backtest import Results, daily_pnl  # noqa: E402

# Palette (light chart surface): categorical blue/orange, red<->gray<->blue diverging.
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

TRADE_COLUMNS = ["period", "symbol", "etf", "date", "signal_time", "entry_time", "exit_time", "direction",
                 "z_signal", "z_at_exit", "z_sector", "beta", "sigma", "has_news", "entry_px", "exit_px",
                 "etf_entry_px", "etf_exit_px", "exit_reason", "bars_held", "hold_min",
                 "gross_bps", "cost_bps", "net_bps"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _label_period(tr: pd.DataFrame, periods: dict) -> pd.DataFrame:
    tr = tr.copy()
    tr["period"] = np.where(tr["date"].isin(periods["in_sample"]), "in_sample",
                            np.where(tr["date"].isin(periods["out_of_sample"]), "out_of_sample", ""))
    return tr


def plot_equity(res: Results, path: Path) -> None:
    days = res.periods["all"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE)
    oos_start = res.periods["out_of_sample"][0] if len(res.periods["out_of_sample"]) else None
    for ax in (ax1, ax2):
        _style(ax)
        if oos_start is not None:
            ax.axvspan(oos_start, days[-1], color="#f0efec", zorder=0)
    for color, m in zip(SERIES, C.COST_MULTIPLIERS):
        eq = C.CAPITAL + daily_pnl(res.portfolio, days, f"pnl_{m:g}x").cumsum()
        ax1.plot(eq.index, eq.values, color=color, linewidth=2, label=f"{m:g}x cost")
        ax1.annotate(f"{m:g}x cost", (eq.index[-1], eq.values[-1]), xytext=(4, 0),
                     textcoords="offset points", color=INK_2, fontsize=8, va="center")
        dd = eq - eq.cummax().clip(lower=C.CAPITAL)
        ax2.plot(dd.index, dd.values, color=color, linewidth=1.5)
    if oos_start is not None:
        ax1.text(oos_start, 1.0, "  out-of-sample", transform=ax1.get_xaxis_transform(),
                 color=MUTED, fontsize=8, va="top")
    ax1.axhline(C.CAPITAL, color="#c3c2b7", linewidth=0.8)
    ax1.set_title("No-news reversal portfolio: equity ($)", loc="left", color=INK, fontsize=11)
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    ax2.set_title("Drawdown ($)", loc="left", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_sweep(sweep: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    Ls, zxs = C.SWEEP_GRID["lookback_bars"], C.SWEEP_GRID["z_exit"]
    vals = sweep[metric].to_numpy(dtype=float)
    lim = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim - 1e-9, vmax=lim + 1e-9)
    fig, axes = plt.subplots(len(Ls), len(zxs), figsize=(3.3 * len(zxs), 2.7 * len(Ls)),
                             facecolor=SURFACE, squeeze=False)
    for r, L in enumerate(Ls):
        for c, zx in enumerate(zxs):
            ax = axes[r, c]
            sub = sweep[(sweep.lookback_bars == L) & (sweep.z_exit == zx)]
            grid = sub.pivot(index="z_entry", columns="hold_bars", values=metric)
            im = ax.imshow(grid.values, cmap=DIVERGING, norm=norm, aspect="auto")
            ax.set_xticks(range(grid.shape[1]), [f"H={h}" for h in grid.columns], fontsize=7, color=MUTED)
            ax.set_yticks(range(grid.shape[0]), [f"z={z:g}" for z in grid.index], fontsize=7, color=MUTED)
            for (y, x), v in np.ndenumerate(grid.values):
                ax.text(x, y, "-" if not np.isfinite(v) else f"{v:.1f}", ha="center", va="center",
                        fontsize=7, color=INK)
            ax.set_title(f"L={L}, z_exit={zx:g}", fontsize=8, color=INK_2)
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.subplots_adjust(wspace=0.4, hspace=0.45)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label=metric)
    fig.savefig(path, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda x: f"{x:,.3f}")


def summary_text(res: Results, panel, synthetic: bool) -> str:
    out = io.StringIO()
    w = lambda s="": print(s, file=out)  # noqa: E731
    w("NEWS-FILTERED INTRADAY REVERSAL BACKTEST")
    w("=" * 72)
    w(f"data: {'SYNTHETIC (planted effect)' if synthetic else 'Alpaca cache'}; "
      f"{panel.n_stocks} stocks, {len(panel.days)} days {panel.days[0].date()} .. {panel.days[-1].date()}")
    for k, v in res.periods.items():
        if len(v):
            w(f"  {k:14s} {len(v):5d} tradable days  {v[0].date()} .. {v[-1].date()}")
    w(f"params: L={C.LOOKBACK_BARS} z_entry={C.Z_ENTRY} z_exit={C.Z_EXIT} H={C.HOLD_BARS} "
      f"stop=+{C.STOP_Z_OFFSET} ({C.STOP_Z_REFERENCE}) sector_filter={C.USE_SECTOR_FILTER} "
      f"hedge={C.USE_HEDGE} cost={C.COST_BPS_PER_SIDE}bps/side/leg sizing={C.SIZING} "
      f"max_positions={C.MAX_POSITIONS}")
    w()
    w("1. SIGNAL STUDY (every bar with |z|>=z_entry, time window, sector filter; news or not)")
    w("   expected if the hypothesis holds: no_news net > 0, news <= 0")
    w("   'every_bar' trades overlap (same shock, consecutive bars): use the day-clustered t-stat")
    for p, tbl in res.study_tables.items():
        w(f"\n-- {p} --")
        w(_fmt(tbl))
    w("\nSignal study by |z| bucket (all dates, 1x cost):")
    for grp, tbl in res.breakdowns.get("signal_study_abs_z", {}).items():
        w(f"  [{grp}]")
        w(_fmt(tbl))
    w()
    w("2/4. PORTFOLIO BACKTEST (no-news trades, position limits) - OUT-OF-SAMPLE IS THE RESULT THAT COUNTS")
    for p, tbl in res.portfolio_tables.items():
        w(f"\n-- {p} --")
        w(_fmt(tbl))
    w()
    w("3. BREAKDOWNS (portfolio trades, 1x cost)")
    for p, tbls in res.breakdowns.items():
        if p == "signal_study_abs_z":
            continue
        for name, tbl in tbls.items():
            w(f"\n-- {p} / by {name} --")
            w(_fmt(tbl))
    if res.sweep is not None and len(res.sweep):
        w()
        w("5. IN-SAMPLE PARAMETER SWEEP (full grid in sweep.csv / heatmaps)")
        sw = res.sweep
        w(f"   share of grid points with mean net bps > 0: {(sw.mean_net_bps > 0).mean():.0%}; "
          f"with Sharpe > 0: {(sw.sharpe > 0).mean():.0%}")
        for col in ("lookback_bars", "z_entry", "z_exit", "hold_bars"):
            w(f"\n   median over the rest of the grid, by {col}:")
            w(_fmt(sw.groupby(col)[["mean_net_bps", "sharpe", "trades"]].median()))
        w("\n   top 10 by Sharpe (look for a broad region, not the single best point):")
        w(_fmt(sw.sort_values("sharpe", ascending=False).head(10)))
    return out.getvalue()


def write_all(res: Results, panel, out_dir, synthetic: bool = False) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _label_period(res.portfolio, res.periods).reindex(
        columns=TRADE_COLUMNS + ["notional", "hedge_notional"] + [f"pnl_{m:g}x" for m in C.COST_MULTIPLIERS]
    ).to_csv(out / "trades.csv", index=False)
    _label_period(res.study, res.periods).reindex(columns=TRADE_COLUMNS + ["group"]).to_csv(
        out / "signal_study_trades.csv", index=False)
    text = summary_text(res, panel, synthetic)
    (out / "summary.txt").write_text(text)
    plot_equity(res, out / "equity.png")
    if res.sweep is not None:
        res.sweep.to_csv(out / "sweep.csv", index=False)
        plot_sweep(res.sweep, "sharpe", "In-sample Sharpe by parameter (rows: L, cols: z_exit)",
                   out / "sweep_sharpe.png")
        plot_sweep(res.sweep, "mean_net_bps", "In-sample mean net bps/trade (rows: L, cols: z_exit)",
                   out / "sweep_net_bps.png")
    print(text)
