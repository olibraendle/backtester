"""
`synthetic`: controlled-truth markets from a generator fitted to the data.

Generator (fitted on the last SCENARIO_FIT_DAYS of the development period):
  * intraday volatility seasonality per slot (stocks and ETFs separately)
  * market factor -> sector ETFs (loading + own residual), each GARCH(1,1) with Student-t shocks
  * stock = beta x ETF + residual, residual GARCH(1,1)-t per stock
  * company news arrives at the fitted per-stock rate and time-of-day distribution; news
    during the session adds a persistent jump (no reversal) sized from the data
Scenarios (SCENARIO_RUNS each):
  (a) no edge, (b) planted edge: no-news shocks that partly revert, (c) edge only in the first half.
Reports the false-positive rate in (a), the detection rate in (b) and, in (c), how long the
kill criteria (see `risk`) take to fire after the edge disappears.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, signal
from scipy import stats as sps

import backtest as bt
import config as C
import report as rp
import signals as sg
from validation import common as cm
from validation import risk
from validation.common import Ctx, fmt_table, summarize, write_summary

S = C.BARS_PER_DAY


# ---------------------------------------------------------------------------
# GARCH(1,1) with standardized Student-t innovations
# ---------------------------------------------------------------------------
def _garch_var(x2: np.ndarray, omega: float, alpha: float, beta: float, v0: float) -> np.ndarray:
    """sigma2_t = omega + alpha * x_{t-1}^2 + beta * sigma2_{t-1}, via a linear filter."""
    drive = np.empty_like(x2)
    drive[0] = v0                                       # sigma2_0 = v0
    drive[1:] = omega + alpha * x2[:-1]
    return signal.lfilter([1.0], [1.0, -beta], drive)   # y_t = drive_t + beta * y_{t-1}


def fit_garch_t(x: np.ndarray) -> dict:
    """MLE of (alpha, beta, nu) with variance targeting on a standardized series."""
    x = x[np.isfinite(x)]
    var = x.var()
    x2 = x * x

    def nll(theta):
        a, b, nu = theta
        if a <= 0 or b <= 0 or a + b >= 0.999 or nu <= 2.05:
            return 1e12
        s2 = _garch_var(x2, var * (1 - a - b), a, b, var)
        s2 = np.maximum(s2, 1e-12)
        scale = np.sqrt(s2 * (nu - 2) / nu)
        return -np.sum(sps.t.logpdf(x / scale, nu) - np.log(scale))

    best = None
    for start in ([0.05, 0.90, 6.0], [0.10, 0.80, 4.0]):
        r = optimize.minimize(nll, start, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-3})
        if best is None or r.fun < best.fun:
            best = r
    a, b, nu = best.x
    return {"var": float(var), "alpha": float(a), "beta": float(b), "nu": float(nu)}


@dataclass
class GenFit:
    season_stock: np.ndarray      # (S,)
    season_etf: np.ndarray        # (S,)
    market: dict                  # GARCH params of the market factor (in seasonality-adjusted units)
    etf_load: np.ndarray          # (E,)
    etf_garch: list[dict]
    beta: np.ndarray              # (N,)
    stock_garch: list[dict]
    news_rate: np.ndarray         # (N,) company-news items per calendar day
    news_tod_min: np.ndarray      # pool of time-of-day (minutes after midnight NY)
    jump_scale: float             # news-bar jump std in residual-sigma units
    stocks: list[str]
    etfs: list[str]
    etf_of: np.ndarray            # (N,) index into etfs


def fit_generator(ctx: Ctx) -> GenFit:
    p, f = ctx.panel, ctx.feats
    dev = np.nonzero(ctx.mask("dev"))[0][-C.SCENARIO_FIT_DAYS:]
    N = p.n_stocks
    etf_rows = np.unique(p.etf_idx)
    eps = f.eps[:, dev].astype(float)
    re = f.ret[etf_rows][:, dev].astype(float)
    sig = f.sigma[:, dev][..., None]
    z_eps = eps / sig                                   # residual in units of its 20-day sigma
    season_stock = np.sqrt(np.nanmean(z_eps ** 2, axis=(0, 1)))
    season_stock /= np.sqrt(np.nanmean(season_stock ** 2))
    season_etf = np.sqrt(np.nanmean(re ** 2, axis=(0, 1)))
    season_etf /= np.sqrt(np.nanmean(season_etf ** 2))

    re_ds = re / season_etf                             # deseasonalized ETF returns
    mkt = np.nanmean(re_ds, axis=0)                     # (D, S) market factor
    load, etf_garch = [], []
    for k in range(len(etf_rows)):
        y, x = re_ds[k].ravel(), mkt.ravel()
        ok = np.isfinite(y) & np.isfinite(x)
        lam = float(np.dot(x[ok], y[ok]) / np.dot(x[ok], x[ok]))
        load.append(lam)
        etf_garch.append(fit_garch_t(y[ok] - lam * x[ok]))
    market = fit_garch_t(mkt.ravel())

    stock_garch = [fit_garch_t((eps[i] / season_stock).ravel()) for i in range(N)]
    beta = np.nan_to_num(np.nanmedian(f.beta[:, dev], axis=1), nan=1.0)

    # News: rate and time of day from the data; jump size from residuals at news-arrival bars.
    times = sg.company_news_times(ctx.news, p.stocks)
    span_days = max((p.days[dev[-1]] - p.days[dev[0]]).days, 1)
    t0 = pd.Timestamp(p.days[dev[0]]).tz_localize(C.TIMEZONE)
    t1 = pd.Timestamp(p.days[dev[-1]] + pd.Timedelta(days=1)).tz_localize(C.TIMEZONE)
    rates, tods = [], []
    for sym in p.stocks:
        ts = pd.DatetimeIndex(pd.to_datetime(times[sym], utc=True)).tz_convert(C.TIMEZONE)
        ts = ts[(ts >= t0) & (ts < t1)]
        rates.append(len(ts) / span_days)
        tods.append((ts.hour * 60 + ts.minute + ts.second / 60).to_numpy())
    tod = np.concatenate(tods) if tods else np.array([600.0])
    arrive = np.diff(f.news_cnt[:, dev].astype(int), axis=-1, prepend=0) > 0
    zs = z_eps / season_stock
    v_news, v_other = np.nanmean(zs[arrive] ** 2), np.nanmean(zs[~arrive] ** 2)
    jump = float(np.sqrt(max(v_news - v_other, 1.0))) if np.isfinite(v_news) else 3.0

    return GenFit(season_stock=season_stock, season_etf=season_etf, market=market, etf_load=np.array(load),
                  etf_garch=etf_garch, beta=beta, stock_garch=stock_garch, news_rate=np.array(rates),
                  news_tod_min=tod if len(tod) else np.array([600.0]), jump_scale=jump,
                  stocks=list(p.stocks), etfs=[p.symbols[k] for k in etf_rows],
                  etf_of=np.searchsorted(etf_rows, p.etf_idx))


def _simulate_garch(params: list[dict], T: int, rng) -> np.ndarray:
    """(K, T) GARCH(1,1)-t paths, vectorised across series, unit-variance-ish innovations."""
    K = len(params)
    var = np.array([q["var"] for q in params])
    a = np.array([q["alpha"] for q in params])
    b = np.array([q["beta"] for q in params])
    nu = np.array([q["nu"] for q in params])
    om = var * (1 - a - b)
    z = rng.standard_t(nu[:, None], size=(K, T)) * np.sqrt((nu[:, None] - 2) / nu[:, None])
    out = np.empty((K, T))
    s2, x_prev = var.copy(), np.zeros(K)
    for t in range(T):
        s2 = om + a * x_prev ** 2 + b * s2
        x_prev = np.sqrt(s2) * z[:, t]
        out[:, t] = x_prev
    return out


def simulate_market(fit: GenFit, n_days: int, scenario: str, rng) -> tuple[sg.Panel, pd.DataFrame]:
    """Generate one market. scenario: "a" no edge, "b" planted edge, "c" edge in the first half only."""
    N, E, D = len(fit.stocks), len(fit.etfs), n_days
    T = D * S
    mkt = _simulate_garch([fit.market], T, rng)[0]
    etf = fit.etf_load[:, None] * mkt[None, :] + _simulate_garch(fit.etf_garch, T, rng)
    r_etf = etf.reshape(E, D, S) * fit.season_etf
    resid = _simulate_garch(fit.stock_garch, T, rng).reshape(N, D, S) * fit.season_stock
    sig_i = np.sqrt([q["var"] for q in fit.stock_garch])
    days = pd.bdate_range("2010-01-04", periods=D)

    # News: Poisson arrivals at the fitted rate, time of day from the empirical pool.
    news_rows = []
    news_day = np.zeros((N, D), bool)
    cal_days = np.r_[1, np.diff(days.values).astype("timedelta64[D]").astype(int)]   # weekends add days
    for i in range(N):
        k = rng.poisson(fit.news_rate[i] * cal_days)
        for d in np.nonzero(k)[0]:
            for _ in range(k[d]):
                tod = float(rng.choice(fit.news_tod_min))
                news_rows.append((days[d] + pd.Timedelta(minutes=tod), fit.stocks[i]))
                slot = int((tod - 570) // C.BAR_MINUTES)
                if 0 <= slot < S:                       # in-session: persistent jump at that bar
                    resid[i, d, slot] += rng.standard_t(3) / np.sqrt(3) * fit.jump_scale * sig_i[i] \
                        * fit.season_stock[slot]
                if tod < 960:
                    news_day[i, d] = True

    # Planted edge: shocks on no-news stock-days that partially revert.
    if scenario in ("b", "c"):
        decay = np.exp(-np.arange(8) / 2.5)
        decay /= decay.sum()
        last_day = D if scenario == "b" else D // 2
        hit = (rng.random((N, last_day)) < C.SCENARIO_SHOCK_PROB) & ~news_day[:, :last_day]
        for i, d in zip(*np.nonzero(hit)):
            s0, w = int(rng.integers(4, 58)), int(rng.integers(1, 3))
            size = rng.uniform(*C.SCENARIO_SHOCK_SIGMA) * sig_i[i] * rng.choice([-1, 1])
            resid[i, d, s0:s0 + w] += size / w
            hi = min(s0 + w + 8, S)
            resid[i, d, s0 + w:hi] -= rng.uniform(*C.SCENARIO_REVERT_FRAC) * size * decay[: hi - s0 - w]

    r_stock = fit.beta[:, None, None] * r_etf[fit.etf_of] + resid
    rets = np.concatenate([r_stock, r_etf]).astype(np.float64)
    M = N + E
    # Prices: each day opens at the previous close +- an overnight gap; 1 bp bid-ask bounce.
    gaps = rng.normal(0, 0.01, (M, D))
    gaps[:, 0] = 0
    prev_day_move = np.zeros((M, D))
    prev_day_move[:, 1:] = rets.sum(-1)[:, :-1]
    day_open = np.log(100.0) + np.cumsum(gaps + prev_day_move, axis=1)
    log_c = day_open[..., None] + np.cumsum(rets, axis=-1)
    log_o = np.empty_like(log_c)
    log_o[..., 0] = day_open
    log_o[..., 1:] = log_c[..., :-1]
    hs = 1e-4
    log_c += hs * rng.choice([-1.0, 1.0], log_c.shape)
    log_o += hs * rng.choice([-1.0, 1.0], log_o.shape)
    o, c = np.exp(log_o).astype(sg.F), np.exp(log_c).astype(sg.F)
    panel = sg.make_panel(fit.stocks + fit.etfs, N, N + fit.etf_of, days, o, np.fmax(o, c), np.fmin(o, c), c,
                          np.full((M, D), 5e9))
    ts = pd.DatetimeIndex([t for t, _ in news_rows]).tz_localize(
        C.TIMEZONE, nonexistent="shift_forward", ambiguous="NaT").tz_convert("UTC")
    news = pd.DataFrame({"created_at": ts, "symbols": [[s] for _, s in news_rows]}).dropna()
    return panel, news


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def _run_one(task) -> dict:
    scenario, k = task
    sh = cm.SHARED
    rng = cm.rng_for(sh["seed"], "scenario", scenario, k)
    panel, news = simulate_market(sh["fit"], C.SCENARIO_DAYS, scenario, rng)
    feats = sg.compute_features(panel, news, C.LOOKBACK_BARS)
    days = panel.days[feats.tradable.any(axis=0)]
    port = bt.run_strategy(panel, feats)
    port = port[port["date"].isin(days)]
    m = summarize(port, days)
    half = days[len(days) // 2]
    first, second = days[days < half], days[days >= half]
    m1, m2 = summarize(port, first), summarize(port, second)
    rec = {"scenario": scenario, "run": k, **{f"all_{a}": b for a, b in m.items()},
           **{f"h1_{a}": b for a, b in m1.items()}, **{f"h2_{a}": b for a, b in m2.items()},
           "detected": bool(m["t_stat"] > C.T_STAT_PASS and m["mean_net_bps"] > 0)}
    # Kill criteria fitted on the first half, monitored live over the second half.
    if scenario in ("b", "c"):
        rec.update(risk.monitor_kill(port, first, second, cm.rng_for(sh["seed"], "kill", scenario, k)))
    return rec


def run(ctx: Ctx) -> dict:
    out = ctx.out("synthetic")
    runs = ctx.opt("scenario_runs", C.SCENARIO_RUNS)
    print("fitting generator ...", flush=True)
    fit = fit_generator(ctx)
    fit_tbl = pd.DataFrame({"alpha": [q["alpha"] for q in fit.stock_garch], "beta": [q["beta"] for q in fit.stock_garch],
                            "nu": [q["nu"] for q in fit.stock_garch], "stock_beta": fit.beta,
                            "news_per_day": fit.news_rate}, index=fit.stocks)
    fit_tbl.to_csv(out / "generator_fit_stocks.csv")
    cm.SHARED.update(fit=fit, seed=ctx.seed)
    tasks = [(s, k) for s in ("a", "b", "c") for k in range(runs)]
    recs = cm.pmap(_run_one, tasks, ctx.workers, "scenario runs")
    cm.SHARED.clear()
    df = pd.DataFrame(recs)
    df.to_csv(out / "scenario_runs.csv", index=False)

    a, b, c = (df[df["scenario"] == s] for s in "abc")
    res = {"runs_per_scenario": runs,
           "fit": {"garch_alpha_median": float(fit_tbl["alpha"].median()), "garch_beta_median": float(fit_tbl["beta"].median()),
                   "t_nu_median": float(fit_tbl["nu"].median()), "market": fit.market, "jump_scale_sigma": fit.jump_scale},
           "a_false_positive_rate": float(a["detected"].mean()),
           "a_mean_net_bps": float(a["all_mean_net_bps"].mean()),
           "b_detection_rate": float(b["detected"].mean()),
           "b_mean_net_bps": float(b["all_mean_net_bps"].mean()),
           "b_false_kill_rate": float(b["killed_mean"].mean()) if "killed_mean" in b else None,
           "c_killed_rate_mean_rule": float(c["killed_mean"].mean()),
           "c_killed_rate_dd_rule": float(c["killed_dd"].mean()),
           "c_median_trades_to_kill": float(c["kill_trades"].median()),
           "c_median_days_to_kill_mean_rule": float(c["kill_days_mean"].median()),
           "c_median_days_to_kill_dd_rule": float(c["kill_days_dd"].median()),
           "c_median_N_kill": float(c["N_kill"].median())}
    res["pass"] = bool(res["a_false_positive_rate"] <= 0.10 and res["b_detection_rate"] >= 0.80)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=rp.SURFACE)
    for ax, (s, g), title in zip(axes, (("a", a), ("b", b)), ("(a) no edge", "(b) planted edge")):
        rp._style(ax)
        ax.hist(g["all_t_stat"].dropna(), bins=30, color=rp.SERIES[0])
        ax.axvline(C.T_STAT_PASS, color=rp.SERIES[1], linewidth=2)
        ax.set_title(f"{title}: t-stat of net bps\n{g['detected'].mean():.0%} of runs pass t > {C.T_STAT_PASS:g}",
                     loc="left", fontsize=9, color=rp.INK)
    ax = axes[2]
    rp._style(ax)
    ax.hist(c["kill_days_mean"].dropna(), bins=30, color=rp.SERIES[0], alpha=0.85, label="mean-net rule")
    ax.hist(c["kill_days_dd"].dropna(), bins=30, color=rp.SERIES[1], alpha=0.6, label="drawdown rule")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(c) edge vanishes at half-time:\ntrading days until the kill rule fires", loc="left",
                 fontsize=9, color=rp.INK)
    fig.tight_layout()
    fig.savefig(out / "scenarios.png", dpi=130, facecolor=rp.SURFACE)
    plt.close(fig)

    cols = ["all_n_trades", "all_mean_net_bps", "all_t_stat", "all_sharpe", "detected"]
    lines = [f"generator fitted on the last {C.SCENARIO_FIT_DAYS} development days of {ctx.dataset} data; "
             f"{runs} runs x 3 scenarios, {C.SCENARIO_DAYS} days each",
             f"fit: median stock GARCH alpha {res['fit']['garch_alpha_median']:.3f}, beta {res['fit']['garch_beta_median']:.3f}, "
             f"t nu {res['fit']['t_nu_median']:.1f}; news jump {fit.jump_scale:.1f} sigma",
             "", "mean over runs:", fmt_table(df.groupby("scenario")[cols].mean()),
             "", f"(a) no edge     : false-positive rate (t > {C.T_STAT_PASS:g}) = {res['a_false_positive_rate']:.1%}  "
             f"(mean net {res['a_mean_net_bps']:+.2f} bps)",
             f"(b) planted edge: detection rate = {res['b_detection_rate']:.1%}  (mean net {res['b_mean_net_bps']:+.2f} bps); "
             f"kill rule fired anyway in {res['b_false_kill_rate']:.1%} of runs (false kills)",
             f"(c) edge vanishes halfway: mean-net rule fired in {res['c_killed_rate_mean_rule']:.0%} of runs, median "
             f"{res['c_median_trades_to_kill']:.0f} trades / {res['c_median_days_to_kill_mean_rule']:.0f} days after the "
             f"change (N_kill median {res['c_median_N_kill']:.0f}); drawdown rule fired in "
             f"{res['c_killed_rate_dd_rule']:.0%}, median {res['c_median_days_to_kill_dd_rule']:.0f} days",
             "", f"{'PASS' if res['pass'] else 'FAIL'}: need false positives <= 10% and detection >= 80%"]
    write_summary(out, "SYNTHETIC - controlled-truth markets", lines, res)
    return res
