"""
Lookahead-bias checklist. Most tests are perturbation tests: change data that
should be invisible at time t and assert nothing computed for time t moves.

    python -m pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest as bt  # noqa: E402
import config as C  # noqa: E402
import signals as sg  # noqa: E402
import synthetic  # noqa: E402

SMALL_MAP = {"AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "JPM": "XLF", "BAC": "XLF", "GS": "XLF"}
S = C.BARS_PER_DAY


@pytest.fixture(scope="module")
def data():
    bars, news, truth = synthetic.generate(n_days=60, sector_map=SMALL_MAP, seed=3)
    return bars, news


def build(bars, news):
    panel = sg.build_panel(bars, list(SMALL_MAP), SMALL_MAP)
    return panel, sg.compute_features(panel, news)


def perturb_from(bars: pd.DataFrame, cutoff_utc: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    """Randomly scramble prices and volumes of every bar starting at or after cutoff."""
    rng = np.random.default_rng(seed)
    out = bars.copy()
    m = out["timestamp"] >= cutoff_utc
    k = int(m.sum())
    factor = np.exp(rng.normal(0, 0.05, k))
    for col in ("open", "high", "low", "close", "vwap"):
        out.loc[m, col] = out.loc[m, col].to_numpy() * factor * np.exp(rng.normal(0, 0.01, k))
    out.loc[m, "volume"] = out.loc[m, "volume"].to_numpy() * rng.uniform(0.01, 5, k)
    return out


def ny(day: pd.Timestamp, hh: int, mm: int) -> pd.Timestamp:
    return (pd.Timestamp(day) + pd.Timedelta(hours=hh, minutes=mm)).tz_localize(C.TIMEZONE).tz_convert("UTC")


# ---------------------------------------------------------------------------
# 1. No fill uses the price of the bar that generated the signal
# ---------------------------------------------------------------------------
def test_fill_never_uses_signal_bar(data):
    bars, news = data
    panel, feats = build(bars, news)
    trades = bt.simulate_trades(panel, feats, sg.find_candidates(feats, 2.0))
    assert len(trades) > 100
    step = pd.Timedelta(minutes=C.BAR_MINUTES)
    # Entry happens at or after the signal bar's close, at the open of a LATER bar.
    assert (trades["entry_time"] >= trades["signal_time"]).all()
    entry_slot = ((trades["entry_time"] - trades["date"]) - pd.Timedelta(hours=9, minutes=30)) // step
    assert (entry_slot > trades["t"]).all()
    i, d = trades["i"].to_numpy(), trades["d"].to_numpy()
    np.testing.assert_allclose(trades["entry_px"], panel.open[i, d, entry_slot.to_numpy()])
    # Signal-driven exits fill after the decision bar's close; EOD fills at the 16:00 close.
    exit_close = trades["date"] + pd.Timedelta(hours=9, minutes=35) + trades["exit_bar"] * step
    assert (trades["exit_time"] >= exit_close).all()


def test_entry_price_unchanged_when_signal_bar_prices_change(data):
    bars, news = data
    panel, feats = build(bars, news)
    tr = bt.simulate_trades(panel, feats, sg.find_candidates(feats, 2.5))
    row = tr.iloc[len(tr) // 2]
    # Move the signal bar's open/close of that stock: the fill price must not depend on them.
    sym, start = row["symbol"], ny(row["date"], 9, 30) + pd.Timedelta(minutes=5 * int(row["t"]))
    b2 = bars.copy()
    m = (b2["symbol"] == sym) & (b2["timestamp"] == start)
    b2.loc[m, ["open", "close"]] *= 1.01
    panel2 = sg.build_panel(b2, list(SMALL_MAP), SMALL_MAP)
    assert panel2.next_open[row["i"], row["d"], row["t"] + 1] == pytest.approx(row["entry_px"])


# ---------------------------------------------------------------------------
# 2. β and σ at time t use only data up to t
# ---------------------------------------------------------------------------
def test_beta_sigma_z_use_only_past(data):
    bars, news = data
    panel, feats = build(bars, news)
    d, t = 40, 30
    day = panel.days[d]
    cutoff = ny(day, 9, 30) + pd.Timedelta(minutes=5 * (t + 1))   # everything after bar t's close
    panel2, feats2 = build(perturb_from(bars, cutoff), news)
    # β/σ for days <= d are unaffected by data from bar t+1 on day d onward.
    np.testing.assert_array_equal(feats.beta[:, : d + 1], feats2.beta[:, : d + 1])
    np.testing.assert_array_equal(feats.sigma[:, : d + 1], feats2.sigma[:, : d + 1])
    np.testing.assert_array_equal(feats.z[:, :d], feats2.z[:, :d])
    np.testing.assert_array_equal(feats.z[:, d, : t + 1], feats2.z[:, d, : t + 1])
    np.testing.assert_array_equal(feats.z_sector[:, d, : t + 1], feats2.z_sector[:, d, : t + 1])
    # ...and the perturbation did change later values (the test has teeth).
    assert not np.allclose(feats.beta[:, d + 1:], feats2.beta[:, d + 1:], equal_nan=True)
    assert not np.allclose(feats.z[:, d, t + 1:], feats2.z[:, d, t + 1:], equal_nan=True)


def test_beta_for_day_excludes_same_day(data):
    bars, news = data
    panel, feats = build(bars, news)
    d = 40
    panel2, feats2 = build(perturb_from(bars, ny(panel.days[d], 9, 30)), news)
    np.testing.assert_array_equal(feats.beta[:, : d + 1], feats2.beta[:, : d + 1])
    np.testing.assert_array_equal(feats.sigma[:, : d + 1], feats2.sigma[:, : d + 1])


# ---------------------------------------------------------------------------
# 3. News counted at bar t has created_at <= bar t's close time
# ---------------------------------------------------------------------------
def test_news_cutoff_is_bar_close_and_uses_created_at(data):
    bars, _ = data
    panel = sg.build_panel(bars, list(SMALL_MAP), SMALL_MAP)
    d, t = 30, 20
    day, prev_day = panel.days[d], panel.days[d - 1]
    close_t = ny(day, 9, 35) + pd.Timedelta(minutes=5 * t)
    ns = pd.Timedelta(1, "ns")
    rows = [
        # (created_at, updated_at, symbols)
        (close_t, close_t, ["AAPL"]),                                  # exactly at close of t: counts at t
        (close_t + ns, close_t + ns, ["MSFT"]),                        # 1ns later: only from t+1
        (ny(prev_day, 11, 0), close_t - pd.Timedelta(minutes=10), ["NVDA"]),  # old, updated today: never
        (close_t - pd.Timedelta(minutes=10),) * 2 + (["JPM", "BAC", "GS", "AAPL"],),  # 4 tags: not company news
        (ny(prev_day, 16, 0), ny(prev_day, 16, 0), ["BAC"]),          # at previous close: belongs to prev day
        (ny(prev_day, 16, 0) + ns, ny(prev_day, 16, 0) + ns, ["GS"]),  # just after previous close: counts
    ]
    news = pd.DataFrame(rows, columns=["created_at", "updated_at", "symbols"])
    cnt = sg.news_counts(panel, sg.company_news_times(news, panel.stocks))
    ix = {s: i for i, s in enumerate(panel.stocks)}
    assert cnt[ix["AAPL"], d, t] == 1 and cnt[ix["AAPL"], d, t - 1] == 0
    assert cnt[ix["MSFT"], d, t] == 0 and cnt[ix["MSFT"], d, t + 1] == 1
    assert cnt[ix["NVDA"], d].max() == 0                 # its update today is ignored...
    assert cnt[ix["NVDA"], d - 1, -1] == 1               # ...it counts on the day it was created
    assert cnt[ix["JPM"], d].max() == 0
    assert cnt[ix["BAC"], d].max() == 0 and cnt[ix["BAC"], d - 1, -1] == 1
    assert cnt[ix["GS"], d, 0] == 1


def test_signal_news_flag_ignores_future_news(data):
    bars, news = data
    panel, feats = build(bars, news)
    tr = bt.simulate_trades(panel, feats, sg.find_candidates(feats, 2.5, news="no_news"))
    row = tr.iloc[0]
    # Add company news one second after the signal bar's close: the signal must stay no-news,
    # and the open position must be stopped out by the news exit.
    t_news = pd.Timestamp(row["signal_time"]).tz_localize(C.TIMEZONE).tz_convert("UTC") + pd.Timedelta(seconds=1)
    extra = pd.DataFrame({"id": [-1], "headline": ["x"], "created_at": [t_news], "updated_at": [t_news],
                          "symbols": [[row["symbol"]]], "source": ["test"]})
    feats2 = sg.compute_features(panel, pd.concat([news, extra], ignore_index=True))
    assert not feats2.has_news[row["i"], row["d"], row["t"]]
    assert feats2.has_news[row["i"], row["d"], row["t"] + 1]
    c = pd.DataFrame([{"i": row["i"], "d": row["d"], "t": row["t"], "z": row["z_signal"], "has_news": False}])
    tr2 = bt.simulate_trades(panel, feats2, c)
    assert tr2.iloc[0]["exit_reason"] in ("news", "reversion", "stop")  # first bar: news unless another fired
    assert tr2.iloc[0]["exit_bar"] == row["t"] + 1


# ---------------------------------------------------------------------------
# 4. Lookback windows never cross into the previous day
# ---------------------------------------------------------------------------
def test_intraday_window_sum_stays_within_day():
    x = np.arange(2 * S, dtype=float).reshape(1, 2, S) + 1.0
    L = 6
    out = sg.intraday_window_sum(x, L)
    assert np.isnan(out[0, 1, : L - 1]).all()            # not enough same-day bars yet
    assert out[0, 1, L - 1] == x[0, 1, :L].sum()
    assert out[0, 1, 40] == x[0, 1, 35:41].sum()
    x2 = x.copy()
    x2[0, 0] *= 1000                                     # previous day changes...
    np.testing.assert_array_equal(sg.intraday_window_sum(x2, L)[0, 1], out[0, 1])  # ...nothing today


def test_returns_exclude_overnight_gap(data):
    bars, news = data
    panel = sg.build_panel(bars, list(SMALL_MAP), SMALL_MAP)
    d = 30
    # Scale the ENTIRE previous day (and earlier) by 2x: a huge overnight gap into day d.
    b2 = bars.copy()
    m = b2["timestamp"] < ny(panel.days[d], 9, 30)
    b2.loc[m, ["open", "high", "low", "close", "vwap"]] *= 2.0
    panel2 = sg.build_panel(b2, list(SMALL_MAP), SMALL_MAP)
    r1, r2 = sg.intraday_log_returns(panel), sg.intraday_log_returns(panel2)
    np.testing.assert_allclose(r1[:, d], r2[:, d])


# ---------------------------------------------------------------------------
# 5. Universe filters on day d use only data before day d
# ---------------------------------------------------------------------------
def test_universe_filter_uses_only_prior_days(data):
    bars, news = data
    panel = sg.build_panel(bars, list(SMALL_MAP), SMALL_MAP)
    d = 40
    b2 = bars.copy()
    m = (b2["timestamp"] >= ny(panel.days[d], 9, 30)) & (b2["symbol"] == "AAPL")
    b2.loc[m, ["open", "high", "low", "close", "vwap"]] = 5.0     # below MIN_PRICE from day d on
    b2.loc[m, "volume"] = 1.0                                      # and illiquid
    panel2 = sg.build_panel(b2, list(SMALL_MAP), SMALL_MAP)
    u1, u2 = sg.universe_mask(panel), sg.universe_mask(panel2)
    a = panel.stocks.index("AAPL")
    assert u1[a, d] and u2[a, d]                         # day d itself still uses day d-1 data
    assert not u2[a, d + 1]                              # the change is seen from day d+1
    np.testing.assert_array_equal(u1[:, : d + 1], u2[:, : d + 1])
    np.testing.assert_array_equal(sg.avg_bar_dollar_volume(panel)[:, : d + 1],
                                  sg.avg_bar_dollar_volume(panel2)[:, : d + 1])


# ---------------------------------------------------------------------------
# End-to-end: synthetic data must show the planted effect
# ---------------------------------------------------------------------------
def test_synthetic_recovers_planted_effect():
    bars, news, _ = synthetic.generate(n_days=120, seed=11)
    panel = sg.build_panel(bars, C.STOCKS, C.SECTOR_MAP)
    feats = sg.compute_features(panel, news)
    res = bt.run_backtest(panel, feats, sweep=False)
    tbl = res.study_tables["all"]
    no_news = tbl.loc[("first_per_stock_day", "no_news", "1x")]
    with_news = tbl.loc[("first_per_stock_day", "news", "1x")]
    assert no_news["mean_net_bps"] > 5 and no_news["t_stat_day_clustered"] > 3
    assert with_news["mean_net_bps"] < 0
    oos = res.portfolio_tables["out_of_sample"]["1x cost"]
    assert oos["total_pnl"] > 0 and oos["sharpe"] > 1
