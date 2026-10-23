"""Tests for the validation machinery (holdout lock, permutation, fill modes, statistics helpers)."""
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
from validation import common as cm  # noqa: E402
from validation import permutation as pm  # noqa: E402
from validation import risk  # noqa: E402
from validation import stats as vs  # noqa: E402
from validation import trial_log  # noqa: E402

SMALL_MAP = {"AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "JPM": "XLF", "BAC": "XLF", "GS": "XLF"}


@pytest.fixture(scope="module")
def small():
    bars, news, _ = synthetic.generate(n_days=70, sector_map=SMALL_MAP, seed=5)
    panel = sg.build_panel(bars, list(SMALL_MAP), SMALL_MAP)
    return panel, sg.compute_features(panel, news), news


def test_rng_streams_are_reproducible_and_distinct():
    a = cm.rng_for(42, "perm", 3).random(5)
    np.testing.assert_array_equal(a, cm.rng_for(42, "perm", 3).random(5))
    assert not np.allclose(a, cm.rng_for(42, "perm", 4).random(5))


def test_holdout_is_cut_and_periods_exclude_it():
    cal = cm.synthetic_calendar()
    h0 = cm.holdout_start(cal, True)
    tradable = np.ones(len(cal), bool)
    locked = cm.make_periods(cal, tradable, True, unlocked=False)
    assert "holdout" not in locked and locked["wf_oos"].max() < h0 and locked["dev"].max() < locked["wf_oos"].min()
    unlocked = cm.make_periods(cal, tradable, True, unlocked=True)
    assert unlocked["holdout"].min() >= h0
    days = pd.bdate_range("2016-01-04", "2026-10-01")
    real = cm.make_periods(days, np.ones(len(days), bool), False, unlocked=False)
    assert real["dev"].max() <= pd.Timestamp(C.DEV_END)
    assert real["wf_oos"].min() >= pd.Timestamp(C.WF_OOS_START)
    assert real["wf_oos"].max() < pd.Timestamp(C.HOLDOUT_START)


def test_permutation_identity_rebuilds_prices(small):
    panel, feats, _ = small

    class Identity:
        def random(self, shape):
            return np.broadcast_to(np.arange(shape[-1], dtype=float), shape)

    p2 = pm.permute_panel(panel, feats, Identity())
    N = panel.n_stocks
    ok = panel.has_bar[:N] & feats.tradable[:, :, None]
    for name in ("open", "high", "low", "close"):
        np.testing.assert_allclose(getattr(p2, name)[:N][ok], getattr(panel, name)[:N][ok], rtol=1e-5)
    for name in ("open", "close"):   # ETFs untouched
        np.testing.assert_array_equal(getattr(p2, name)[N:], getattr(panel, name)[N:])


def test_permutation_preserves_daily_residuals_and_destroys_order(small):
    panel, feats, news = small
    p2 = pm.permute_panel(panel, feats, np.random.default_rng(0))
    N = panel.n_stocks
    r1 = sg.intraday_log_returns(panel)[:N]
    r2 = sg.intraday_log_returns(p2)[:N]
    beta = np.nan_to_num(feats.beta)[..., None]
    re = np.nan_to_num(feats.ret[panel.etf_idx])
    e1, e2 = r1 - beta * re, r2 - beta * re
    ok = feats.tradable
    # same multiset of residuals within each stock-day (sorted values match), different order
    np.testing.assert_allclose(np.sort(e1[ok], axis=-1), np.sort(e2[ok], axis=-1), atol=2e-5)
    assert not np.allclose(e1[ok], e2[ok], atol=1e-6)
    pn = pm.permute_news(news, panel, np.random.default_rng(0))
    assert len(pn) > 0 and pn["symbols"].apply(len).eq(1).all()


def test_fill_modes(small):
    panel, feats, _ = small
    cands = sg.find_candidates(feats, 2.0, news="no_news")
    base = bt.simulate_trades(panel, feats, cands)
    trap = bt.simulate_trades(panel, feats, cands, fill="signal_close")
    pess = bt.simulate_trades(panel, feats, cands, fill="pessimistic")
    i, d, t = trap["i"].to_numpy(), trap["d"].to_numpy(), trap["t"].to_numpy()
    np.testing.assert_allclose(trap["entry_px"], panel.close_ff[i, d, t])        # the deliberate bug
    # pessimistic: never a better price than the open fill, on either side
    worse_in = np.where(base["direction"] > 0, pess["entry_px"] >= base["entry_px"] - 1e-6,
                        pess["entry_px"] <= base["entry_px"] + 1e-6)
    assert worse_in.all()
    assert pess["gross_bps"].mean() < base["gross_bps"].mean()
    delayed = bt.simulate_trades(panel, feats, cands, entry_delay=2)
    assert (pd.to_datetime(delayed["entry_time"]) >= pd.to_datetime(delayed["signal_time"])
            + pd.Timedelta(minutes=10)).all()


def test_random_entry_fixed_holding(small):
    panel, feats, _ = small
    cands = pd.DataFrame({"i": [0, 1], "d": [40, 41], "t": [10, 20], "z": [1.0, -1.0], "has_news": [False, False]})
    tr = bt.simulate_trades(panel, feats, cands, exit_after_bars=np.array([3, 5]))
    assert list(tr["bars_held"]) == [3, 5] and set(tr["exit_reason"]) == {"time"}


def test_deflated_sharpe_penalises_more_trials():
    r = np.random.default_rng(1).normal(0.001, 0.01, 500)
    one = vs.deflated_sharpe(r, 1, np.nan)["dsr"]
    many = vs.deflated_sharpe(r, 1000, 1.0)["dsr"]
    assert 0 <= many < one <= 1


def test_kill_n_scales_with_noise():
    assert risk.kill_n(10.0, 50.0) < risk.kill_n(10.0, 100.0) < risk.kill_n(5.0, 100.0)
    assert risk.kill_n(-1.0, 50.0) == -1


def test_block_bootstrap_indices_shape_and_contiguity():
    idx = cm.block_bootstrap_indices(np.random.default_rng(0), 100, 252, 5, 7)
    assert idx.shape == (7, 252) and idx.min() >= 0 and idx.max() < 100
    assert (np.diff(idx[:, :5], axis=1) == 1).all()


def test_trial_log_counts_distinct_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "TRIAL_LOG_PATH", tmp_path / "log.csv")
    p = bt.Params().as_dict()
    trial_log.log([{**p, "n_trades": 10, "mean_net_bps": 1, "sharpe": 1.0},
                   {**p, "n_trades": 10, "mean_net_bps": 1, "sharpe": 1.5, "extra": {"stress": "x"}},
                   {**p, "z_entry": 3.0, "n_trades": 5, "mean_net_bps": 2, "sharpe": 3.0}], "real", "heatmap", "dev")
    trial_log.log([{**p, "n_trades": 1, "mean_net_bps": 0, "sharpe": 9.0}], "permuted", "heatmap", "dev")
    trial_log.log([{**p, "z_exit": 0.7, "n_trades": 9, "mean_net_bps": 1, "sharpe": 0.5}], "real", "jitter", "dev")
    n, var = trial_log.trial_stats("real", selection_only=True)     # jitter draw not a selection trial
    assert n == 2 and var == pytest.approx(np.var([1.25, 3.0], ddof=1))
    n_all, _ = trial_log.trial_stats("real", selection_only=False)
    assert n_all == 3
