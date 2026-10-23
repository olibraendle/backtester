"""
Synthetic 5-min bars + news with a planted effect, used by `backtest.py --synthetic`
and the tests. Output has exactly the same format as the fetch_data cache
readers, so the whole pipeline is exercised.

Planted effect:
  * Stock returns = β · sector ETF return + idiosyncratic noise (U-shaped intraday vol).
  * Idiosyncratic shocks of 10-16 σ spread over 1-2 bars hit a random 25% of stock-days.
  * Half of the shocks come with a company-specific news item (created during the
    first shock bar). Those shocks PERSIST (drift a further 0-30% in the same direction).
  * Shocks without news REVERT 40-90% over the following ~8 bars.

Traps that only a correct implementation handles:
  * Market-wide articles tagged with 5 symbols at no-news shocks (must be ignored,
    they are not company-specific).
  * Articles created the day before but UPDATED at the time of a no-news shock
    (must be ignored: only created_at counts).
  * Company news created ~2.5 h AFTER some no-news shocks (must not be seen at
    signal time - a lookahead leak would move those trades into the news group).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

S = C.BARS_PER_DAY


def _vol_profile() -> np.ndarray:
    """U-shaped intraday volatility multiplier, mean ~1."""
    x = np.linspace(0, 1, S)
    prof = 0.75 + 1.4 * (x - 0.45) ** 2 * 4 / 1.2
    prof[:3] *= [2.0, 1.5, 1.2]
    return prof / prof.mean()


def generate(n_days: int = C.SYNTH_DAYS, sector_map: dict[str, str] = C.SECTOR_MAP,
             seed: int = C.SYNTH_SEED, start=C.SYNTH_START, shock_prob: float = C.SYNTH_SHOCK_PROB,
             half_spread_bps: float = C.SYNTH_HALF_SPREAD_BPS) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns (bars, news, truth); truth["shocks"] lists the planted shocks.
    shock_prob=0 gives a pure null market: random-walk residuals, no planted effect.
    Observed open/close prices carry a random bid-ask bounce of +-half_spread_bps around the mid.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    stocks = list(sector_map)
    etfs = sorted(set(sector_map.values()))
    prof = _vol_profile()
    D = n_days

    # --- ETF returns -------------------------------------------------------
    etf_sig = rng.uniform(0.0006, 0.0010, len(etfs))
    r_etf = rng.standard_normal((len(etfs), D, S)) * etf_sig[:, None, None] * prof
    etf_pos = {e: k for k, e in enumerate(etfs)}

    # --- Stock returns -----------------------------------------------------
    N = len(stocks)
    beta = rng.uniform(0.7, 1.3, N)
    idio_sig = rng.uniform(0.0008, 0.0016, N)
    e_of = np.array([etf_pos[sector_map[s]] for s in stocks])
    idio = rng.standard_normal((N, D, S)) * idio_sig[:, None, None] * prof
    r_stock = beta[:, None, None] * r_etf[e_of] + idio

    # --- Planted shocks ----------------------------------------------------
    news_rows, shocks = [], []
    decay = np.exp(-np.arange(8) / 2.5)
    decay /= decay.sum()
    for i in range(N):
        for d in range(1, D):
            if rng.random() >= shock_prob:
                continue
            s0 = int(rng.integers(4, 58))
            width = int(rng.integers(1, 3))
            size = rng.uniform(10, 16) * idio_sig[i] * rng.choice([-1, 1])
            r_stock[i, d, s0:s0 + width] += size / width
            with_news = rng.random() < C.SYNTH_NEWS_SHOCK_FRAC
            after = slice(s0 + width, s0 + width + 8)
            # drift after the shock as a fraction of the shock: + persists, - reverts
            drift = rng.uniform(0.0, 0.3) if with_news else -rng.uniform(0.4, 0.9)
            r_stock[i, d, after] += drift * size * decay[: len(range(*after.indices(S)))]
            bar_start = days[d] + pd.Timedelta(minutes=570 + 5 * s0)
            shocks.append({"stock": stocks[i], "day": days[d], "slot": s0, "width": width, "size": size,
                           "news": with_news, "drift_frac": drift})
            if with_news:
                if rng.random() < C.SYNTH_NEWS_AFTER_MOVE_FRAC:
                    # The wire story lands 0-4 min AFTER the move's last bar closes: correct code
                    # sees no news at the signal bar; "news 5 min early" lookahead code would.
                    t = bar_start + pd.Timedelta(minutes=5 * width) + pd.Timedelta(seconds=float(rng.uniform(1, 240)))
                else:
                    t = bar_start + pd.Timedelta(seconds=float(rng.uniform(0, 290)))
                news_rows.append((t, t, [stocks[i]], "company"))
            else:
                u = rng.random()
                if u < 0.4:   # market wrap mentioning 5 tickers -> not company-specific
                    others = list(rng.choice([s for s in stocks if s != stocks[i]], 4, replace=False))
                    news_rows.append((bar_start, bar_start, [stocks[i]] + others, "wrap"))
                elif u < 0.6:  # old article updated now -> created_at is outside the window
                    created = days[d - 1] + pd.Timedelta(hours=11)
                    news_rows.append((created, bar_start + pd.Timedelta(minutes=1), [stocks[i]], "updated"))
                elif u < 0.8 and s0 + 30 < S:  # news arriving much later in the day
                    t = bar_start + pd.Timedelta(minutes=150)
                    news_rows.append((t, t, [stocks[i]], "late"))

    # --- Random unrelated company news (overnight or intraday) ------------
    for i in range(N):
        for d in range(1, D):
            if rng.random() < C.SYNTH_RANDOM_NEWS_PROB:
                t = days[d - 1] + pd.Timedelta(hours=16) + pd.Timedelta(
                    minutes=float(rng.uniform(0, (days[d] - days[d - 1]).total_seconds() / 60)))
                news_rows.append((t, t, [stocks[i]], "random"))

    # --- Prices ------------------------------------------------------------
    symbols = stocks + etfs
    rets = np.concatenate([r_stock, r_etf], axis=0)
    M = len(symbols)
    p0 = np.log(rng.uniform(40, 400, M))
    gaps = rng.standard_normal((M, D)) * 0.008
    gaps[:, 0] = 0
    log_close = np.empty((M, D, S))
    log_open = np.empty((M, D, S))
    level = p0.copy()
    micro = rng.standard_normal((M, D, S)) * 0.00015
    for d in range(D):
        day_open = level + gaps[:, d]
        # First bar's return is close/open of the bar itself (no overnight gap inside).
        lc = day_open[:, None] + np.cumsum(rets[:, d, :], axis=1)
        lo = np.empty_like(lc)
        lo[:, 0] = day_open
        lo[:, 1:] = lc[:, :-1] + micro[:, d, 1:]
        log_close[:, d], log_open[:, d] = lc, lo
        level = lc[:, -1]
    # Observed trade prices bounce between bid and ask around the mid.
    hs = half_spread_bps / 1e4
    log_close = log_close + hs * rng.choice([-1.0, 1.0], log_close.shape)
    log_open = log_open + hs * rng.choice([-1.0, 1.0], log_open.shape)
    close, open_ = np.exp(log_close), np.exp(log_open)
    spread = np.abs(rng.standard_normal((M, D, S))) * 0.0004
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    vol = np.round(rng.lognormal(np.log(2e5), 0.4, (M, D, S)) * prof)

    starts = (days.values[:, None] + (np.timedelta64(570, "m") + np.arange(S) * np.timedelta64(5, "m"))[None, :])
    ts = pd.DatetimeIndex(starts.ravel()).tz_localize(C.TIMEZONE, nonexistent="shift_forward", ambiguous="NaT").tz_convert("UTC")
    bars = pd.DataFrame({
        "symbol": np.repeat(symbols, D * S),
        "timestamp": np.tile(ts, M),
        "open": open_.ravel(), "high": high.ravel(), "low": low.ravel(), "close": close.ravel(),
        "volume": vol.ravel(), "vwap": ((open_ + high + low + close) / 4).ravel(),
        "trade_count": (vol / 100).ravel().astype(int),
    })

    news = pd.DataFrame(news_rows, columns=["created_at", "updated_at", "symbols", "source"])
    for col in ("created_at", "updated_at"):
        news[col] = pd.DatetimeIndex(news[col]).tz_localize(C.TIMEZONE, nonexistent="shift_forward", ambiguous="NaT").tz_convert("UTC")
    news = news.dropna(subset=["created_at", "updated_at"]).sort_values("created_at", ignore_index=True)
    news.insert(0, "id", np.arange(len(news)))
    news.insert(1, "headline", "synthetic " + news["source"])
    truth = {"shocks": pd.DataFrame(shocks), "beta": dict(zip(stocks, beta))}
    return bars, news, truth
