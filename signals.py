"""
Turn raw bars + news into a dense (symbol, day, slot) panel and compute the
signal features.

Timing conventions (the source of most lookahead bugs, so spelled out):
  * Alpaca bar timestamps are bar START times. Slot t on a day covers
    [09:30 + 5t, 09:35 + 5t) and its close is only known at 09:35 + 5t.
  * Everything indexed [.., d, t] is known at the CLOSE of bar t on day d.
  * β and σ for day d use only the previous BETA_LOOKBACK_DAYS full days
    (days d-20 .. d-1); they are constant within day d.
  * Intraday returns / lookback windows never use data from a previous day.
  * Universe filters for day d use only days before d.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as C

S = C.BARS_PER_DAY
F = np.float32          # price / feature grids: float32 halves memory; precision is ample for bps


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
@dataclass
class Panel:
    symbols: list[str]           # stocks first, then ETFs
    n_stocks: int
    etf_idx: np.ndarray          # (n_stocks,) index of each stock's sector ETF in `symbols`
    days: pd.DatetimeIndex       # trading dates (tz-naive midnight)
    open: np.ndarray             # (M, D, S) NaN where no bar
    high: np.ndarray             # (M, D, S)
    low: np.ndarray              # (M, D, S)
    close: np.ndarray            # (M, D, S)
    has_bar: np.ndarray          # (M, D, S) bool
    close_ff: np.ndarray         # close forward-filled within the day (last traded price)
    next_open: np.ndarray        # open of the first bar at slot >= t (fill price for an order sent at start of t)
    next_open_slot: np.ndarray   # slot of that bar, -1 if none left today
    daily_dollar_vol: np.ndarray  # (M, D) RTH dollar volume
    early_close: np.ndarray      # (D,) bool, session ended before 16:00
    bar_close_ns: np.ndarray     # (D, S) int64 UTC ns of each bar's CLOSE time
    prev_close_ns: np.ndarray    # (D,) int64 UTC ns of the previous trading day's session close

    @property
    def stocks(self) -> list[str]:
        return self.symbols[: self.n_stocks]

    def bar_start(self, d, t) -> pd.Timestamp:
        """NY timestamp of the START of slot t on day index d (t may be an array)."""
        return (pd.Timestamp(self.days[d]) + pd.Timedelta(hours=9, minutes=30)
                + pd.to_timedelta(np.asarray(t) * C.BAR_MINUTES, unit="min"))


def _ffill_last_axis(a: np.ndarray) -> np.ndarray:
    """Forward fill NaNs along the last axis (never pulls values from the future)."""
    idx = np.where(~np.isnan(a), np.arange(a.shape[-1]), 0)
    np.maximum.accumulate(idx, axis=-1, out=idx)
    out = np.take_along_axis(a, idx, axis=-1)
    return out


def _bfill_index_last_axis(mask: np.ndarray) -> np.ndarray:
    """For each t, the smallest s >= t with mask[..., s] True, else -1."""
    n = mask.shape[-1]
    idx = np.where(mask, np.arange(n), n)
    idx = np.minimum.accumulate(idx[..., ::-1], axis=-1)[..., ::-1]
    return np.where(idx == n, -1, idx)


def session_times(days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(D, S) UTC-ns bar CLOSE times, and (D,) UTC-ns 16:00 of each day."""
    open_local = (days + pd.Timedelta(hours=9, minutes=30)).tz_localize(C.TIMEZONE)
    open_ns = open_local.tz_convert("UTC").as_unit("ns").asi8
    step = np.int64(C.BAR_MINUTES * 60 * 1_000_000_000)
    close_ns = open_ns[:, None] + step * (np.arange(S, dtype=np.int64) + 1)[None, :]
    sixteen = (days + pd.Timedelta(hours=16)).tz_localize(C.TIMEZONE).tz_convert("UTC").as_unit("ns").asi8
    return close_ns, sixteen


def _slotify(df: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-hours bars on the 5-min grid; add `day` (NY date) and `slot` (0..77)."""
    local = df["timestamp"].dt.tz_convert(C.TIMEZONE)
    minutes = (local.dt.hour * 60 + local.dt.minute) - (C.SESSION_OPEN.hour * 60 + C.SESSION_OPEN.minute)
    # Regular trading hours only: bar starts 09:30 .. 15:55, aligned to the 5-min grid.
    keep = (minutes >= 0) & (minutes < S * C.BAR_MINUTES) & (minutes % C.BAR_MINUTES == 0) & (local.dt.second == 0)
    out = df[keep].copy()
    out["day"] = local[keep].dt.tz_localize(None).dt.normalize()
    out["slot"] = (minutes[keep] // C.BAR_MINUTES).astype(int)
    return out


def build_panel(bars: pd.DataFrame, stocks: list[str], sector_map: dict[str, str]) -> Panel:
    """bars: long frame with symbol, timestamp (UTC, bar start), open, high, low, close, volume, vwap."""
    groups = {str(k): g for k, g in bars.groupby(bars["symbol"].astype(str), sort=False)}
    return build_panel_streaming(lambda sym: groups.get(sym, bars.iloc[:0]), stocks, sector_map, set(groups))


def build_panel_streaming(load_symbol, stocks: list[str], sector_map: dict[str, str], available: set[str]) -> Panel:
    """
    Build the panel one symbol at a time (load_symbol(sym) -> that symbol's bars), so the
    full multi-year bar history never has to sit in memory as one DataFrame. The trading
    calendar is the union of the sector ETFs' bar days.
    """
    stocks = [s for s in stocks if s in available and sector_map[s] in available]
    etfs = sorted({sector_map[s] for s in stocks})
    symbols = stocks + etfs
    sym_index = {s: i for i, s in enumerate(symbols)}
    days = pd.DatetimeIndex(sorted(set().union(*[set(_slotify(load_symbol(e))["day"]) for e in etfs])))
    M, D = len(symbols), len(days)
    o, h, lo, c = (np.full((M, D, S), np.nan, dtype=F) for _ in range(4))
    daily_dv = np.zeros((M, D))
    for k, sym in enumerate(symbols):
        df = _slotify(load_symbol(sym))
        di = days.get_indexer(df["day"])
        ok = di >= 0
        di, ti, df = di[ok], df["slot"].to_numpy()[ok], df[ok]
        close = df["close"].to_numpy(dtype=float)
        o[k, di, ti] = df["open"].to_numpy(dtype=float)
        c[k, di, ti] = close
        h[k, di, ti] = df["high"].to_numpy(dtype=float) if "high" in df else np.fmax(o[k, di, ti], close)
        lo[k, di, ti] = df["low"].to_numpy(dtype=float) if "low" in df else np.fmin(o[k, di, ti], close)
        vw = df["vwap"].to_numpy(dtype=float) if "vwap" in df else close
        np.add.at(daily_dv[k], di, np.nan_to_num(df["volume"].to_numpy(dtype=float) * np.where(np.isnan(vw), close, vw)))
    return make_panel(symbols, len(stocks), np.array([sym_index[sector_map[s]] for s in stocks], dtype=int),
                      days, o, h, lo, c, daily_dv)


def make_panel(symbols: list[str], n_stocks: int, etf_idx: np.ndarray, days: pd.DatetimeIndex,
               o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, daily_dollar_vol: np.ndarray) -> Panel:
    """Assemble a Panel from (M, D, S) OHLC grids (NaN = no bar). Used for real, permuted and generated data."""
    D = len(days)
    has = ~np.isnan(c)

    # A session that ends early has no bars after its close for ANY symbol.
    last_slot = np.where(has.any(axis=0), np.arange(S), -1).max(axis=1)
    early_close = last_slot < S - 1

    bar_close_ns, sixteen_ns = session_times(days)
    session_close_ns = bar_close_ns[np.arange(D), np.maximum(last_slot, 0)]
    prev_close_ns = np.empty(D, dtype=np.int64)
    prev_close_ns[1:] = session_close_ns[:-1]
    prev_close_ns[0] = sixteen_ns[0] - np.int64(86_400 * 1_000_000_000)

    nxt_slot = _bfill_index_last_axis(has).astype(np.int16)
    nxt_open = np.take_along_axis(o, np.maximum(nxt_slot, 0).astype(np.intp), axis=-1)
    nxt_open[nxt_slot < 0] = np.nan

    return Panel(
        symbols=list(symbols), n_stocks=n_stocks, etf_idx=np.asarray(etf_idx, dtype=int),
        days=days, open=o, high=h, low=lo, close=c, has_bar=has, close_ff=_ffill_last_axis(c),
        next_open=nxt_open, next_open_slot=nxt_slot,
        daily_dollar_vol=np.asarray(daily_dollar_vol, dtype=float),
        early_close=early_close, bar_close_ns=bar_close_ns, prev_close_ns=prev_close_ns,
    )


def panel_subset(panel: Panel, d0: int, d1: int) -> Panel:
    """Days [d0, d1) of a panel (all symbols). Features must be recomputed on the subset."""
    sl = slice(d0, d1)
    return make_panel(panel.symbols, panel.n_stocks, panel.etf_idx, panel.days[sl],
                      panel.open[:, sl], panel.high[:, sl], panel.low[:, sl], panel.close[:, sl],
                      panel.daily_dollar_vol[:, sl])


# ---------------------------------------------------------------------------
# Building blocks (each is unit-tested for lookahead)
# ---------------------------------------------------------------------------
def intraday_log_returns(panel: Panel) -> np.ndarray:
    """
    5-min log returns that never span the overnight gap.
    Slot 0 (and the first traded bar of a day) uses close/open of that same bar,
    later slots use close_t / close_{t-1} (forward-filled within the day, so a
    bar without trades contributes 0). Stock-days with too few bars are NaN.
    """
    cff = panel.close_ff
    prev = np.empty_like(cff)
    prev[..., 1:] = cff[..., :-1]
    prev[..., 0] = np.nan
    prev = np.where(np.isnan(prev), panel.open, prev)       # first traded bar of the day: close / open
    r = np.log(cff / prev).astype(F)
    r[np.isnan(r)] = 0.0                                    # no trade yet / no trade this bar
    valid_day = panel.has_bar.sum(axis=-1) >= C.MIN_BARS_PER_DAY
    r[~valid_day] = np.nan
    return r


def intraday_window_sum(x: np.ndarray, L: int) -> np.ndarray:
    """Sum of the last L slots ending at t (inclusive), within the same day; NaN if t < L-1."""
    cs = np.cumsum(x, axis=-1, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=x.dtype)
    out[..., L - 1] = cs[..., L - 1]
    out[..., L:] = cs[..., L:] - cs[..., :-L]
    return out


def past_days_sum(daily: np.ndarray, window: int, min_days: int) -> np.ndarray:
    """
    daily: (X, D). Returns sum over days d-window .. d-1 (strictly before d),
    NaN where fewer than min_days valid (non-NaN) days are available.
    """
    return pd.DataFrame(daily.T).rolling(window, min_periods=min_days).sum().shift(1).to_numpy().T


def past_days_mean(daily: np.ndarray, window: int, min_days: int) -> np.ndarray:
    """Mean over valid days among d-window .. d-1 (strictly before d)."""
    return pd.DataFrame(daily.T).rolling(window, min_periods=min_days).mean().shift(1).to_numpy().T


def rolling_beta_sigma(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    y, x: (N, D, S) intraday returns (stock, ETF). OLS slope of y on x and the
    std of the residual y - βx, both estimated on the previous BETA_LOOKBACK_DAYS
    days only. Returns (beta, sigma), each (N, D).
    """
    both = ~np.isnan(y) & ~np.isnan(x)
    yv, xv = np.where(both, y, 0.0), np.where(both, x, 0.0)
    n_day = both.sum(-1, dtype=np.float64).astype(float)
    nan_day = np.where(n_day > 0, 0.0, np.nan)              # NaN marks days without data
    sums = {k: past_days_sum(v + nan_day, C.BETA_LOOKBACK_DAYS, C.MIN_LOOKBACK_DAYS)
            for k, v in dict(n=n_day, x=xv.sum(-1, dtype=np.float64), y=yv.sum(-1, dtype=np.float64), xx=(xv * xv).sum(-1, dtype=np.float64),
                             yy=(yv * yv).sum(-1, dtype=np.float64), xy=(xv * yv).sum(-1, dtype=np.float64)).items()}
    n = sums["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        sxx = sums["xx"] - sums["x"] ** 2 / n
        syy = sums["yy"] - sums["y"] ** 2 / n
        sxy = sums["xy"] - sums["x"] * sums["y"] / n
        beta = sxy / sxx
        var_res = (syy - 2 * beta * sxy + beta ** 2 * sxx) / (n - 1)
        sigma = np.sqrt(np.maximum(var_res, 0.0))
    return beta, sigma


def rolling_sigma(x: np.ndarray) -> np.ndarray:
    """Std of x (M, D, S) over the previous BETA_LOOKBACK_DAYS days. Returns (M, D)."""
    ok = ~np.isnan(x)
    xv = np.where(ok, x, 0.0)
    n_day = ok.sum(-1, dtype=np.float64).astype(float)
    nan_day = np.where(n_day > 0, 0.0, np.nan)
    n = past_days_sum(n_day + nan_day, C.BETA_LOOKBACK_DAYS, C.MIN_LOOKBACK_DAYS)
    s1 = past_days_sum(xv.sum(-1, dtype=np.float64) + nan_day, C.BETA_LOOKBACK_DAYS, C.MIN_LOOKBACK_DAYS)
    s2 = past_days_sum((xv * xv).sum(-1, dtype=np.float64) + nan_day, C.BETA_LOOKBACK_DAYS, C.MIN_LOOKBACK_DAYS)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.maximum((s2 - s1 ** 2 / n) / (n - 1), 0.0))


def company_news_times(news: pd.DataFrame, stocks: list[str]) -> dict[str, np.ndarray]:
    """Sorted created_at (UTC ns) of company-specific articles (<= MAX_NEWS_SYMBOLS tags) per stock."""
    out = {s: np.array([], dtype=np.int64) for s in stocks}
    if news is None or news.empty:
        return out
    n = news[["created_at", "symbols"]].copy()
    n["symbols"] = n["symbols"].apply(lambda x: list(x) if x is not None else [])
    n = n[n["symbols"].apply(len).between(1, C.MAX_NEWS_SYMBOLS)]
    n = n.explode("symbols")
    n = n[n["symbols"].isin(stocks)]
    created = pd.DatetimeIndex(pd.to_datetime(n["created_at"], utc=True))  # created_at, never updated_at
    n = n.assign(ns=created.as_unit("ns").asi8)
    for sym, g in n.groupby("symbols"):
        out[sym] = np.sort(g["ns"].to_numpy(dtype=np.int64))
    return out


def news_counts(panel: Panel, news_times: dict[str, np.ndarray]) -> np.ndarray:
    """
    (N, D, S) int: company-specific articles with
        previous session close < created_at <= CLOSE time of bar t.
    """
    N, D = panel.n_stocks, len(panel.days)
    out = np.zeros((N, D, S), dtype=np.int16)
    for i, sym in enumerate(panel.stocks):
        ts = news_times.get(sym)
        if ts is None or len(ts) == 0:
            continue
        upto = np.searchsorted(ts, panel.bar_close_ns, side="right")      # created_at <= bar close
        before = np.searchsorted(ts, panel.prev_close_ns, side="right")   # created_at <= prev close
        out[i] = upto - before[:, None]
    return out


def universe_mask(panel: Panel) -> np.ndarray:
    """(N, D) bool: price and liquidity filters for day d using only days < d."""
    N = panel.n_stocks
    last_close = panel.close_ff[:N, :, -1]                   # forward-filled, valid on half days too
    prev_close = np.full_like(last_close, np.nan)
    prev_close[:, 1:] = last_close[:, :-1]
    dv = panel.daily_dollar_vol[:N].copy()
    dv[panel.has_bar[:N].sum(-1) < C.MIN_BARS_PER_DAY] = np.nan
    avg_dv = past_days_mean(dv, C.DOLLAR_VOLUME_LOOKBACK_DAYS, C.MIN_LOOKBACK_DAYS)
    with np.errstate(invalid="ignore"):
        return (prev_close >= C.MIN_PRICE) & (avg_dv >= C.MIN_AVG_DOLLAR_VOLUME)


def avg_bar_dollar_volume(panel: Panel) -> np.ndarray:
    """(N, D) average dollar volume per 5-min bar over the previous days."""
    N = panel.n_stocks
    dv = panel.daily_dollar_vol[:N].copy()
    nb = panel.has_bar[:N].sum(-1).astype(float)
    bad = nb < C.MIN_BARS_PER_DAY
    dv[bad], nb[bad] = np.nan, np.nan
    w, m = C.DOLLAR_VOLUME_LOOKBACK_DAYS, C.MIN_LOOKBACK_DAYS
    return past_days_sum(dv, w, m) / past_days_sum(nb, w, m)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
@dataclass
class Features:
    lookback: int
    ret: np.ndarray          # (M, D, S) intraday log returns
    beta: np.ndarray         # (N, D)
    sigma: np.ndarray        # (N, D) std of 5-min residual returns
    sigma_raw: np.ndarray    # (M, D) std of each symbol's own 5-min returns (used for ETF z)
    etf_idx: np.ndarray      # (N,) index of each stock's ETF in the M axis
    eps: np.ndarray          # (N, D, S) residual returns
    z: np.ndarray            # (N, D, S)
    z_sector: np.ndarray     # (N, D, S) ETF's own z, aligned to each stock
    news_cnt: np.ndarray     # (N, D, S) company news since previous close, up to bar close
    tradable: np.ndarray     # (N, D) bool
    bar_dollar_vol: np.ndarray  # (N, D)

    @property
    def has_news(self) -> np.ndarray:
        return self.news_cnt > 0


def compute_features(panel: Panel, news: pd.DataFrame, lookback: int = C.LOOKBACK_BARS) -> Features:
    N = panel.n_stocks
    r = intraday_log_returns(panel)
    rs, re = r[:N], r[panel.etf_idx]
    beta, sigma = rolling_beta_sigma(rs, re)
    sigma_raw = rolling_sigma(r)
    sigma_etf = sigma_raw[panel.etf_idx]

    eps = (rs - beta[..., None].astype(F) * re).astype(F)
    del rs, re
    feats = Features(
        lookback=lookback, ret=r, beta=beta, sigma=sigma, sigma_raw=sigma_raw,
        etf_idx=panel.etf_idx, eps=eps,
        z=np.empty(0), z_sector=np.empty(0),
        news_cnt=news_counts(panel, company_news_times(news, panel.stocks)),
        tradable=np.zeros((N, len(panel.days)), bool),
        bar_dollar_vol=avg_bar_dollar_volume(panel),
    )
    set_lookback(feats, lookback)

    valid_day = panel.has_bar.sum(-1) >= C.MIN_BARS_PER_DAY
    with np.errstate(invalid="ignore"):
        tradable = (universe_mask(panel) & valid_day[:N] & valid_day[panel.etf_idx]
                    & np.isfinite(beta) & (sigma > 0) & (sigma_etf > 0))
    if C.SKIP_EARLY_CLOSE_DAYS:
        tradable &= ~panel.early_close[None, :]
    feats.tradable = tradable
    return feats


def set_lookback(feats: Features, lookback: int) -> None:
    """(Re)compute z and sector z for lookback L; β, σ and ε don't depend on L."""
    root_l = np.sqrt(lookback)
    with np.errstate(invalid="ignore", divide="ignore"):
        feats.z = intraday_window_sum(feats.eps, lookback) / (feats.sigma[..., None] * root_l).astype(F)
        # Sector z: the ETF's own L-bar return vs the ETF's own 5-min volatility (ETF rows only).
        etfs = np.unique(feats.etf_idx)
        z_etf = intraday_window_sum(feats.ret[etfs], lookback) / (feats.sigma_raw[etfs][..., None] * root_l).astype(F)
    feats.z_sector = z_etf[np.searchsorted(etfs, feats.etf_idx)]
    feats.lookback = lookback


def entry_slot_range() -> tuple[int, int]:
    """Slots t whose CLOSE time (start + 5 min) lies in [ENTRY_WINDOW_START, ENTRY_WINDOW_END]."""
    def minutes(t):
        return t.hour * 60 + t.minute - (C.SESSION_OPEN.hour * 60 + C.SESSION_OPEN.minute)
    lo = minutes(C.ENTRY_WINDOW_START) // C.BAR_MINUTES - 1
    hi = minutes(C.ENTRY_WINDOW_END) // C.BAR_MINUTES - 1
    return lo, hi


def find_candidates(feats: Features, z_entry: float, news: str = "any",
                    use_sector_filter: bool = C.USE_SECTOR_FILTER, day_mask: np.ndarray | None = None
                    ) -> pd.DataFrame:
    """
    Bars where entry rules 1, 3 and 5 hold (rule 2, news, selectable via `news`:
    "any" | "no_news" | "news"). Rule 4 (position limits) is the portfolio's job.
    Returns a frame with stock index i, day index d, signal slot t, z, has_news.
    """
    lo, hi = entry_slot_range()
    z = feats.z[:, :, lo:hi + 1]
    with np.errstate(invalid="ignore"):
        m = (np.abs(z) >= z_entry) & feats.tradable[:, :, None]
        if use_sector_filter:
            m &= np.abs(feats.z_sector[:, :, lo:hi + 1]) < C.SECTOR_Z_MAX
    if day_mask is not None:
        m &= day_mask[None, :, None]
    hn = feats.has_news[:, :, lo:hi + 1]
    if news == "no_news":
        m &= ~hn
    elif news == "news":
        m &= hn
    i, d, k = np.nonzero(m)
    return pd.DataFrame({"i": i, "d": d, "t": k + lo, "z": z[i, d, k], "has_news": hn[i, d, k]})
