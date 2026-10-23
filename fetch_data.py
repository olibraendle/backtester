"""
Download 5-minute bars and news from the Alpaca Market Data API and cache them
as parquet under data/.

Cache layout (one file per symbol-month for bars, one per month for news):
    data/bars/<SYMBOL>/<YYYY-MM>.parquet
    data/news/<YYYY-MM>.parquet
    data/manifest.json     chunk key -> the request `end` date it was downloaded up to

A chunk is re-downloaded only if it is missing or was downloaded before the
month was complete, so re-runs only fetch new data.

Usage:
    python fetch_data.py                       # full universe, 2021-01-01..yesterday
    python fetch_data.py --start 2024-01-01 --symbols AAPL,XLK
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

import config as C

BAR_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap", "trade_count"]
NEWS_COLUMNS = ["id", "headline", "created_at", "updated_at", "symbols", "source"]


def read_dotenv(path) -> dict[str, str]:
    """Minimal KEY=VALUE parser for a gitignored .env file (no shell expansion)."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().removeprefix("export ").strip()] = v.strip().strip("'\"")
    return out


# ---------------------------------------------------------------------------
# HTTP client with throttling and retry
# ---------------------------------------------------------------------------
class AlpacaClient:
    def __init__(self, key: str, secret: str):
        self.session = requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.min_interval = 60.0 / C.MAX_REQUESTS_PER_MINUTE
        self._last_request = 0.0
        self._lock = threading.Lock()
        self.feed = C.PREFERRED_FEED

    @classmethod
    def from_env(cls) -> "AlpacaClient":
        dotenv = read_dotenv(C.DOTENV_PATH)
        key = os.environ.get(C.API_KEY_ENV) or dotenv.get(C.API_KEY_ENV)
        secret = os.environ.get(C.API_SECRET_ENV) or dotenv.get(C.API_SECRET_ENV)
        if not key or not secret:
            sys.exit(f"Set {C.API_KEY_ENV} and {C.API_SECRET_ENV} in the environment or in "
                     f"{C.DOTENV_PATH} (or use backtest.py --synthetic).")
        return cls(key, secret)

    def _throttle(self) -> None:
        """Thread-safe: reserve the next request slot, then sleep until it (rate stays <= the limit)."""
        with self._lock:
            slot = max(time.monotonic(), self._last_request + self.min_interval)
            self._last_request = slot
        wait = slot - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def get(self, endpoint: str, params: dict) -> dict:
        """GET with throttling; retries with exponential backoff on 429 / 5xx / network errors."""
        url = C.DATA_BASE_URL + endpoint
        for attempt in range(C.MAX_RETRIES + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=C.HTTP_TIMEOUT_S)
            except requests.RequestException as exc:
                if attempt == C.MAX_RETRIES:
                    raise
                self._backoff(attempt, f"network error: {exc}")
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == C.MAX_RETRIES:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After")
                self._backoff(attempt, f"HTTP {resp.status_code}",
                              float(retry_after) if retry_after else None)
                continue
            # 4xx other than 429: let the caller decide (e.g. SIP not permitted -> 403)
            msg = f"HTTP {resp.status_code} for {endpoint}: {resp.text[:300]}"
            if resp.status_code == 403:
                raise PermissionError(msg)
            raise requests.HTTPError(msg)
        raise RuntimeError("unreachable")

    @staticmethod
    def _backoff(attempt: int, why: str, retry_after: float | None = None) -> None:
        delay = retry_after if retry_after else min(60.0, 2.0 ** attempt) + random.uniform(0, 1)
        print(f"    {why}; retrying in {delay:.1f}s", flush=True)
        time.sleep(delay)

    def paginate(self, endpoint: str, params: dict) -> list[dict]:
        """Follow next_page_token until exhausted; returns the list of page payloads."""
        pages, token = [], None
        while True:
            p = dict(params)
            if token:
                p["page_token"] = token
            payload = self.get(endpoint, p)
            pages.append(payload)
            token = payload.get("next_page_token")
            if not token:
                return pages


# ---------------------------------------------------------------------------
# Month chunking + manifest
# ---------------------------------------------------------------------------
def month_chunks(start: dt.date, end: dt.date) -> list[tuple[str, dt.date, dt.date]]:
    """[(YYYY-MM, chunk_start, chunk_end)] covering start..end inclusive."""
    out, cur = [], dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        out.append((cur.strftime("%Y-%m"), max(cur, start), min(nxt - dt.timedelta(days=1), end)))
        cur = nxt
    return out


def write_parquet_atomic(df: pd.DataFrame, path) -> None:
    """Write to a temp file, then rename: a crash never leaves a half-written file under the real name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)          # verify_cache() re-checks every file at startup


def verify_cache(manifest: dict) -> dict:
    """Drop manifest entries whose parquet file is missing or unreadable (e.g. after a crash)."""
    import pyarrow.parquet as pq
    for tmp in C.DATA_DIR.glob("**/*.parquet.tmp"):
        tmp.unlink()
    bad = []
    for key in list(manifest):
        path = C.DATA_DIR / f"{key}.parquet"
        try:
            pq.read_metadata(path)
        except Exception:
            bad.append(key)
            manifest.pop(key)
            path.unlink(missing_ok=True)
    if bad:
        print(f"cache check: {len(bad)} damaged/missing chunk(s) will be re-downloaded: {bad[:5]}")
        save_manifest(manifest)
    return manifest


def load_manifest() -> dict:
    if C.MANIFEST_PATH.exists():
        return json.loads(C.MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    C.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = C.MANIFEST_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(manifest, indent=0, sort_keys=True))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(C.MANIFEST_PATH)


def chunk_is_complete(manifest: dict, key: str, chunk_start: dt.date, chunk_end: dt.date, path) -> bool:
    """Complete if the file exists and it was downloaded covering the whole chunk range."""
    rec = manifest.get(key)
    if not rec or not path.exists():
        return False
    return dt.date.fromisoformat(rec["start"]) <= chunk_start and dt.date.fromisoformat(rec["end"]) >= chunk_end


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------
def _bars_request(client: AlpacaClient, symbols: list[str], start: dt.date, end: dt.date) -> list[dict]:
    params = {
        "symbols": ",".join(symbols),
        "timeframe": C.BAR_TIMEFRAME,
        "start": start.isoformat(),
        # `end` is inclusive; use the following midnight UTC so the whole NY session is covered.
        "end": (end + dt.timedelta(days=1)).isoformat() + "T00:00:00Z",
        "adjustment": C.BAR_ADJUSTMENT,
        "feed": client.feed,
        "limit": C.BARS_PAGE_LIMIT,
        "sort": "asc",
    }
    try:
        return client.paginate(C.BARS_ENDPOINT, params)
    except PermissionError as exc:
        if client.feed != C.FALLBACK_FEED:
            print(f"  feed '{client.feed}' not permitted ({exc}); falling back to '{C.FALLBACK_FEED}'")
            client.feed = C.FALLBACK_FEED
            return _bars_request(client, symbols, start, end)
        raise


def _bars_pages_to_frames(pages: list[dict], symbols: list[str]) -> dict[str, pd.DataFrame]:
    rows: dict[str, list] = {s: [] for s in symbols}
    for page in pages:
        for sym, bars in (page.get("bars") or {}).items():
            rows.setdefault(sym, []).extend(bars)
    out = {}
    for sym, bars in rows.items():
        df = pd.DataFrame(bars, columns=["t", "o", "h", "l", "c", "v", "vw", "n"])
        df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close",
                                "v": "volume", "vw": "vwap", "n": "trade_count"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = sym
        out[sym] = df[BAR_COLUMNS].drop_duplicates("timestamp").sort_values("timestamp")
    return out


def fetch_bars(client: AlpacaClient, symbols: list[str], start: dt.date, end: dt.date) -> None:
    manifest = load_manifest()
    with ThreadPoolExecutor(max_workers=C.FETCH_THREADS) as pool:
        for month, c_start, c_end in month_chunks(start, end):
            todo = [s for s in symbols
                    if not chunk_is_complete(manifest, f"bars/{s}/{month}", c_start, c_end,
                                             C.BARS_DIR / s / f"{month}.parquet")]
            if not todo:
                continue
            print(f"bars {month}: {len(todo)} symbols", flush=True)
            # Always download the full month so a chunk file is never a partial month
            month_start = dt.date.fromisoformat(month + "-01")
            batches = [todo[i:i + C.BAR_SYMBOLS_PER_REQUEST] for i in range(0, len(todo), C.BAR_SYMBOLS_PER_REQUEST)]
            pages = pool.map(lambda batch: (batch, _bars_request(client, batch, month_start, c_end)), batches)
            for batch, pg in pages:
                for sym, df in _bars_pages_to_frames(pg, batch).items():
                    if sym not in batch:
                        continue
                    write_parquet_atomic(df, C.BARS_DIR / sym / f"{month}.parquet")
                    manifest[f"bars/{sym}/{month}"] = {"start": month_start.isoformat(), "end": c_end.isoformat(),
                                                       "feed": client.feed}
            save_manifest(manifest)


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------
def news_query_symbols(stocks: list[str]) -> list[str]:
    syms = list(stocks)
    for s in stocks:
        syms += C.NEWS_SYMBOL_ALIASES.get(s, [])
    return syms


def _news_month(client: AlpacaClient, query_syms: list[str], month_start: dt.date, c_end: dt.date) -> pd.DataFrame:
    params = {
        "symbols": ",".join(query_syms),
        "start": month_start.isoformat() + "T00:00:00Z",
        "end": (c_end + dt.timedelta(days=1)).isoformat() + "T00:00:00Z",
        "limit": C.NEWS_PAGE_LIMIT,
        "sort": "asc",
        "include_content": "false",
    }
    items = [n for page in client.paginate(C.NEWS_ENDPOINT, params) for n in page.get("news", [])]
    df = pd.DataFrame(items, columns=NEWS_COLUMNS) if items else pd.DataFrame(columns=NEWS_COLUMNS)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
    df["symbols"] = df["symbols"].apply(lambda s: list(s) if s is not None else [])
    return df.drop_duplicates("id")


def fetch_news(client: AlpacaClient, stocks: list[str], start: dt.date, end: dt.date) -> None:
    """Months are paginated sequentially inside, but FETCH_THREADS months download concurrently."""
    manifest = load_manifest()
    query_syms = news_query_symbols(stocks)
    todo = []
    for month, c_start, c_end in month_chunks(start, end):
        path, key = C.NEWS_DIR / f"{month}.parquet", f"news/{month}"
        rec = manifest.get(key)
        # A universe change invalidates the news cache for that month.
        if chunk_is_complete(manifest, key, c_start, c_end, path) and set(rec.get("symbols", [])) >= set(query_syms):
            continue
        todo.append((month, dt.date.fromisoformat(month + "-01"), c_end))
    with ThreadPoolExecutor(max_workers=C.FETCH_THREADS) as pool:
        futures = {pool.submit(_news_month, client, query_syms, m0, c_end): (month, m0, c_end)
                   for month, m0, c_end in todo}
        for fut in futures:                       # in month order
            month, m0, c_end = futures[fut]
            df = fut.result()
            write_parquet_atomic(df, C.NEWS_DIR / f"{month}.parquet")
            manifest[f"news/{month}"] = {"start": m0.isoformat(), "end": c_end.isoformat(), "symbols": query_syms}
            save_manifest(manifest)
            print(f"news {month}: {len(df)} articles", flush=True)


# ---------------------------------------------------------------------------
# Cache readers (used by backtest.py)
# ---------------------------------------------------------------------------
def regular_hours_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep bars whose START is 09:30 .. 15:55 New York time (RTH). Timestamps stay UTC."""
    local = df["timestamp"].dt.tz_convert(C.TIMEZONE)
    minutes = local.dt.hour * 60 + local.dt.minute
    open_min = C.SESSION_OPEN.hour * 60 + C.SESSION_OPEN.minute
    return df[(minutes >= open_min) & (minutes < open_min + C.BARS_PER_DAY * C.BAR_MINUTES)]


def cached_symbols() -> set[str]:
    return {d.name for d in C.BARS_DIR.glob("*") if d.is_dir() and any(d.glob("*.parquet"))}


def load_symbol_bars(sym: str, before: pd.Timestamp | None = None) -> pd.DataFrame:
    """One symbol's cached RTH bars (bar-start timestamps, UTC), optionally only bars before `before`."""
    frames = []
    for f in sorted((C.BARS_DIR / sym).glob("*.parquet")):
        df = pd.read_parquet(f, columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if before is not None:
            df = df[df["timestamp"] < before]
        frames.append(regular_hours_only(df))
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp", ignore_index=True)


def load_bars(symbols: list[str]) -> pd.DataFrame:
    """Long frame of cached RTH bars for `symbols` (timestamps tz-aware UTC, bar START times)."""
    frames = []
    for sym in symbols:
        files = sorted((C.BARS_DIR / sym).glob("*.parquet"))
        if not files:
            print(f"warning: no cached bars for {sym}")
            continue
        for f in files:
            df = pd.read_parquet(f, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            frames.append(regular_hours_only(df))
    if not frames:
        sys.exit("No cached bars found. Run `python fetch_data.py` first (or use --synthetic).")
    df = pd.concat(frames, ignore_index=True)
    df["symbol"] = df["symbol"].astype("category")
    return df.drop_duplicates(["symbol", "timestamp"]).sort_values(["symbol", "timestamp"], ignore_index=True)


def load_news() -> pd.DataFrame:
    """All cached news, de-duplicated by id, with aliases (e.g. FB) mapped to current tickers."""
    files = sorted(C.NEWS_DIR.glob("*.parquet"))
    if not files:
        print("warning: no cached news; every bar will be treated as no-news")
        return pd.DataFrame(columns=NEWS_COLUMNS)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # Same article can appear in two monthly chunks (e.g. if the API filters on updated_at).
    df = df.sort_values("updated_at").drop_duplicates("id", keep="first")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    reverse = {old: new for new, olds in C.NEWS_SYMBOL_ALIASES.items() for old in olds}
    df["symbols"] = df["symbols"].apply(
        lambda syms: sorted({reverse.get(s, s) for s in (list(syms) if syms is not None else [])}))
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=dt.date.fromisoformat, default=C.HISTORY_START)
    ap.add_argument("--end", type=dt.date.fromisoformat, default=None,
                    help="inclusive, must be at least 1 day in the past (default: yesterday)")
    ap.add_argument("--symbols", default=None, help="comma-separated subset of the universe")
    ap.add_argument("--skip-news", action="store_true")
    ap.add_argument("--skip-bars", action="store_true")
    args = ap.parse_args()

    end = args.end or C.default_history_end()
    if end >= dt.date.today():
        sys.exit("--end must be at least 1 day in the past (free plan delays recent SIP data).")
    # Default: main universe + out-of-universe list + their ETFs + the volatility proxy.
    sector_map = {**C.SECTOR_MAP, **C.OUT_OF_UNIVERSE_MAP}
    stocks = args.symbols.split(",") if args.symbols else list(sector_map)
    etfs = sorted({sector_map[s] for s in stocks if s in sector_map} | {s for s in stocks if s in C.ETFS})
    extra = [] if args.symbols else [C.VIX_PROXY]
    stocks = [s for s in stocks if s not in C.ETFS and s != C.VIX_PROXY]

    verify_cache(load_manifest())
    client = AlpacaClient.from_env()
    if not args.skip_bars:
        fetch_bars(client, stocks + etfs + extra, args.start, end)
    if not args.skip_news:
        fetch_news(client, stocks, args.start, end)
    print("done.")


if __name__ == "__main__":
    main()
