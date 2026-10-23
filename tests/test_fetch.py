"""Offline tests of the download/cache layer with a fake Alpaca client (no network)."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
import fetch_data as fd  # noqa: E402


def test_month_chunks_cover_range():
    chunks = fd.month_chunks(dt.date(2021, 1, 15), dt.date(2021, 3, 2))
    assert [c[0] for c in chunks] == ["2021-01", "2021-02", "2021-03"]
    assert chunks[0][1] == dt.date(2021, 1, 15) and chunks[-1][2] == dt.date(2021, 3, 2)
    assert chunks[1][2] == dt.date(2021, 2, 28)


class FakeClient:
    """Two pages of bars; SIP is refused so the code must fall back to IEX."""
    def __init__(self):
        self.feed = "sip"
        self.calls = []

    def paginate(self, endpoint, params):
        self.calls.append((endpoint, dict(params)))
        if endpoint == C.BARS_ENDPOINT:
            if params["feed"] == "sip":
                raise PermissionError("HTTP 403")
            bar = lambda t: {"t": t, "o": 1, "h": 1, "l": 1, "c": 1, "v": 10, "vw": 1, "n": 1}  # noqa: E731
            return [{"bars": {"AAPL": [bar("2021-01-04T14:30:00Z")]}, "next_page_token": "x"},
                    {"bars": {"AAPL": [bar("2021-01-04T14:35:00Z")], "XLK": [bar("2021-01-04T14:30:00Z")]},
                     "next_page_token": None}]
        return [{"news": [{"id": 1, "headline": "h", "created_at": "2021-01-04T15:00:00Z",
                           "updated_at": "2021-01-05T15:00:00Z", "symbols": ["FB"], "source": "s"}],
                 "next_page_token": None}]


def test_fetch_caches_and_skips_complete_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "BARS_DIR", tmp_path / "bars")
    monkeypatch.setattr(C, "NEWS_DIR", tmp_path / "news")
    monkeypatch.setattr(C, "MANIFEST_PATH", tmp_path / "manifest.json")
    client = FakeClient()
    start, end = dt.date(2021, 1, 1), dt.date(2021, 1, 31)
    fd.fetch_bars(client, ["AAPL", "XLK"], start, end)
    assert client.feed == "iex"
    bars = fd.load_bars(["AAPL", "XLK"])
    assert len(bars) == 3 and set(bars.symbol) == {"AAPL", "XLK"}
    assert str(bars.timestamp.dt.tz) == "UTC"
    n_calls = len(client.calls)
    fd.fetch_bars(client, ["AAPL", "XLK"], start, end)          # complete -> no new requests
    assert len(client.calls) == n_calls
    fd.fetch_bars(client, ["AAPL", "XLK"], start, dt.date(2021, 2, 3))   # only February is new
    assert all(c[1]["start"].startswith("2021-02") for c in client.calls[n_calls:])

    fd.fetch_news(client, ["META"], start, end)
    assert "FB" in client.calls[-1][1]["symbols"]
    news = fd.load_news()
    assert news.loc[0, "symbols"] == ["META"]                  # FB alias mapped to META
    assert news.loc[0, "created_at"] == pd.Timestamp("2021-01-04T15:00:00Z")
