from __future__ import annotations

import re
from typing import Any

import feedparser
import pandas as pd
import yfinance as yf


class MarketDataProvider:
    def normalize_symbol(self, symbol: str) -> str:
        clean = symbol.strip().upper()
        if clean.endswith((".NS", ".BO")):
            return clean
        return f"{clean}.NS"

    def price_history(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.dropna(how="all")

    def quote_summary(self, ticker: str) -> dict[str, Any]:
        stock = yf.Ticker(ticker)
        try:
            return dict(stock.info or {})
        except Exception:
            return {}


class NewsProvider:
    def latest_headlines(self, symbol: str, limit: int = 8) -> list[dict[str, str]]:
        query = re.sub(r"[^A-Za-z0-9 ]+", " ", symbol).strip()
        feed_url = f"https://news.google.com/rss/search?q={query}%20NSE%20stock%20India&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(feed_url)
        headlines: list[dict[str, str]] = []
        for entry in feed.entries[:limit]:
            headlines.append(
                {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                }
            )
        return headlines
