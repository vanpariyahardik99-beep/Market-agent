from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .providers import MarketDataProvider, NewsProvider
from .market_sentiment import MarketSentimentEngine
from .strategy import build_trade_plan, score_setup


@dataclass(frozen=True)
class AgentConfig:
    lookback_period: str = "2y"
    interval: str = "1d"
    risk_per_trade_pct: float = 1.0
    min_signal_score: int = 80


class IndianMarketAgent:
    def __init__(
        self,
        market_data: MarketDataProvider | None = None,
        news: NewsProvider | None = None,
        sentiment: MarketSentimentEngine | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.market_data = market_data or MarketDataProvider()
        self.news = news or NewsProvider()
        self.sentiment = sentiment or MarketSentimentEngine(self.market_data)
        self.config = config or AgentConfig()

    def analyze_stock(
        self,
        symbol: str,
        capital: float = 100000.0,
        fii_net_cr: float | None = None,
        dii_net_cr: float | None = None,
    ) -> dict[str, Any]:
        ticker = self.market_data.normalize_symbol(symbol)
        history = self.market_data.price_history(
            ticker,
            period=self.config.lookback_period,
            interval=self.config.interval,
        )
        if history.empty:
            raise ValueError(f"No price data found for {symbol}. Try NSE symbol like RELIANCE or ticker RELIANCE.NS.")

        indicators = self._add_indicators(history)
        quote = self.market_data.quote_summary(ticker)
        headlines = self.news.latest_headlines(symbol)
        market_sentiment = self.sentiment.analyze(
            sector=quote.get("sector"),
            fii_net_cr=fii_net_cr,
            dii_net_cr=dii_net_cr,
        )
        score = score_setup(indicators, quote, headlines, fii_net_cr=fii_net_cr, dii_net_cr=dii_net_cr)
        trade_plan = build_trade_plan(
            indicators,
            score=score,
            market_sentiment=market_sentiment,
            capital=capital,
            risk_per_trade_pct=self.config.risk_per_trade_pct,
            min_signal_score=self.config.min_signal_score,
        )

        return {
            "symbol": ticker,
            "company": quote.get("longName") or quote.get("shortName") or ticker,
            "price_data": indicators,
            "fundamentals": self._fundamental_view(quote),
            "valuation": self._valuation_view(quote, indicators),
            "news": headlines,
            "score": score,
            "market_sentiment": market_sentiment,
            "trade_plan": trade_plan,
            "institutional_context": {
                "fii_net_cr": fii_net_cr,
                "dii_net_cr": dii_net_cr,
                "interpretation": self._flow_interpretation(fii_net_cr, dii_net_cr),
            },
        }

    def scan_watchlist(
        self,
        symbols: list[str],
        capital: float = 100000.0,
        fii_net_cr: float | None = None,
        dii_net_cr: float | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            try:
                report = self.analyze_stock(symbol, capital=capital, fii_net_cr=fii_net_cr, dii_net_cr=dii_net_cr)
                plan = report["trade_plan"]
                rows.append(
                    {
                        "symbol": report["symbol"],
                        "company": report["company"],
                        "score": report["score"]["total"],
                        "market_score": report["market_sentiment"]["score"],
                        "market_mood": report["market_sentiment"]["mood"],
                        "signal": plan["signal"],
                        "entry": plan["entry"],
                        "stop_loss": plan["stop_loss"],
                        "target_1": plan["target_1"],
                        "risk_reward": plan["risk_reward"],
                        "reason": "; ".join(report["score"]["positives"][:3]),
                    }
                )
            except Exception as exc:  # keep scans useful even if one ticker fails
                rows.append({"symbol": symbol, "signal": "ERROR", "reason": str(exc)})
        return pd.DataFrame(rows).sort_values("score", ascending=False, na_position="last")

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        data["SMA_20"] = close.rolling(20).mean()
        data["SMA_50"] = close.rolling(50).mean()
        data["SMA_200"] = close.rolling(200).mean()
        data["EMA_21"] = close.ewm(span=21, adjust=False).mean()
        data["Volume_SMA_20"] = volume.rolling(20).mean()

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        data["RSI_14"] = 100 - (100 / (1 + rs))

        prev_close = close.shift(1)
        true_range = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        data["ATR_14"] = true_range.rolling(14).mean()
        data["Return_1M"] = close.pct_change(21)
        data["Return_3M"] = close.pct_change(63)
        data["Drawdown_1Y"] = close / close.rolling(252).max() - 1
        return data.dropna()

    def _fundamental_view(self, quote: dict[str, Any]) -> dict[str, Any]:
        fields = [
            "sector",
            "industry",
            "marketCap",
            "totalRevenue",
            "revenueGrowth",
            "grossMargins",
            "operatingMargins",
            "profitMargins",
            "returnOnEquity",
            "debtToEquity",
            "freeCashflow",
        ]
        return {field: quote.get(field) for field in fields}

    def _valuation_view(self, quote: dict[str, Any], data: pd.DataFrame) -> dict[str, Any]:
        latest = data.iloc[-1]
        return {
            "trailingPE": quote.get("trailingPE"),
            "forwardPE": quote.get("forwardPE"),
            "priceToBook": quote.get("priceToBook"),
            "enterpriseToEbitda": quote.get("enterpriseToEbitda"),
            "dividendYield": quote.get("dividendYield"),
            "current_price": float(latest["Close"]),
            "distance_from_200dma_pct": float(latest["Close"] / latest["SMA_200"] - 1),
            "one_year_drawdown_pct": float(latest["Drawdown_1Y"]),
        }

    def _flow_interpretation(self, fii_net_cr: float | None, dii_net_cr: float | None) -> str:
        if fii_net_cr is None and dii_net_cr is None:
            return "No FII/DII values supplied. Use this as stock-level analysis only."
        total = (fii_net_cr or 0) + (dii_net_cr or 0)
        if total > 1000:
            return "Market-wide institutional flows are supportive."
        if total < -1000:
            return "Market-wide institutional flows are a headwind; demand stricter entries."
        return "Institutional flow context is mixed or neutral."
