from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .providers import MarketDataProvider


@dataclass(frozen=True)
class MarketSymbol:
    ticker: str
    label: str


class MarketSentimentEngine:
    def __init__(self, market_data: MarketDataProvider | None = None) -> None:
        self.market_data = market_data or MarketDataProvider()
        self.nifty = MarketSymbol("^NSEI", "Nifty 50")
        self.banknifty = MarketSymbol("^NSEBANK", "Bank Nifty")
        self.vix = MarketSymbol("^INDIAVIX", "India VIX")

    def analyze(
        self,
        sector: str | None = None,
        fii_net_cr: float | None = None,
        dii_net_cr: float | None = None,
    ) -> dict[str, Any]:
        positives: list[str] = []
        negatives: list[str] = []
        warnings: list[str] = []
        score = 0
        components: dict[str, Any] = {}

        nifty_view = self._index_view(self.nifty)
        components["nifty"] = nifty_view
        score += self._score_index(nifty_view, positives, negatives, "Nifty")

        if self._is_financial_sector(sector):
            bank_view = self._index_view(self.banknifty)
            components["banknifty"] = bank_view
            score += self._score_index(bank_view, positives, negatives, "Bank Nifty", weight=0.75)

        vix_view = self._vix_view()
        components["vix"] = vix_view
        score += self._score_vix(vix_view, positives, negatives)

        flow_view = self._flow_view(fii_net_cr, dii_net_cr)
        components["flows"] = flow_view
        score += self._score_flows(flow_view, positives, negatives)

        available_score = 75 if self._is_financial_sector(sector) else 60
        normalized = int(round(max(0, min(100, (score / available_score) * 100))))
        mood = self._mood(normalized)
        threshold_adjustment = self._threshold_adjustment(normalized)
        risk_multiplier = self._risk_multiplier(normalized, vix_view)

        if any(component.get("status") == "unavailable" for component in components.values()):
            warnings.append("Some market sentiment data was unavailable; trade plan uses stricter caution.")
            threshold_adjustment = max(threshold_adjustment, 5)
            risk_multiplier = min(risk_multiplier, 0.75)

        return {
            "score": normalized,
            "mood": mood,
            "threshold_adjustment": threshold_adjustment,
            "risk_multiplier": risk_multiplier,
            "positives": positives,
            "negatives": negatives,
            "warnings": warnings,
            "components": components,
        }

    def _index_view(self, symbol: MarketSymbol) -> dict[str, Any]:
        data = self.market_data.price_history(symbol.ticker, period="1y", interval="1d")
        if data.empty or len(data) < 210:
            return {"label": symbol.label, "status": "unavailable"}

        close = data["Close"]
        view = pd.DataFrame({"Close": close}).dropna()
        view["SMA_20"] = view["Close"].rolling(20).mean()
        view["SMA_50"] = view["Close"].rolling(50).mean()
        view["SMA_200"] = view["Close"].rolling(200).mean()
        view["Return_5D"] = view["Close"].pct_change(5)
        view["Return_1M"] = view["Close"].pct_change(21)
        view = view.dropna()
        if view.empty:
            return {"label": symbol.label, "status": "unavailable"}

        latest = view.iloc[-1]
        return {
            "label": symbol.label,
            "status": "ok",
            "close": float(latest["Close"]),
            "sma_20": float(latest["SMA_20"]),
            "sma_50": float(latest["SMA_50"]),
            "sma_200": float(latest["SMA_200"]),
            "return_5d_pct": float(latest["Return_5D"] * 100),
            "return_1m_pct": float(latest["Return_1M"] * 100),
        }

    def _vix_view(self) -> dict[str, Any]:
        data = self.market_data.price_history(self.vix.ticker, period="6mo", interval="1d")
        if data.empty or len(data) < 25:
            return {"label": self.vix.label, "status": "unavailable"}

        close = data["Close"].dropna()
        latest = close.iloc[-1]
        previous_week = close.iloc[-6] if len(close) >= 6 else close.iloc[0]
        return {
            "label": self.vix.label,
            "status": "ok",
            "close": float(latest),
            "change_5d_pct": float((latest / previous_week - 1) * 100),
        }

    def _flow_view(self, fii_net_cr: float | None, dii_net_cr: float | None) -> dict[str, Any]:
        if fii_net_cr is None and dii_net_cr is None:
            return {"status": "unavailable", "fii_net_cr": None, "dii_net_cr": None, "total_net_cr": None}
        total = (fii_net_cr or 0) + (dii_net_cr or 0)
        return {
            "status": "ok",
            "fii_net_cr": fii_net_cr,
            "dii_net_cr": dii_net_cr,
            "total_net_cr": total,
        }

    def _score_index(
        self,
        view: dict[str, Any],
        positives: list[str],
        negatives: list[str],
        name: str,
        weight: float = 1.0,
    ) -> int:
        if view.get("status") != "ok":
            negatives.append(f"{name} trend data is unavailable.")
            return 0

        score = 0
        if view["close"] > view["sma_200"]:
            score += int(10 * weight)
            positives.append(f"{name} is above 200 DMA.")
        else:
            negatives.append(f"{name} is below 200 DMA.")

        if view["close"] > view["sma_50"] and view["sma_50"] > view["sma_200"]:
            score += int(10 * weight)
            positives.append(f"{name} is in a bullish trend structure.")
        else:
            negatives.append(f"{name} trend structure is not fully bullish.")

        if view["return_5d_pct"] > -1 and view["return_1m_pct"] > 0:
            score += int(5 * weight)
            positives.append(f"{name} short-term momentum is stable.")
        else:
            negatives.append(f"{name} short-term momentum is weak.")

        return score

    def _score_vix(self, view: dict[str, Any], positives: list[str], negatives: list[str]) -> int:
        if view.get("status") != "ok":
            negatives.append("India VIX data is unavailable.")
            return 0
        if view["close"] < 15 and view["change_5d_pct"] < 10:
            positives.append("India VIX is calm.")
            return 15
        if view["close"] < 20 and view["change_5d_pct"] < 20:
            positives.append("India VIX is acceptable, but not ideal.")
            return 8
        negatives.append("India VIX is elevated or rising fast.")
        return 0

    def _score_flows(self, view: dict[str, Any], positives: list[str], negatives: list[str]) -> int:
        if view.get("status") != "ok":
            negatives.append("FII/DII flow data was not supplied.")
            return 0
        total = view["total_net_cr"]
        if total > 1500:
            positives.append("FII/DII combined cash flow is strongly positive.")
            return 20
        if total > 0:
            positives.append("FII/DII combined cash flow is positive.")
            return 12
        if total > -1500:
            negatives.append("FII/DII combined cash flow is mildly negative.")
            return 5
        negatives.append("FII/DII combined cash flow is strongly negative.")
        return 0

    def _is_financial_sector(self, sector: str | None) -> bool:
        if not sector:
            return False
        return "financial" in sector.lower() or "bank" in sector.lower()

    def _mood(self, score: int) -> str:
        if score >= 75:
            return "Bullish"
        if score >= 55:
            return "Neutral to Positive"
        if score >= 40:
            return "Cautious"
        return "Risk-Off"

    def _threshold_adjustment(self, score: int) -> int:
        if score >= 75:
            return -5
        if score >= 55:
            return 0
        if score >= 40:
            return 5
        return 10

    def _risk_multiplier(self, score: int, vix_view: dict[str, Any]) -> float:
        if score >= 75:
            base = 1.0
        elif score >= 55:
            base = 0.8
        elif score >= 40:
            base = 0.5
        else:
            base = 0.0

        if vix_view.get("status") == "ok" and vix_view["close"] >= 20:
            base = min(base, 0.5)
        return base
