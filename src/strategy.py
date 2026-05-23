from __future__ import annotations

from typing import Any

import pandas as pd


def score_setup(
    data: pd.DataFrame,
    fundamentals: dict[str, Any],
    news: list[dict[str, str]],
    fii_net_cr: float | None = None,
    dii_net_cr: float | None = None,
) -> dict[str, Any]:
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    positives: list[str] = []
    negatives: list[str] = []
    score = 0

    if latest["Close"] > latest["SMA_200"]:
        score += 15
        positives.append("Price is above 200 DMA, so long-term trend is positive.")
    else:
        negatives.append("Price is below 200 DMA.")

    if latest["SMA_50"] > latest["SMA_200"]:
        score += 12
        positives.append("50 DMA is above 200 DMA.")
    else:
        negatives.append("50 DMA is below 200 DMA.")

    if latest["Close"] > latest["EMA_21"] and latest["EMA_21"] > latest["SMA_50"]:
        score += 13
        positives.append("Short-term trend is aligned above 21 EMA and 50 DMA.")
    else:
        negatives.append("Short-term trend is not fully aligned.")

    if 48 <= latest["RSI_14"] <= 67:
        score += 12
        positives.append("RSI is bullish but not overheated.")
    elif latest["RSI_14"] > 75:
        negatives.append("RSI is overheated; chase risk is high.")
    else:
        negatives.append("RSI is weak or stretched.")

    if latest["Volume"] > latest["Volume_SMA_20"] * 1.15 and latest["Close"] > previous["Close"]:
        score += 10
        positives.append("Up move came with above-average volume.")
    else:
        negatives.append("Volume confirmation is weak.")

    if latest["Return_3M"] > 0.08:
        score += 8
        positives.append("Three-month relative momentum is healthy.")
    else:
        negatives.append("Three-month momentum is not strong.")

    profit_margin = fundamentals.get("profitMargins")
    roe = fundamentals.get("returnOnEquity")
    debt_to_equity = fundamentals.get("debtToEquity")
    revenue_growth = fundamentals.get("revenueGrowth")

    if profit_margin is not None and profit_margin > 0.08:
        score += 8
        positives.append("Profit margin is acceptable.")
    else:
        negatives.append("Profit margin data is weak or unavailable.")

    if roe is not None and roe > 0.12:
        score += 8
        positives.append("Return on equity is healthy.")
    else:
        negatives.append("ROE is weak or unavailable.")

    if debt_to_equity is not None and debt_to_equity < 120:
        score += 5
        positives.append("Debt-to-equity is not excessive.")
    else:
        negatives.append("Debt level needs caution or data is unavailable.")

    if revenue_growth is not None and revenue_growth > 0.06:
        score += 5
        positives.append("Revenue growth is positive.")
    else:
        negatives.append("Revenue growth is weak or unavailable.")

    flow_total = (fii_net_cr or 0) + (dii_net_cr or 0)
    if fii_net_cr is not None or dii_net_cr is not None:
        if flow_total > 1000:
            score += 4
            positives.append("FII/DII market-wide net flow is supportive.")
        elif flow_total < -1000:
            negatives.append("FII/DII market-wide net flow is negative.")

    bad_news_words = ("fraud", "probe", "default", "downgrade", "loss", "penalty", "raid")
    negative_headlines = [item["title"] for item in news if any(word in item["title"].lower() for word in bad_news_words)]
    if not negative_headlines:
        score += 5
        positives.append("No obvious severe negative headline detected.")
    else:
        negatives.append("Negative news words detected in recent headlines.")

    return {
        "total": min(score, 100),
        "positives": positives,
        "negatives": negatives,
        "latest": {
            "close": float(latest["Close"]),
            "rsi": float(latest["RSI_14"]),
            "atr": float(latest["ATR_14"]),
            "volume": int(latest["Volume"]),
        },
    }


def build_trade_plan(
    data: pd.DataFrame,
    score: dict[str, Any],
    market_sentiment: dict[str, Any] | None,
    capital: float,
    risk_per_trade_pct: float,
    min_signal_score: int,
) -> dict[str, Any]:
    latest = data.iloc[-1]
    close = float(latest["Close"])
    atr = float(latest["ATR_14"])
    swing_low = float(data["Low"].tail(10).min())
    stop_loss = min(close - 1.5 * atr, swing_low * 0.995)
    risk_per_share = max(close - stop_loss, 0.01)
    market_sentiment = market_sentiment or {
        "score": 50,
        "mood": "Unknown",
        "threshold_adjustment": 5,
        "risk_multiplier": 0.75,
        "negatives": ["Market sentiment data is unavailable."],
        "warnings": ["Market sentiment data is unavailable."],
    }
    effective_min_score = min_signal_score + market_sentiment["threshold_adjustment"]
    adjusted_risk_pct = risk_per_trade_pct * market_sentiment["risk_multiplier"]
    rupee_risk = capital * (adjusted_risk_pct / 100)
    quantity = int(rupee_risk // risk_per_share)
    target_1 = close + 2 * risk_per_share
    target_2 = close + 3 * risk_per_share
    signal = "BUY WATCHLIST" if score["total"] >= effective_min_score and adjusted_risk_pct > 0 else "NO TRADE"

    if signal == "NO TRADE":
        entry = None
        rationale = (
            "Score is below the market-adjusted threshold or market sentiment is risk-off. "
            "Wait for better stock confirmation and supportive Nifty/FII-DII/VIX context."
        )
    else:
        entry = round(close, 2)
        rationale = (
            "High-confluence setup with acceptable market sentiment. "
            "Use limit entry near current price or wait for pullback to 21 EMA."
        )

    return {
        "signal": signal,
        "entry": entry,
        "market_mood": market_sentiment["mood"],
        "market_score": market_sentiment["score"],
        "base_min_score": min_signal_score,
        "market_adjusted_min_score": effective_min_score,
        "adjusted_risk_pct": round(adjusted_risk_pct, 2),
        "stop_loss": round(stop_loss, 2),
        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),
        "risk_reward": "1:2 to first target, 1:3 to second target",
        "position_size_qty": quantity,
        "max_loss_at_stop": round(quantity * risk_per_share, 2),
        "exit_plan": (
            "Book partial profit near target 1, trail remaining position below 21 EMA or previous swing low. "
            "Exit faster if Nifty loses 50 DMA, India VIX spikes, or FII/DII flows turn strongly negative."
        ),
        "invalid_if": "Daily close below stop loss, major negative news, Nifty below 50 DMA, or risk-off market sentiment.",
        "rationale": rationale,
    }
