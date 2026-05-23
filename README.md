---
title: Market Agent India
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Indian Stock Market Research Agent

This is a local research agent for Indian equities. It is designed to be selective: it can skip trades unless technicals, fundamentals, valuation, institutional flow context, and news sentiment align.

Important: no strategy can honestly guarantee 90% accuracy. The app therefore uses a high-confluence scoring model and encourages backtesting before any real trade.

## Features

- Analyze NSE/BSE stocks by symbol, for example `RELIANCE`, `TCS`, `HDFCBANK`, `INFY`.
- Pull price history, returns, moving averages, RSI, ATR, volume trend, fundamentals, valuation ratios, and recent news.
- Estimate conservative swing-trade entry, stop loss, targets, position risk, and invalidation levels.
- Include FII/DII market-flow context via manual values.
- Score market sentiment using Nifty 50 trend, Bank Nifty trend for financial stocks, India VIX, and FII/DII flow.
- Adjust entry thresholds and position risk based on market sentiment.
- Find 80+ score stocks from a built-in liquid NSE universe, Nifty 50, or a live NSE listed-stock universe.
- Run a high-selectivity signal scan over a watchlist.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run src/app.py
```

## Notes

- Yahoo Finance data for Indian stocks uses suffixes such as `.NS` and `.BO`; the app adds `.NS` automatically if no suffix is provided.
- FII/DII stock-specific daily buying is generally not freely available from official exchanges. The app supports market-wide FII/DII flow context because NSE publishes it publicly.
- When market sentiment is weak, the app raises the required score, reduces position size, or blocks fresh long entries.
- Use the **Find Stocks** tab to scan without manually entering symbols. If no stock crosses 80, the agent will show no result instead of forcing a weak trade.
- Treat this as decision support, not financial advice.
