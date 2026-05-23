from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.indian_market_agent import AgentConfig, IndianMarketAgent
from src.universe import LIQUID_NSE_SYMBOLS, NIFTY_50_SYMBOLS, load_nse_equity_universe


st.set_page_config(page_title="Indian Market Research Agent", layout="wide")


@st.cache_data(ttl=900)
def analyze(symbol: str, capital: float, risk: float, min_score: int, fii: float | None, dii: float | None):
    agent = IndianMarketAgent(config=AgentConfig(risk_per_trade_pct=risk, min_signal_score=min_score))
    return agent.analyze_stock(symbol, capital=capital, fii_net_cr=fii, dii_net_cr=dii)


@st.cache_data(ttl=900)
def scan(symbols: tuple[str, ...], capital: float, risk: float, min_score: int, fii: float | None, dii: float | None):
    agent = IndianMarketAgent(config=AgentConfig(risk_per_trade_pct=risk, min_signal_score=min_score))
    return agent.scan_watchlist(list(symbols), capital=capital, fii_net_cr=fii, dii_net_cr=dii)


@st.cache_data(ttl=900)
def find_stocks(
    symbols: tuple[str, ...],
    capital: float,
    risk: float,
    min_score: int,
    cutoff: int,
    fii: float | None,
    dii: float | None,
):
    result = scan(symbols, capital, risk, min_score, fii, dii)
    if "score" not in result.columns:
        return result
    numeric_score = pd.to_numeric(result["score"], errors="coerce")
    return result[numeric_score >= cutoff].sort_values(["score", "market_score"], ascending=False)


@st.cache_data(ttl=86400)
def nse_universe(limit: int):
    return load_nse_equity_universe(limit=limit)


def maybe_number(value: str) -> float | None:
    if value.strip() == "":
        return None
    return float(value)


st.title("Indian Stock Market Research Agent")

with st.sidebar:
    st.subheader("Controls")
    capital = st.number_input("Capital", min_value=1000.0, value=100000.0, step=5000.0)
    risk = st.slider("Risk per trade (%)", min_value=0.25, max_value=3.0, value=1.0, step=0.25)
    min_score = st.slider("Minimum signal score", min_value=60, max_value=95, value=80, step=5)
    fii_text = st.text_input("FII net cash market, Cr", placeholder="Optional")
    dii_text = st.text_input("DII net cash market, Cr", placeholder="Optional")
    fii = maybe_number(fii_text) if fii_text else None
    dii = maybe_number(dii_text) if dii_text else None

tab_stock, tab_find, tab_scan = st.tabs(["Stock Analysis", "Find Stocks", "Watchlist Scan"])

with tab_stock:
    symbol = st.text_input("Stock symbol", value="RELIANCE")
    if st.button("Analyze", type="primary"):
        with st.spinner("Collecting market data, fundamentals, news, and signal score..."):
            report = analyze(symbol, capital, risk, min_score, fii, dii)

        price_data = report["price_data"]
        latest = price_data.iloc[-1]
        plan = report["trade_plan"]
        score = report["score"]
        market = report["market_sentiment"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Company", report["company"])
        c2.metric("Close", f"{latest['Close']:.2f}")
        c3.metric("Score", f"{score['total']}/100")
        c4.metric("Market", f"{market['score']}/100")
        c5.metric("Signal", plan["signal"])

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=price_data.index,
            open=price_data["Open"],
            high=price_data["High"],
            low=price_data["Low"],
            close=price_data["Close"],
            name="Price",
        ))
        fig.add_trace(go.Scatter(x=price_data.index, y=price_data["EMA_21"], name="21 EMA"))
        fig.add_trace(go.Scatter(x=price_data.index, y=price_data["SMA_50"], name="50 DMA"))
        fig.add_trace(go.Scatter(x=price_data.index, y=price_data["SMA_200"], name="200 DMA"))
        fig.update_layout(height=520, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Entry Exit Plan")
        st.json(plan)

        st.subheader("Market Sentiment")
        m1, m2, m3 = st.columns(3)
        m1.metric("Mood", market["mood"])
        m2.metric("Risk Multiplier", f"{market['risk_multiplier']:.2f}x")
        m3.metric("Threshold Adjustment", market["threshold_adjustment"])
        st.write(pd.DataFrame({"positive market factor": market["positives"]}))
        st.write(pd.DataFrame({"market risk": market["negatives"] + market["warnings"]}))
        st.json(market["components"])

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Positive Factors")
            st.write(pd.DataFrame({"factor": score["positives"]}))
        with col_b:
            st.subheader("Risks / Negatives")
            st.write(pd.DataFrame({"factor": score["negatives"]}))

        st.subheader("Fundamentals")
        st.write(pd.Series(report["fundamentals"]).to_frame("value"))

        st.subheader("Valuation")
        st.write(pd.Series(report["valuation"]).to_frame("value"))

        st.subheader("FII/DII Context")
        st.write(report["institutional_context"])

        st.subheader("Latest News")
        for item in report["news"]:
            st.markdown(f"- [{item['title']}]({item['link']})  \n  {item['published']}")

with tab_find:
    st.subheader("Find 80+ Score Stocks")
    cutoff = st.slider("Show stocks with score at least", min_value=70, max_value=95, value=80, step=5)
    universe_source = st.radio(
        "Universe",
        options=["Liquid NSE", "Nifty 50", "NSE listed stocks"],
        horizontal=True,
    )
    source_sizes = {
        "Liquid NSE": len(LIQUID_NSE_SYMBOLS),
        "Nifty 50": len(NIFTY_50_SYMBOLS),
        "NSE listed stocks": 300,
    }
    max_limit = source_sizes[universe_source]
    default_limit = min(75 if universe_source == "Liquid NSE" else 50, max_limit)
    max_stocks = st.slider("Maximum stocks to scan", min_value=10, max_value=max_limit, value=default_limit, step=10)
    if universe_source == "NSE listed stocks":
        symbols = tuple(nse_universe(max_stocks))
    elif universe_source == "Liquid NSE":
        symbols = tuple(LIQUID_NSE_SYMBOLS[:max_stocks])
    else:
        symbols = tuple(NIFTY_50_SYMBOLS[:max_stocks])
    st.caption(f"Ready to scan {len(symbols)} stocks from {universe_source}.")
    if st.button("Find Stocks", type="primary"):
        with st.spinner("Scanning for high-score stocks. This can take a few minutes for a large universe..."):
            result = find_stocks(symbols, capital, risk, min_score, cutoff, fii, dii)
        if result.empty:
            st.warning("No stocks crossed the selected score. This is good discipline: no forced trades.")
        else:
            st.success(f"Found {len(result)} stock(s) with score >= {cutoff}.")
            st.dataframe(result, use_container_width=True, hide_index=True)

with tab_scan:
    watchlist = st.text_area("Watchlist", value="RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, LT, SBIN")
    if st.button("Scan Watchlist", type="primary"):
        symbols = tuple(item.strip() for item in watchlist.replace("\n", ",").split(",") if item.strip())
        with st.spinner("Scanning only for high-confluence setups..."):
            result = scan(symbols, capital, risk, min_score, fii, dii)
        st.dataframe(result, use_container_width=True, hide_index=True)
