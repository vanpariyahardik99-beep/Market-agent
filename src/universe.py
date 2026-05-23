from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


NIFTY_50_SYMBOLS = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO",
]


LIQUID_NSE_SYMBOLS = [
    *NIFTY_50_SYMBOLS,
    "ABB",
    "ABCAPITAL",
    "ABFRL",
    "ACC",
    "ALKEM",
    "AMBUJACEM",
    "APLAPOLLO",
    "ASHOKLEY",
    "ASTRAL",
    "AUBANK",
    "AUROPHARMA",
    "BANDHANBNK",
    "BANKBARODA",
    "BERGEPAINT",
    "BHARATFORG",
    "BHEL",
    "BIOCON",
    "BOSCHLTD",
    "CANBK",
    "CGPOWER",
    "CHOLAFIN",
    "COLPAL",
    "CONCOR",
    "CUMMINSIND",
    "DABUR",
    "DALBHARAT",
    "DEEPAKNTR",
    "DIVISLAB",
    "DLF",
    "DMART",
    "FEDERALBNK",
    "GAIL",
    "GODREJCP",
    "GODREJPROP",
    "HAL",
    "HAVELLS",
    "IDFCFIRSTB",
    "INDHOTEL",
    "IOC",
    "IRCTC",
    "JSWENERGY",
    "LICHSGFIN",
    "LODHA",
    "LUPIN",
    "MFSL",
    "MPHASIS",
    "NAUKRI",
    "OBEROIRLTY",
    "OFSS",
    "PAYTM",
    "PEL",
    "PERSISTENT",
    "PETRONET",
    "PIDILITIND",
    "PNB",
    "POLYCAB",
    "RECLTD",
    "SAIL",
    "SIEMENS",
    "SRF",
    "TATACHEM",
    "TATACOMM",
    "TATAELXSI",
    "TVSMOTOR",
    "UNIONBANK",
    "UNITDSPR",
    "VOLTAS",
    "YESBANK",
    "ZYDUSLIFE",
]


NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"


def load_nse_equity_universe(limit: int | None = None, series: str = "EQ") -> list[str]:
    """Load NSE listed equity symbols, falling back to Nifty 50 if NSE is unavailable."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/csv,text/plain,*/*",
    }
    try:
        response = requests.get(NSE_EQUITY_LIST_URL, headers=headers, timeout=20)
        response.raise_for_status()
        listed = pd.read_csv(StringIO(response.text))
        if "SYMBOL" not in listed.columns:
            raise ValueError("NSE equity list did not include SYMBOL column.")
        if " SERIES" in listed.columns:
            listed = listed[listed[" SERIES"].astype(str).str.strip().eq(series)]
        symbols = listed["SYMBOL"].dropna().astype(str).str.strip()
        symbols = symbols.loc[symbols.ne("")].drop_duplicates().sort_values().tolist()
        if not symbols:
            raise ValueError("NSE equity list returned no symbols.")
        return symbols[:limit] if limit else symbols
    except Exception:
        return NIFTY_50_SYMBOLS[:limit] if limit else NIFTY_50_SYMBOLS
