import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- Core Screener Functions ---

def calculate_technical_filters(df: pd.DataFrame, period: int = 14) -> dict:
    if len(df) < period * 2: return {}
    high, low, close = df["High"], df["Low"], df["Close"]

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    up_move, down_move = high - high.shift(1), low.shift(1) - low
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    pos_di = 100 * (pd.Series(pos_dm, index=df.index).rolling(period).mean() / atr)
    neg_di = 100 * (pd.Series(neg_dm, index=df.index).rolling(period).mean() / atr)

    dx = (100 * (pos_di - neg_di).abs() / (pos_di + neg_di)).fillna(0)
    adx = dx.rolling(window=period).mean()

    sma20, std20 = close.rolling(window=20).mean(), close.rolling(window=20).std()
    bb_width = ((sma20 + (2 * std20)) - (sma20 - (2 * std20))) / sma20

    return {
        "current_price": close.iloc[-1],
        "adx": adx.iloc[-1],
        "bb_width_percentile": (bb_width.rank(pct=True).iloc[-1] * 100)
    }

def get_pricing(leg: pd.Series, is_buy: bool) -> float:
    price = leg['ask'] if is_buy else leg['bid']
    return price if price > 0 else leg['lastPrice']

def analyze_call_butterfly(asset: yf.Ticker, current_price: float) -> dict:
    expirations = asset.options
    if not expirations: return {}

    today = datetime.today().date()
    valid_expirations = [(exp, (datetime.strptime(exp, "%Y-%m-%d").date() - today).days) 
                         for exp in expirations if 0 < (datetime.strptime(exp, "%Y-%m-%d").date() - today).days <= 30]
            
    if not valid_expirations: return {}
    expiry, dte = valid_expirations[-1]
    
    try:
        chain = asset.option_chain(expiry)
        calls = chain.calls
        available_strikes = calls['strike'].values
        if len(available_strikes) < 5: return {}

        atm_strike = available_strikes[np.abs(available_strikes - current_price).argmin()]
        wing_width = np.median(np.diff(available_strikes)) * 2  
        lower_strike, upper_strike = atm_strike - wing_width, atm_strike + wing_width

        if lower_strike not in available_strikes or upper_strike not in available_strikes: return {}

        lower_leg = calls[calls['strike'] == lower_strike].iloc[0]
        center_leg = calls[calls['strike'] == atm_strike].iloc[0]
        upper_leg = calls[calls['strike'] == upper_strike].iloc[0]

        net_debit = get_pricing(lower_leg, True) - (2 * get_pricing(center_leg, False)) + get_pricing(upper_leg, True)
        if net_debit <= 0: return {}

        max_profit = wing_width - net_debit
        return {
            "Expiry": expiry, "DTE": dte, "Strikes (L/C/U)": f"{lower_strike}/{atm_strike}/{upper_strike}",
            "Width": wing_width, "Net Debit": round(net_debit, 2), "Max Profit": round(max_profit, 2), "R:R": round(max_profit / net_debit, 2)
        }
    except Exception: return {}

def screen_options_universe(tickers: list[str]) -> pd.DataFrame:
    candidates = []
    for ticker in tickers:
        try:
            asset = yf.Ticker(ticker)
            hist = asset.history(period="6mo")
            if hist.empty: continue

            metrics = calculate_technical_filters(hist)
            if metrics and metrics["adx"] <= 24.0 and metrics["bb_width_percentile"] <= 30.0:
                butterfly_metrics = analyze_call_butterfly(asset, metrics["current_price"])
                if butterfly_metrics:
                    candidates.append({
                        "Ticker": ticker, "Price": round(metrics["current_price"], 2),
                        "ADX": round(metrics["adx"], 1), "BBW %": round(metrics["bb_width_percentile"], 1),
                        **butterfly_metrics
                    })
        except Exception: continue
    return pd.DataFrame(candidates).sort_values(by="R:R", ascending=False) if candidates else pd.DataFrame()

# --- Streamlit UI ---

st.set_page_config(page_title="Butterfly Screener", layout="wide")
st.title("🦋 Butterfly Options Screener")
st.markdown("Scans for consolidation (ADX < 24, BBW < 30%) and calculates 1-2-1 Call Butterfly R:R under 30 DTE.")

default_tickers = "SPY, QQQ, IWM, AAPL, MSFT, GOOGL, JNJ, XOM, PG, KO, PFE, WMT, IBM, CSCO, TSLA, NVDA, META, AMZN, AMD, NFLX, BA, DIS, V, MA, UNH, HD, GLD, SLV, TLT, HYG, XLF, XLE, SMH, ARKK"
watchlist = st.text_area("Watchlist (comma-separated)", default_tickers)

if st.button("Run Screener", type="primary"):
    tickers = [x.strip().upper() for x in watchlist.split(",")]
    
    with st.spinner(f"Screening {len(tickers)} tickers. This may take a minute..."):
        results = screen_options_universe(tickers)
        
        if not results.empty:
            st.success("Screening Complete!")
            st.dataframe(results, use_container_width=True, hide_index=True)
        else:
            st.warning("No candidates passed both the technical and options pricing filters.")