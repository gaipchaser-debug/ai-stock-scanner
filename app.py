import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import difflib

# FinanceDataReader 임포트
try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except:
    FDR_AVAILABLE = False

st.set_page_config(page_title="전문가급 주식 분석 시스템", page_icon="📊", layout="wide")

# ========== 세션 스테이트 ==========
if 'stock_list' not in st.session_state:
    st.session_state.stock_list = None
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'scan_mode' not in st.session_state:
    st.session_state.scan_mode = None
if 'radar_results' not in st.session_state:
    st.session_state.radar_results = None
if 'radar_scan_mode' not in st.session_state:
    st.session_state.radar_scan_mode = None
if 'radar_kospi_change' not in st.session_state:
    st.session_state.radar_kospi_change = None
if 'radar_kospi_current' not in st.session_state:
    st.session_state.radar_kospi_current = None
if 'div_scan_results' not in st.session_state:
    st.session_state.div_scan_results = None

def reset_session():
    st.cache_data.clear()

@st.cache_data(ttl=86400)
def load_all_korean_stocks():
    """한국 주식 전체 리스트 로드"""
    try:
        if FDR_AVAILABLE:
            kospi = fdr.StockListing('KOSPI')
            kospi['Market'] = 'KOSPI'
            kosdaq = fdr.StockListing('KOSDAQ')
            kosdaq['Market'] = 'KOSDAQ'
            all_stocks = pd.concat([kospi, kosdaq], ignore_index=True)
            stock_dict = {}
            for _, row in all_stocks.iterrows():
                code = str(row['Code'])
                name = str(row['Name']).lower().strip()
                market = str(row['Market'])
                ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
                stock_dict[name] = ticker
                stock_dict[code] = ticker
            return stock_dict, all_stocks
        return {}, None
    except Exception as e:
        st.error(f"종목 리스트 로드 실패: {str(e)}")
        return {}, None

def search_stock(query, stock_dict, all_stocks_df):
    query_raw = str(query).strip()
    query_lower = query_raw.lower()
    if query_raw.isdigit():
        code_padded = query_raw.zfill(6)
        if code_padded in stock_dict: return stock_dict[code_padded], None, 'exact'
        if query_raw in stock_dict: return stock_dict[query_raw], None, 'exact'
    q_clean = query_lower.replace(" ", "").replace("(", "").replace(")", "")
    if query_lower in stock_dict: return stock_dict[query_lower], None, 'exact'
    if q_clean in stock_dict: return stock_dict[q_clean], None, 'exact'
    if all_stocks_df is not None:
        names_lower = all_stocks_df['Name'].str.lower().str.strip()
        mask_partial = names_lower.str.contains(query_lower, na=False, regex=False)
        partial_matches = all_stocks_df[mask_partial]
        if len(partial_matches) == 1:
            code = str(partial_matches.iloc[0]['Code'])
            market = str(partial_matches.iloc[0]['Market'])
            return f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ", None, 'exact'
        elif len(partial_matches) > 1:
            return None, partial_matches.head(10).reset_index(drop=True), 'partial'
        for cutoff in [0.55, 0.40, 0.30]:
            close_names = difflib.get_close_matches(query_lower, names_lower.tolist(), n=10, cutoff=cutoff)
            if close_names: break
        if close_names:
            sim_mask = names_lower.isin(close_names)
            similar_df = all_stocks_df[sim_mask].copy()
            similar_df['_sim'] = similar_df['Name'].str.lower().apply(lambda n: difflib.SequenceMatcher(None, query_lower, n).ratio())
            return None, similar_df.sort_values('_sim', ascending=False).drop(columns=['_sim']).head(10).reset_index(drop=True), 'similar'
    return None, None, 'notfound' 

def load_stock_data(ticker, max_retries=2):
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            for period in ["3mo", "2mo"]:
                hist = stock.history(period=period)
                if not hist.empty and len(hist) >= 20:
                    return True, stock.info.get('longName', ticker), float(hist['Close'].iloc[-1]), hist
        except: time.sleep(0.5)
    return False, None, None, None

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + (gain / loss)))

def detect_candle_pattern_advanced(hist):
    if len(hist) < 15: return "데이터 부족", 50, 0, {}
    hist['RSI'] = calculate_rsi(hist, period=14)
    c, o, h, l, v = hist['Close'].iloc[-5:].values, hist['Open'].iloc[-5:].values, hist['High'].iloc[-5:].values, hist['Low'].iloc[-5:].values, hist['Volume'].iloc[-5:].values
    last_body, prev_body = abs(c[-1] - o[-1]), abs(c[-2] - o[-2])
    avg_volume = hist['Volume'].tail(20).mean()
    volume_ratio = float(v[-1]) / avg_volume if avg_volume > 0 else 1
    current_rsi = float(hist['RSI'].iloc[-1]) if pd.notna(hist['RSI'].iloc[-1]) else 50
    high_20, low_20 = hist['Close'].tail(20).max(), hist['Close'].tail(20).min()
    price_pos = (c[-1] - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
    details = {'rsi': current_rsi, 'volume_ratio': volume_ratio, 'price_position': price_pos * 100}

    if (c[-3]>o[-3] and c[-2]>o[-2] and c[-1]>o[-1] and c[-3]<c[-2]<c[-1] and o[-3]<o[-2]<c[-3] and o[-2]<o[-1]<c[-2]):
        return "적삼병 (사카타5법) 🚀", 95, int(min(100, (volume_ratio*20) + (100-current_rsi)*0.5 + 40)), details
    if (c[-3]<o[-3] and c[-2]<o[-2] and c[-1]<o[-1] and c[-3]>c[-2]>c[-1] and c[-3]<o[-2]<o[-3] and c[-2]<o[-1]<o[-2]):
        return "흑삼병 (사카타5법) ⚠️", 10, int(min(100, current_rsi*0.8 + 40)), details
    if (c[-5]>o[-5] and c[-4]<o[-4] and c[-3]<o[-3] and c[-2]<o[-2] and c[-5]>max(c[-4],c[-3],c[-2]) and o[-5]<min(c[-4],c[-3],c[-2]) and c[-1]>o[-1] and c[-1]>c[-5]):
        return "상승 삼법 (사카타5법) 📈", 90, 85, details
    if (c[-2]<o[-2] and c[-1]>o[-1] and last_body>prev_body*1.5 and c[-1]>o[-2] and o[-1]<c[-2]):
        return "상승 장악형 (Bullish Engulfing) 🟢", 80, int(min(30, (last_body/prev_body)*10) + (20 if 30<=current_rsi<=50 else 0) + min(25, volume_ratio*12.5)), details
    
    lower_shadow, upper_shadow = min(o[-1], c[-1]) - l[-1], h[-1] - max(o[-1], c[-1])
    if last_body > 0 and lower_shadow > last_body*2 and upper_shadow < last_body*0.5:
        return "해머형 (망치형) 🟢", 75, int(min(35, (lower_shadow/last_body)*12) + (25 if current_rsi<35 else 5) + min(20, volume_ratio*10)), details
    return "일반 캔들 (패턴 없음)", 50, 0, details

def calculate_stock_score(hist, current_price, vs_kospi=None, verdict=None):
    try:
        if len(hist) < 20: return 0, {}
        hist = hist.copy()
        hist['MA5'], hist['MA20'], hist['MA60'], hist['MA120'] = hist['Close'].rolling(5).mean(), hist['Close'].rolling(20).mean(), hist['Close'].rolling(60).mean(), hist['Close'].rolling(120).mean()
        latest = hist.iloc[-1]
        ma5, ma20, ma60, ma120 = float(latest['MA5']) if pd.notna(latest['MA5']) else current_price, float(latest['MA20']) if pd.notna(latest['MA20']) else current_price, float(latest['MA60']) if pd.notna(latest['MA60']) else current_price, float(latest['MA120']) if pd.notna(latest['MA120']) else current_price

        pattern_name, candle_score, _, _ = detect_candle_pattern_advanced(hist)
        align_count = sum([ma5>ma20, ma20>ma60, ma60>ma120, ma5>ma60, ma20>ma120])
        ma_score = {5:92, 4:78, 3:62, 2:46, 1:32, 0:15}[align_count]
        cross_score = 50
        if len(hist) >= 5 and ma20 > ma
