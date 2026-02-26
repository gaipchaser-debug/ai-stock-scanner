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
