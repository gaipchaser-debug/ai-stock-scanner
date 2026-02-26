import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import difflib

# [필독] Streamlit 설정은 반드시 코드의 최상단에 위치해야 합니다.
st.set_page_config(page_title="전문가급 주식 분석 시스템", page_icon="📊", layout="wide")

# ========== 1. 데이터 로드 및 유틸리티 함수 (기존 코드 유지) ==========

try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except:
    FDR_AVAILABLE = False

@st.cache_data(ttl=86400)
def load_all_korean_stocks():
    try:
        if FDR_AVAILABLE:
            kospi = fdr.StockListing('KOSPI')
            kospi['Market'] = 'KOSPI'
            kosdaq = fdr.StockListing('KOSDAQ')
            kosdaq['Market'] = 'KOSDAQ'
            all_stocks = pd.concat([kospi, kosdaq], ignore_index=True)
            stock_dict = {}
            for _, row in all_stocks.iterrows():
                code, name, market = str(row['Code']), str(row['Name']).lower().strip(), str(row['Market'])
                ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
                stock_dict[name], stock_dict[code] = ticker, ticker
            return stock_dict, all_stocks
        return {}, None
    except Exception as e:
        st.error(f"종목 리스트 로드 실패: {str(e)}")
        return {}, None

# (중략: search_stock, load_stock_data, calculate_stock_score 등 기존 분석 로직 포함됨)
# ... [기존 app.py의 분석 관련 함수들이 여기에 위치합니다] ...

# ========== 2. 세션 스테이트 관리 ==========
if 'stock_list' not in st.session_state:
    with st.spinner("📡 전체 종목 리스트 로딩 중..."):
        st.session_state.stock_list = load_all_korean_stocks()

stock_dict, all_stocks_df = st.session_state.stock_list

# ========== 3. 메인 UI 레이아웃 (4개 탭 구성) ==========
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🇰🇷 한국 주식 전체 검색 + AI 기반 4대 모듈 분석")

# 오류의 원인이었던 탭 정의 부분입니다. 변수 4개와 리스트 요소 4개를 정확히 일치시켰습니다.
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 시장 레이더", 
    "🎯 투자 적합 종목 추천", 
    "🔍 개별 종목 분석", 
    "🎁 배당주 투자 가이드"
])

# ----- TAB 1: 시장 레이더 (기존 소스코드) -----
with tab1:
    st.subheader("📡 오늘의 시장 레이더")
    # [기존 Tab1 코드 내용 삽입]
    st.info("시총 상위 종목의 실시간 등락률을 코스피와 비교 분석합니다.")

# ----- TAB 2: 투자 적합 종목 추천 (기존 소스코드) -----
with tab2:
    st.subheader("🎯 투자 적합 종목 추천")
    # [기존 Tab2 코드 내용 삽입]
    st.info("M1~M5 모듈을 기반으로 최적의 매수 후보군을 스캔합니다.")

# ----- TAB 3: 개별 종목 분석 (기존 소스코드) -----
with tab3:
    st.subheader("🔍 개별 종목 분석")
    # [기존 Tab3 코드 내용 삽입]
    st.info("특정 종목의 기술적 지표와 리스크:리워드를 상세 분석합니다.")

# ----- TAB 4: 배당주 투자 가이드 (신규 추가) -----
with tab4:
    st.subheader("🎁 배당주 투자 가이드 (과거 데이터 백테스팅)")
    st.markdown("""
    > 💡 **통계 기반 전략**: 과거 10년치 배당락 데이터를 시뮬레이션하여 최적의 매수/매도 시점을 도출합니다.
    """)

    # 1. 컨트롤 패널
    st.markdown("### ⚙️ 시뮬레이션 설정")
    c_p1, c_p2, c_p3 = st.columns([2, 2, 1])
    with c_p1:
        div_stock = st.text_input("분석할 배당주 입력", placeholder="예: 삼성전자, 맥쿼리인프라", key="div_input")
    with c_p2:
