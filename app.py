import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
import time
import re
import requests
from io import BytesIO

st.set_page_config(page_title="전문가급 주식 분석 시스템", page_icon="📊", layout="wide")

# ========== 세션 스테이트 초기화 ==========
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'force_analyze' not in st.session_state:
    st.session_state.force_analyze = False
if 'krx_data' not in st.session_state:
    st.session_state.krx_data = None

def reset_session():
    """세션 초기화"""
    st.cache_data.clear()
    st.session_state.force_analyze = False

@st.cache_data(ttl=86400)  # 24시간 캐시
def load_krx_stock_list():
    """KRX 전체 상장 종목 리스트 로드 (코스피 + 코스닥)"""
    try:
        # 네이버 금융에서 전체 종목 코드 가져오기
        kospi_url = "https://api.stock.naver.com/stock/exchange/STOCK/marketValue"
        kosdaq_url = "https://api.stock.naver.com/stock/exchange/KOSDAQ/marketValue"
        
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://finance.naver.com/'
        }
        
        all_stocks = {}
        
        # 코스피
        try:
            response = requests.get(kospi_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'stocks' in data:
                    for stock in data['stocks']:
                        code = stock.get('itemCode', '')
                        name = stock.get('stockName', '')
                        if code and name:
                            all_stocks[name.lower()] = f"{code}.KS"
                            all_stocks[code] = f"{code}.KS"
        except:
            pass
        
        # 코스닥
        try:
            response = requests.get(kosdaq_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'stocks' in data:
                    for stock in data['stocks']:
                        code = stock.get('itemCode', '')
                        name = stock.get('stockName', '')
                        if code and name:
                            all_stocks[name.lower()] = f"{code}.KQ"
                            all_stocks[code] = f"{code}.KQ"
        except:
            pass
        
        # 폴백: 주요 종목 수동 리스트 (API 실패 시)
        if len(all_stocks) < 100:
            major_stocks = {
                "삼성전자": "005930.KS", "sk하이닉스": "000660.KS", "lg전자": "066570.KS",
                "삼성바이오로직스": "207940.KS", "카카오": "035720.KS", "네이버": "035420.KS",
                "셀트리온": "068270.KS", "현대차": "005380.KS", "기아": "000270.KS",
                "삼성sdi": "006400.KS", "포스코홀딩스": "005490.KS", "lg화학": "051910.KS",
                "삼성전기": "009150.KS", "sk이노베이션": "096770.KS", "현대모비스": "012330.KS",
                "kb금융": "105560.KS", "신한지주": "055550.KS", "하나금융지주": "086790.KS",
                "삼성물산": "028260.KS", "포스코퓨처엠": "003670.KS", "lg에너지솔루션": "373220.KS",
                "카카오뱅크": "323410.KS", "크래프톤": "259960.KS", "엔씨소프트": "036570.KS",
                "sk텔레콤": "017670.KS", "kt": "030200.KS", "하이브": "352820.KS",
                "셀트리온헬스케어": "091990.KS", "삼성생명": "032830.KS", "삼성화재": "000810.KS",
                "한국전력": "015760.KS", "대한항공": "003490.KS", "아모레퍼시픽": "090430.KS",
                "lg생활건강": "051900.KS", "현대건설": "000720.KS", "gs건설": "006360.KS",
                "두산에너빌리티": "034020.KS", "한화에어로스페이스": "012450.KS", 
                "에코프로": "086520.KS", "에코프로비엠": "247540.KS", "엘앤에프": "066970.KS",
                "포스코인터내셔널": "047050.KS", "삼성중공업": "010140.KS", "한국조선해양": "009540.KS",
                "lg디스플레이": "034220.KS", "sk스퀘어": "402340.KS", "카카오게임즈": "293490.KS",
                "넷마블": "251270.KS", "펄어비스": "263750.KS", "위메이드": "112040.KQ",
                "컴투스": "078340.KQ", "유한양행": "000100.KS", "한미약품": "128940.KS",
                "대웅제약": "069620.KS", "녹십자": "006280.KS", "씨젠": "096530.KS",
                "솔브레인": "357780.KS", "티씨케이": "064760.KS", "원익ips": "240810.KS",
                "피에스케이": "319660.KS", "주성엔지니어링": "036930.KS", "원익머트리얼즈": "104830.KS",
                "sk": "034730.KS", "롯데케미칼": "011170.KS", "한화솔루션": "009830.KS",
                "s-oil": "010950.KS", "현대글로비스": "086280.KS", "cj제일제당": "097950.KS",
                "신세계": "004170.KS", "롯데쇼핑": "023530.KS", "이마트": "139480.KS",
                "우리금융지주": "316140.KS", "한국금융지주": "071050.KS", "키움증권": "039490.KS",
                "미래에셋증권": "006800.KS", "현대해상": "001450.KS", "한화생명": "088350.KS",
                "대림산업": "000210.KS", "gs리테일": "007070.KS", "롯데지주": "004990.KS",
                "현대위아": "011210.KS", "만도": "204320.KS", "한온시스템": "018880.KS",
                "lg유플러스": "032640.KS", "제일기획": "030000.KS", "한진칼": "180640.KS",
                "cj대한통운": "000120.KS", "cj": "001040.KS", "오리온": "271560.KS",
                "jyp엔터": "035900.KS", "sm": "041510.KS", "yg엔터": "122870.KS",
                "두산": "000150.KS", "한화": "000880.KS", "gs": "078930.KS",
                "대우조선해양": "042660.KS", "현대미포조선": "010620.KS", "삼성엔지니어링": "028050.KS",
                "에스원": "012750.KS", "한국가스공사": "036460.KS", "알테오젠": "196170.KS",
                "셀트리온제약": "068760.KS", "압타바이오": "206640.KS", "종근당": "185750.KS",
                "일동제약": "249420.KS", "한국콜마": "161890.KS", "코웨이": "021240.KS",
                "지누스": "013890.KS", "lgcns": "034220.KS", "sk네트웍스": "001740.KS",
                "sk가스": "018670.KS", "sk케미칼": "285130.KS", "sk실트론": "101490.KS",
                "현대제철": "004020.KS", "동국제강": "001230.KS", "세아베스틸": "001430.KS"
            }
            
            # 코드 매핑 추가
            for name, ticker in major_stocks.items():
                all_stocks[name.lower()] = ticker
                code = ticker.split('.')[0]
                all_stocks[code] = ticker
        
        return all_stocks
    
    except Exception as e:
        st.error(f"종목 리스트 로드 오류: {str(e)}")
        return {}

def search_stock(user_input, krx_data):
    """종목 검색 (이름 또는 코드)"""
    user_input = user_input.strip().lower()
    
    # 1. 정확한 매치 (종목명)
    if user_input in krx_data:
        return krx_data[user_input], True
    
    # 2. 6자리 코드 매치
    if re.match(r'^\d{6}$', user_input):
        if user_input in krx_data:
            return krx_data[user_input], True
        # 자동으로 .KS, .KQ 시도
        for suffix in ['.KS', '.KQ']:
            ticker = user_input + suffix
            if validate_ticker_quick(ticker):
                return ticker, True
        return None, False
    
    # 3. 부분 매치 (종목명)
    matches = []
    for key, ticker in krx_data.items():
        if user_input in key and len(key) <= 20:  # 종목명만 (코드 제외)
            matches.append((key, ticker))
    
    if len(matches) == 1:
        return matches[0][1], True
    elif len(matches) > 1:
        # 여러 매치 발견 - 사용자에게 선택하도록
        return matches[:5], False  # 최대 5개
    
    return None, False

def validate_ticker_quick(ticker):
    """티커 빠른 검증 (데이터 최소 로드)"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        return not hist.empty
    except:
        return False

def load_stock_data(ticker, max_retries=3):
    """주식 데이터 로드 (재시도 포함)"""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            
            # 다양한 기간으로 시도
            for period in ["6mo", "3mo", "2mo", "1mo"]:
                hist = stock.history(period=period)
                if not hist.empty and len(hist) >= 20:
                    # 종목명
                    try:
                        info = stock.info
                        name = info.get('longName', info.get('shortName', ticker))
                    except:
                        name = ticker
                    
                    current_price = hist['Close'].iloc[-1]
                    return True, name, current_price, hist
            
            # 데이터가 너무 적음
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            
            return False, None, None, None
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False, None, None, None
    
    return False, None, None, None

# ========== UI 헤더 ==========
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🇰🇷 한국 주식 전체 종목 검색 (코스피 + 코스닥)")
st.markdown("**✅ 실시간 거래 종목 | ✅ 종목명/코드 검색 | ✅ 4대 모듈 분석**")
st.markdown("---")

# KRX 데이터 로드
if st.session_state.krx_data is None:
    with st.spinner("📡 한국거래소 종목 리스트 로딩 중..."):
        st.session_state.krx_data = load_krx_stock_list()
        if st.session_state.krx_data:
            st.success(f"✅ {len(st.session_state.krx_data)}개 종목 데이터베이스 로드 완료")
        else:
            st.warning("⚠️ 종목 리스트 로드 실패 (주요 종목만 지원)")

krx_data = st.session_state.krx_data

# ========== 종목 검색 ==========
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input(
        "🔍 종목 검색", 
        placeholder="종목명(예: 삼성전자, 카카오) 또는 코드(예: 005930, 035720)",
        help="한국 주식 전체 종목 검색 가능"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🔎 검색하기", type="primary", use_container_width=True)

if not ticker_input:
    st.info("👆 검색창에 종목명 또는 6자리 코드를 입력하세요")
    
    with st.expander("💡 검색 가능한 모든 종목 예시"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **대형주**
            - 삼성전자 (005930)
            - SK하이닉스 (000660)
            - LG전자 (066570)
            - 현대차 (005380)
            - 기아 (000270)
            """)
        with col2:
            st.markdown("""
            **IT/게임**
            - 카카오 (035720)
            - 네이버 (035420)
            - 크래프톤 (259960)
            - 엔씨소프트 (036570)
            - 넷마블 (251270)
            """)
        with col3:
            st.markdown("""
            **바이오/배터리**
            - 셀트리온 (068270)
            - 삼성바이오 (207940)
            - 에코프로 (086520)
            - LG화학 (051910)
            - 엘앤에프 (066970)
            """)
    
    st.markdown(f"**📊 검색 가능 종목 수: 약 {len(krx_data) if krx_data else '2,000'}개+**")
    st.stop()

if not search_btn:
    st.stop()

# ========== 검색 실행 ==========
with st.spinner("🔍 종목 검색 중..."):
    search_result, exact_match = search_stock(ticker_input, krx_data)
    
    if exact_match and isinstance(search_result, str):
        # 정확한 매치 발견
        final_ticker = search_result
        st.success(f"✅ 종목 발견: {final_ticker}")
    
    elif isinstance(search_result, list):
        # 여러 매치 발견
        st.warning(f"⚠️ '{ticker_input}'와 유사한 종목이 {len(search_result)}개 발견되었습니다.")
        st.markdown("### 📋 검색 결과 - 선택하세요")
        
        for idx, (name, ticker) in enumerate(search_result):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{idx+1}. {name.upper()}** ({ticker})")
            with col2:
                if st.button("선택", key=f"select_{idx}", use_container_width=True):
                    st.session_state.current_ticker = ticker
                    st.session_state.force_analyze = True
                    st.rerun()
        
        st.markdown("---")
        if st.button("🔙 다시 검색"):
            reset_session()
            st.rerun()
        
        st.stop()
    
    else:
        # 검색 실패
        st.error(f"❌ '{ticker_input}' 종목을 찾을 수 없습니다.")
        st.markdown("**검색 팁:**")
        st.markdown("- 종목명을 정확히 입력하세요 (예: 삼성전자, 카카오)")
        st.markdown("- 6자리 코드로 입력하세요 (예: 005930, 035720)")
        st.markdown("- 띄어쓰기 없이 입력하세요")
        
        # 유사 종목 제안 (간단 검색)
        suggestions = []
        input_lower = ticker_input.lower()
        for key in list(krx_data.keys())[:1000]:  # 처음 1000개만
            if len(key) > 6 and (input_lower in key or key in input_lower):
                suggestions.append(key)
                if len(suggestions) >= 5:
                    break
        
        if suggestions:
            st.markdown("**💡 혹시 이 종목을 찾으셨나요?**")
            for sugg in suggestions:
                st.markdown(f"- {sugg}")
        
        st.stop()

# ========== 데이터 로드 ==========
st.markdown("---")
with st.spinner("📊 주식 데이터 로딩 중 (최대 3회 재시도)..."):
    is_valid, company_name, current_price, hist = load_stock_data(final_ticker, max_retries=3)

if not is_valid or hist is None or hist.empty:
    st.error(f"❌ '{final_ticker}' 데이터를 로드할 수 없습니다.")
    st.markdown("**가능한 원인:**")
    st.markdown("- 상장폐지된 종목")
    st.markdown("- 거래 정지 중인 종목")
    st.markdown("- 일시적인 네트워크 오류")
    
    if st.button("🔄 재시도"):
        st.rerun()
    
    st.stop()

# ========== 종목 헤더 ==========
st.header(f"🏢 {company_name}")
st.subheader(f"📊 종목코드: **{final_ticker}**")
st.subheader(f"💰 현재가: **{current_price:,.0f}원**")

if len(hist) >= 2:
    prev = hist['Close'].iloc[-2]
    change = current_price - prev
    pct = (change / prev) * 100
    
    if change > 0:
        st.markdown(f"📈 **전일 대비**: +{change:,.0f}원 (**+{pct:.2f}%**)")
    elif change < 0:
        st.markdown(f"📉 **전일 대비**: {change:,.0f}원 (**{pct:.2f}%**)")
    else:
        st.markdown(f"📊 **전일 대비**: 보합")

st.caption(f"📅 데이터: {hist.index[0].strftime('%Y-%m-%d')} ~ {hist.index[-1].strftime('%Y-%m-%d')} ({len(hist)}일)")

if st.button("🔄 다른 종목 검색", use_container_width=True):
    reset_session()
    st.rerun()

st.markdown("---")

# ========== 모듈 1: 추세 & 패턴 ==========
st.subheader("📈 모듈 1: 추세 & 패턴 인식")

def calculate_ma(data, period):
    return data['Close'].rolling(window=period).mean()

hist['MA5'] = calculate_ma(hist, 5)
hist['MA20'] = calculate_ma(hist, 20)
hist['MA60'] = calculate_ma(hist, 60)
hist['MA120'] = calculate_ma(hist, 120)

latest = hist.iloc[-1]
ma5 = latest['MA5']
ma20 = latest['MA20']
ma60 = latest['MA60']
ma120 = latest['MA120']

def detect_candle_pattern(hist):
    if len(hist) < 3:
        return "데이터 부족", 50
    
    last = hist.iloc[-1]
    prev = hist.iloc[-2]
    
    last_body = abs(last['Close'] - last['Open'])
    prev_body = abs(prev['Close'] - prev['Open'])
    
    if (prev['Close'] < prev['Open'] and 
        last['Close'] > last['Open'] and 
        last_body > prev_body * 1.5):
        return "Bullish Engulfing 🟢", 80
    
    lower_shadow = min(last['Open'], last['Close']) - last['Low']
    upper_shadow = last['High'] - max(last['Open'], last['Close'])
    if last_body > 0 and lower_shadow > last_body * 2 and upper_shadow < last_body * 0.5:
        return "Hammer 🟢", 75
    
    if (prev['Close'] > prev['Open'] and 
        last['Close'] < last['Open'] and 
        last_body > prev_body * 1.5):
        return "Bearish Engulfing 🔴", 20
    
    return "패턴 없음", 50

candle_pattern, candle_score = detect_candle_pattern(hist)

if pd.notna(ma5) and pd.notna(ma20) and pd.notna(ma60) and pd.notna(ma120):
    if ma5 > ma20 > ma60 > ma120:
        ma_alignment = "정배열 🟢"
        ma_score = 85
    elif ma5 < ma20 < ma60 < ma120:
        ma_alignment = "역배열 🔴"
        ma_score = 20
    else:
        ma_alignment = "혼조 🟡"
        ma_score = 50
else:
    ma_alignment = "데이터 부족"
    ma_score = 50

if len(hist) >= 2 and pd.notna(ma20) and pd.notna(ma60):
    prev_ma20 = hist['MA20'].iloc[-2]
    prev_ma60 = hist['MA60'].iloc[-2]
    
    if pd.notna(prev_ma20) and pd.notna(prev_ma60):
        if ma20 > ma60 and prev_ma20 <= prev_ma60:
            cross = "골든크로스 🟢"
            cross_score = 90
        elif ma20 < ma60 and prev_ma20 >= prev_ma60:
            cross = "데드크로스 🔴"
            cross_score = 10
        else:
            cross = "신호 없음"
            cross_score = 50
    else:
        cross = "데이터 부족"
        cross_score = 50
else:
    cross = "데이터 부족"
    cross_score = 50

module1_score = int((candle_score * 0.3 + ma_score * 0.4 + cross_score * 0.3))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("캔들 패턴", candle_pattern, f"{candle_score}점")
with col2:
    st.metric("이동평균", ma_alignment, f"{ma_score}점")
with col3:
    st.metric("크로스", cross, f"{cross_score}점")

st.success(f"**📊 모듈1 종합: {module1_score}점**")
st.markdown("---")

# ========== 모듈 2: 거래량 ==========
st.subheader("📊 모듈 2: 거래량 검증")

hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()
current_volume = hist['Volume'].iloc[-1]
avg_volume = hist['Volume_MA20'].iloc[-1]

if pd.notna(avg_volume) and avg_volume > 0:
    volume_ratio = current_volume / avg_volume
    
    if volume_ratio >= 2.0:
        breakout = "높음 🟢"
        volume_score = 85
    elif volume_ratio >= 1.5:
        breakout = "중간 🟡"
        volume_score = 60
    else:
        breakout = "낮음 🔴"
        volume_score = 30
else:
    volume_ratio = 0
    breakout = "데이터 부족"
    volume_score = 50

if len(hist) >= 2:
    price_chg = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
    vol_chg = (hist['Volume'].iloc[-1] - hist['Volume'].iloc[-2]) / hist['Volume'].iloc[-2]
    
    if price_chg < 0 and vol_chg < 0:
        adj_health = "건전 🟢"
        adj_score = 10
    elif price_chg < 0 and vol_chg > 0.5:
        adj_health = "위험 🔴"
        adj_score = -40
    else:
        adj_health = "보통"
        adj_score = 0
else:
    adj_health = "데이터 부족"
    adj_score = 0

avg_trade_val = (hist['Close'] * hist['Volume']).tail(20).mean()
if avg_trade_val < 10_000_000_000:
    liquidity = "낮음 🔴"
    liq_score = -20
else:
    liquidity = "충분 🟢"
    liq_score = 0

module2_score = max(0, min(100, volume_score + adj_score + liq_score))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("거래량", f"{volume_ratio:.2f}배", breakout)
with col2:
    st.metric("조정", adj_health, f"{adj_score:+d}점")
with col3:
    st.metric("유동성", liquidity, f"{avg_trade_val/100_000_000:.0f}억")

st.success(f"**📊 모듈2 종합: {module2_score}점**")
st.markdown("---")

# ========== 모듈 3: 매수 신호 ==========
st.subheader("🎯 모듈 3: 매수 신호")

cond1 = current_price > ma120 if pd.notna(ma120) else False
high_20d = hist['Close'].tail(20).max()
cond2 = current_price >= high_20d

if len(hist) >= 2:
    pct_chg = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
    cond3 = 5 <= pct_chg <= 15
else:
    pct_chg = 0
    cond3 = False

cond4 = volume_ratio >= 2.0

satisfied = sum([cond1, cond2, cond3, cond4])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("120일선↑", "✅" if cond1 else "❌")
with col2:
    st.metric("20일고", "✅" if cond2 else "❌")
with col3:
    st.metric("상승", "✅" if cond3 else "❌")
with col4:
    st.metric("거래량", "✅" if cond4 else "❌")

if satisfied == 4:
    module3_score = 95
    st.success("🎉 **강력 매수**")
elif satisfied == 3:
    module3_score = 70
    st.warning("⚠️ **신중 매수**")
else:
    module3_score = 40
    st.error("❌ **부적합**")

st.success(f"**📊 모듈3 종합: {module3_score}점**")
st.markdown("---")

# ========== 모듈 4: 리스크 ==========
st.subheader("🛡️ 모듈 4: 리스크 관리")

sl_methods = {
    'open': latest['Open'],
    'low': latest['Low'],
    '3pct': current_price * 0.97,
    '5pct': current_price * 0.95,
    'ma20': ma20 if pd.notna(ma20) else current_price * 0.97
}

final_sl = max(sl_methods.values())
risk = ((final_sl - current_price) / current_price) * 100
target = current_price + abs(current_price - final_sl) * 2
reward = ((target - current_price) / current_price) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("진입", f"{current_price:,.0f}원")
with col2:
    st.metric("손절", f"{final_sl:,.0f}원", f"{risk:.2f}%")
with col3:
    st.metric("목표", f"{target:,.0f}원", f"+{reward:.2f}%")

st.success("**📊 모듈4: 완료**")
st.markdown("---")

# ========== 최종 평가 ==========
st.header("🏆 최종 평가")

final_score = int(module1_score * 0.3 + module2_score * 0.3 + module3_score * 0.4)

if final_score >= 75:
    rec = "🟢 **강력 매수**"
    detail = "적극 매수"
elif final_score >= 55:
    rec = "🟡 **신중 매수**"
    detail = "리스크 관리"
else:
    rec = "🔴 **부적합**"
    detail = "관망"

st.markdown(f"### {rec}")
st.markdown(f"**{final_score}점 / 100점**")
st.markdown(f"_{detail}_")
st.progress(final_score / 100)

# ========== 차트 ==========
st.subheader("📊 차트")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=hist.index,
    open=hist['Open'],
    high=hist['High'],
    low=hist['Low'],
    close=hist['Close'],
    name='캔들'
))

if pd.notna(ma5).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA5'], mode='lines', name='MA5', line=dict(color='orange', width=1)))
if pd.notna(ma20).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1)))
if pd.notna(ma60).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], mode='lines', name='MA60', line=dict(color='green', width=1)))
if pd.notna(ma120).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA120'], mode='lines', name='MA120', line=dict(color='red', width=1)))

fig.add_hline(y=target, line_dash="dot", line_color="green", annotation_text=f"목표: {target:,.0f}")
fig.add_hline(y=final_sl, line_dash="dot", line_color="red", annotation_text=f"손절: {final_sl:,.0f}")

fig.update_layout(
    title=f"{company_name}",
    xaxis_title="날짜",
    yaxis_title="가격",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("🔔 본 분석은 참고용입니다.")
