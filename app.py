import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
import time
import re

st.set_page_config(page_title="전문가급 주식 분석 시스템", page_icon="📊", layout="wide")

# ========== 세션 스테이트 초기화 ==========
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'force_analyze' not in st.session_state:
    st.session_state.force_analyze = False
if 'display_name' not in st.session_state:
    st.session_state.display_name = None

def reset_session():
    """세션 초기화"""
    st.cache_data.clear()
    st.session_state.force_analyze = False

def normalize_ticker_input(user_input):
    """사용자 입력을 정규화하여 티커 코드로 변환"""
    user_input = user_input.strip()
    
    # 1. 6자리 숫자 (한국 주식 코드)
    if re.match(r'^\d{6}$', user_input):
        return user_input + ".KS", "KS"
    
    # 2. 이미 .KS 또는 .KQ 접미사가 있는 경우
    if re.match(r'^\d{6}\.(KS|KQ)$', user_input.upper()):
        return user_input.upper(), user_input.upper().split('.')[-1]
    
    # 3. 영문 티커 (미국 주식)
    if re.match(r'^[A-Z]{1,5}$', user_input.upper()):
        return user_input.upper(), "US"
    
    # 4. 한글 입력 (한국 주식명)
    if re.search(r'[가-힣]', user_input):
        return user_input, "KR_NAME"
    
    # 5. 기타 (그대로 시도)
    return user_input, "UNKNOWN"

def search_korean_stock_by_name(stock_name):
    """한국 주식명으로 코드 검색 (주요 종목 사전 + 동적 검색)"""
    # 주요 종목 빠른 매핑 (자주 검색되는 상위 100개)
    MAJOR_STOCKS = {
        "삼성전자": "005930.KS", "sk하이닉스": "000660.KS", "카카오": "035720.KS",
        "네이버": "035420.KS", "lg화학": "051910.KS", "현대차": "005380.KS",
        "셀트리온": "068270.KS", "삼성바이오로직스": "207940.KS", "포스코홀딩스": "005490.KS",
        "kb금융": "105560.KS", "신한지주": "055550.KS", "삼성물산": "028260.KS",
        "기아": "000270.KS", "현대모비스": "012330.KS", "엔씨소프트": "036570.KS",
        "크래프톤": "259960.KS", "카카오뱅크": "323410.KS", "삼성sdi": "006400.KS",
        "lg전자": "066570.KS", "하이브": "352820.KS", "에코프로": "086520.KS",
        "에코프로비엠": "247540.KS", "엘앤에프": "066970.KS", "포스코퓨처엠": "003670.KS",
        "삼성전기": "009150.KS", "sk이노베이션": "096770.KS", "lg에너지솔루션": "373220.KS",
        "하나금융지주": "086790.KS", "sk텔레콤": "017670.KS", "kt": "030200.KS",
        "카카오게임즈": "293490.KS", "넷마블": "251270.KS", "펄어비스": "263750.KS",
        "셀트리온헬스케어": "091990.KS", "유한양행": "000100.KS", "한미약품": "128940.KS",
        "대한항공": "003490.KS", "아모레퍼시픽": "090430.KS", "lg생활건강": "051900.KS",
        "현대건설": "000720.KS", "삼성중공업": "010140.KS", "한국조선해양": "009540.KS",
        "신세계": "004170.KS", "롯데쇼핑": "023530.KS", "cj제일제당": "097950.KS",
        "두산에너빌리티": "034020.KS", "한화에어로스페이스": "012450.KS", "한국전력": "015760.KS",
        "sk스퀘어": "402340.KS", "lg디스플레이": "034220.KS", "현대글로비스": "086280.KS",
        "키움증권": "039490.KS", "미래에셋증권": "006800.KS", "삼성화재": "000810.KS",
        "우리금융지주": "316140.KS", "한국금융지주": "071050.KS", "jyp엔터": "035900.KS",
        "sm": "041510.KS", "yg엔터": "122870.KS", "씨젠": "096530.KS",
        "솔브레인": "357780.KS", "티씨케이": "064760.KS", "원익ips": "240810.KS",
        "대웅제약": "069620.KS", "녹십자": "006280.KS", "종근당": "185750.KS",
        "gs건설": "006360.KS", "대림산업": "000210.KS", "롯데케미칼": "011170.KS",
        "한화솔루션": "009830.KS", "s-oil": "010950.KS", "쿠팡": "CPNG",
        "테슬라": "TSLA", "애플": "AAPL", "마이크로소프트": "MSFT", "엔비디아": "NVDA",
        "아마존": "AMZN", "구글": "GOOGL", "메타": "META", "넷플릭스": "NFLX"
    }
    
    stock_name_lower = stock_name.lower().strip()
    
    # 빠른 검색
    if stock_name_lower in MAJOR_STOCKS:
        return MAJOR_STOCKS[stock_name_lower], True
    
    # 부분 매칭 시도
    for name, code in MAJOR_STOCKS.items():
        if stock_name_lower in name or name in stock_name_lower:
            return code, True
    
    # 찾지 못함
    return None, False

def validate_ticker_with_yfinance(ticker_code, max_retries=2):
    """yfinance로 티커 유효성 검증 및 데이터 로드"""
    for attempt in range(max_retries):
        try:
            ticker_obj = yf.Ticker(ticker_code)
            
            # 다양한 기간으로 시도
            for period in ["6mo", "3mo", "1mo", "1wk"]:
                hist = ticker_obj.history(period=period)
                if not hist.empty and len(hist) >= 5:
                    # 유효한 데이터 발견
                    current_price = hist['Close'].iloc[-1]
                    
                    # 종목명 추출
                    try:
                        info = ticker_obj.info
                        name = info.get('longName', info.get('shortName', ticker_code))
                    except:
                        name = ticker_code
                    
                    return True, name, current_price, hist
            
            # 모든 기간 시도했지만 데이터 없음
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

def try_multiple_ticker_formats(base_input):
    """여러 티커 형식으로 시도"""
    candidates = []
    
    # 6자리 숫자인 경우
    if re.match(r'^\d{6}$', base_input):
        candidates = [
            base_input + ".KS",  # 코스피
            base_input + ".KQ",  # 코스닥
            base_input           # 접미사 없음
        ]
    # 영문 티커
    elif re.match(r'^[A-Z]+$', base_input.upper()):
        candidates = [base_input.upper()]
    else:
        candidates = [base_input]
    
    return candidates

# ========== UI 헤더 ==========
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🎯 모든 주식 종목 검색 가능 (한국 + 미국)")
st.markdown("**✅ 실시간 검색 | ✅ 코스피/코스닥 자동 인식 | ✅ 미국 주식 지원**")
st.markdown("---")

# ========== 종목 검색 섹션 ==========
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input(
        "🔍 종목 입력", 
        placeholder="예: 삼성전자, 005930, TSLA, AAPL",
        help="종목명, 6자리 코드, 또는 미국 티커를 입력하세요"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🔎 검색 & 분석", type="primary", use_container_width=True)

if not ticker_input:
    st.info("👆 종목을 입력하고 [검색 & 분석] 버튼을 클릭하세요")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 💡 한국 주식 검색 방법
        - **종목명**: 삼성전자, 카카오, 네이버
        - **6자리 코드**: 005930, 035720, 035420
        - 코스피/코스닥 자동 판별
        """)
    with col2:
        st.markdown("""
        ### 💡 미국 주식 검색 방법
        - **티커**: TSLA, AAPL, MSFT, NVDA
        - **영문 대문자** 입력
        - 나스닥/NYSE 자동 지원
        """)
    st.stop()

if not search_btn:
    st.stop()

# ========== 종목 검색 프로세스 ==========
with st.spinner("🔍 종목 검색 중..."):
    normalized_ticker, input_type = normalize_ticker_input(ticker_input)
    
    final_ticker = None
    search_success = False
    
    # Case 1: 한글 종목명
    if input_type == "KR_NAME":
        st.info(f"📝 한국 주식명으로 검색 중: '{ticker_input}'")
        found_code, found = search_korean_stock_by_name(ticker_input)
        
        if found:
            final_ticker = found_code
            st.success(f"✅ 종목 코드 발견: {final_ticker}")
        else:
            st.error(f"❌ '{ticker_input}' 종목을 찾을 수 없습니다.")
            st.markdown("**해결 방법:**")
            st.markdown("- 종목명을 정확히 입력하세요 (예: 삼성전자, 카카오)")
            st.markdown("- 또는 6자리 코드로 입력하세요 (예: 005930)")
            st.stop()
    
    # Case 2: 코드 또는 미국 티커
    else:
        candidates = try_multiple_ticker_formats(normalized_ticker)
        st.info(f"🔎 티커 검증 중: {candidates}")
        
        for candidate in candidates:
            with st.spinner(f"⏳ {candidate} 유효성 확인 중..."):
                is_valid, name, price, hist = validate_ticker_with_yfinance(candidate, max_retries=2)
                
                if is_valid:
                    final_ticker = candidate
                    st.session_state.current_ticker = final_ticker
                    st.session_state.display_name = name
                    search_success = True
                    st.success(f"✅ 종목 발견: {name} ({final_ticker})")
                    break
        
        if not search_success:
            st.error(f"❌ '{ticker_input}' 종목 데이터를 찾을 수 없습니다.")
            st.markdown("**시도한 티커:**")
            for c in candidates:
                st.markdown(f"- {c}")
            st.markdown("---")
            st.markdown("**해결 방법:**")
            st.markdown("1. 종목 코드를 다시 확인하세요")
            st.markdown("2. 한국 주식: 6자리 코드 (예: 005930)")
            st.markdown("3. 미국 주식: 영문 티커 (예: TSLA, AAPL)")
            st.markdown("4. 종목명으로 검색 (예: 삼성전자, 카카오)")
            st.stop()

# ========== 데이터 로드 (재검증) ==========
st.markdown("---")
with st.spinner("📊 상세 데이터 로딩 중..."):
    is_valid, company_name, current_price, hist = validate_ticker_with_yfinance(final_ticker, max_retries=3)

if not is_valid or hist is None or hist.empty:
    st.error(f"❌ 데이터 로드 실패: {final_ticker}")
    st.stop()

display_name = st.session_state.display_name if st.session_state.display_name else company_name

# ========== 종목 헤더 ==========
st.header(f"🏢 {display_name}")
st.subheader(f"📊 티커: **{final_ticker}**")
st.subheader(f"💰 현재가: **{current_price:,.2f}원**")

if len(hist) >= 2:
    prev = hist['Close'].iloc[-2]
    change = current_price - prev
    pct = (change / prev) * 100
    
    if change > 0:
        st.markdown(f"📈 **전일 대비**: +{change:,.2f}원 (**+{pct:.2f}%**)")
    elif change < 0:
        st.markdown(f"📉 **전일 대비**: {change:,.2f}원 (**{pct:.2f}%**)")
    else:
        st.markdown(f"📊 **전일 대비**: 보합")

# 데이터 기간 정보
st.caption(f"📅 데이터 기간: {hist.index[0].strftime('%Y-%m-%d')} ~ {hist.index[-1].strftime('%Y-%m-%d')} ({len(hist)}일)")

# 새로운 검색 버튼
if st.button("🔄 다른 종목 검색", use_container_width=True):
    reset_session()
    st.rerun()

st.markdown("---")

# ========== 모듈 1: 추세 & 패턴 인식 ==========
st.subheader("📈 모듈 1: 추세 & 패턴 인식")

def calculate_ma(data, period):
    return data['Close'].rolling(window=period).mean()

# 이동평균선 계산
hist['MA5'] = calculate_ma(hist, 5)
hist['MA20'] = calculate_ma(hist, 20)
hist['MA60'] = calculate_ma(hist, 60)
hist['MA120'] = calculate_ma(hist, 120)

latest = hist.iloc[-1]
ma5 = latest['MA5']
ma20 = latest['MA20']
ma60 = latest['MA60']
ma120 = latest['MA120']

# 캔들 패턴
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
    if lower_shadow > last_body * 2 and upper_shadow < last_body * 0.5:
        return "Hammer 🟢", 75
    
    if (prev['Close'] > prev['Open'] and 
        last['Close'] < last['Open'] and 
        last_body > prev_body * 1.5):
        return "Bearish Engulfing 🔴", 20
    
    return "패턴 없음", 50

candle_pattern, candle_score = detect_candle_pattern(hist)

# 이동평균 정배열
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

# 골든/데드크로스
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

# ========== 모듈 2: 거래량 검증 ==========
st.subheader("📊 모듈 2: 거래량 & 공급 검증")

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
st.subheader("🎯 모듈 3: 매수 신호 (4-AND)")

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
    st.metric("상승폭", "✅" if cond3 else "❌")
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

# ========== 모듈 4: 리스크 관리 ==========
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
    st.metric("진입", f"{current_price:,.2f}원")
with col2:
    st.metric("손절", f"{final_sl:,.2f}원", f"{risk:.2f}%")
with col3:
    st.metric("목표", f"{target:,.2f}원", f"+{reward:.2f}%")

st.success("**📊 모듈4: 설정 완료**")
st.markdown("---")

# ========== 최종 평가 ==========
st.header("🏆 최종 종합 평가")

final_score = int(module1_score * 0.3 + module2_score * 0.3 + module3_score * 0.4)

if final_score >= 75:
    rec = "🟢 **강력 매수**"
    detail = "적극 매수 고려"
elif final_score >= 55:
    rec = "🟡 **신중 매수**"
    detail = "리스크 관리 필수"
else:
    rec = "🔴 **매수 부적합**"
    detail = "관망 권장"

st.markdown(f"### {rec}")
st.markdown(f"**종합: {final_score}점 / 100점**")
st.markdown(f"_{detail}_")
st.progress(final_score / 100)

# ========== 차트 ==========
st.subheader("📊 기술 분석 차트")

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
    title=f"{display_name} 분석",
    xaxis_title="날짜",
    yaxis_title="가격",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("🔔 본 분석은 참고용이며, 투자 결정은 본인의 책임입니다.")
