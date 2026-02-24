import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
import os
from difflib import get_close_matches
import time

st.set_page_config(page_title="전문가급 주식 분석 시스템", page_icon="📊", layout="wide")

# ========== 확장된 종목 데이터베이스 (150+ 종목) ==========
STOCK_MAP = {
    # IT/테크
    "삼성전자": "005930.KS", "sk하이닉스": "000660.KS", "카카오": "035720.KS",
    "네이버": "035420.KS", "엔씨소프트": "036570.KS", "크래프톤": "259960.KS",
    "컴투스": "078340.KS", "펄어비스": "263750.KS", "넷마블": "251270.KS",
    "카카오게임즈": "293490.KS", "카카오뱅크": "323410.KS", "카카오페이": "377300.KS",
    
    # 반도체/디스플레이
    "삼성sdi": "006400.KS", "lg디스플레이": "034220.KS", "sk스퀘어": "402340.KS",
    "솔브레인": "357780.KS", "원익ips": "240810.KS", "티씨케이": "064760.KS",
    "피에스케이": "319660.KS", "원익머트리얼즈": "104830.KS",
    
    # 바이오/제약
    "셀트리온": "068270.KS", "삼성바이오로직스": "207940.KS", "셀트리온헬스케어": "091990.KS",
    "셀트리온제약": "068760.KS", "유한양행": "000100.KS", "녹십자": "006280.KS",
    "대웅제약": "069620.KS", "한미약품": "128940.KS", "종근당": "185750.KS",
    "씨젠": "096530.KS", "압타바이오": "206640.KS",
    
    # 자동차/모빌리티
    "현대차": "005380.KS", "기아": "000270.KS", "현대모비스": "012330.KS",
    "현대위아": "011210.KS", "만도": "204320.KS", "현대글로비스": "086280.KS",
    
    # 화학/배터리
    "lg화학": "051910.KS", "포스코케미칼": "003670.KS", "에코프로": "086520.KS",
    "에코프로비엠": "247540.KS", "엘앤에프": "066970.KS", "천보": "278280.KS",
    "포스코홀딩스": "005490.KS", "롯데케미칼": "011170.KS",
    
    # 금융
    "kb금융": "105560.KS", "신한지주": "055550.KS", "하나금융지주": "086790.KS",
    "우리금융지주": "316140.KS", "삼성화재": "000810.KS", "현대해상": "001450.KS",
    "키움증권": "039490.KS", "미래에셋증권": "006800.KS", "한국금융지주": "071050.KS",
    
    # 건설/부동산
    "삼성물산": "028260.KS", "현대건설": "000720.KS", "대림산업": "000210.KS",
    "gs건설": "006360.KS", "대우건설": "047040.KS",
    
    # 유통/소비재
    "신세계": "004170.KS", "롯데쇼핑": "023530.KS", "이마트": "139480.KS",
    "lg생활건강": "051900.KS", "아모레퍼시픽": "090430.KS", "cj제일제당": "097950.KS",
    "cj": "001040.KS", "오리온": "271560.KS",
    
    # 통신/미디어
    "sk텔레콤": "017670.KS", "kt": "030200.KS", "lg유플러스": "032640.KS",
    "제일기획": "030000.KS",
    
    # 조선/중공업
    "한국조선해양": "009540.KS", "삼성중공업": "010140.KS", "대우조선해양": "042660.KS",
    "현대미포조선": "010620.KS",
    
    # 항공/물류
    "대한항공": "003490.KS", "한진칼": "180640.KS", "cj대한통운": "000120.KS",
    
    # 에너지
    "sk이노베이션": "096770.KS", "s-oil": "010950.KS", "gs칼텍스": "117580.KS",
    "한국전력": "015760.KS", "한국가스공사": "036460.KS",
    
    # 기타 주요 종목
    "한화": "000880.KS", "두산": "000150.KS", "롯데지주": "004990.KS",
    "하이브": "352820.KS", "jyp엔터": "035900.KS", "sm": "041510.KS",
    "yg엔터": "122870.KS", "에스엠": "041510.KS",
    
    # 미국 주식
    "테슬라": "TSLA", "애플": "AAPL", "마이크로소프트": "MSFT",
    "아마존": "AMZN", "구글": "GOOGL", "메타": "META",
    "엔비디아": "NVDA", "넷플릭스": "NFLX", "쿠팡": "CPNG"
}

# 역방향 매핑 (코드 -> 이름)
CODE_TO_NAME = {v: k for k, v in STOCK_MAP.items()}

# ========== 세션 스테이트 초기화 ==========
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'suggested_stocks' not in st.session_state:
    st.session_state.suggested_stocks = []
if 'search_mode' not in st.session_state:
    st.session_state.search_mode = 'input'  # 'input' or 'suggest'
if 'selected_suggestion' not in st.session_state:
    st.session_state.selected_suggestion = None
if 'force_analyze' not in st.session_state:
    st.session_state.force_analyze = False

def reset_session():
    """세션 초기화"""
    st.cache_data.clear()
    st.session_state.search_mode = 'input'
    st.session_state.suggested_stocks = []
    st.session_state.selected_suggestion = None
    st.session_state.force_analyze = False

def find_similar_stocks(user_input, n=3):
    """유사한 종목 이름 찾기"""
    user_input = user_input.strip().lower()
    stock_names = list(STOCK_MAP.keys())
    
    # 1. 정확한 매치
    if user_input in stock_names:
        return [user_input], True
    
    # 2. 부분 문자열 매치
    partial_matches = [name for name in stock_names if user_input in name or name in user_input]
    if partial_matches:
        return partial_matches[:n], False
    
    # 3. difflib 유사도 매치
    close_matches = get_close_matches(user_input, stock_names, n=n, cutoff=0.4)
    
    return close_matches, False

def load_data_with_retry(yf_ticker, max_retries=3):
    """재시도 메커니즘이 포함된 데이터 로드"""
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(yf_ticker)
            
            # 1. 역사 데이터 로드 (6개월)
            hist = ticker.history(period="6mo")
            
            if hist.empty:
                # 2. 다른 기간으로 재시도
                hist = ticker.history(period="3mo")
            
            if hist.empty:
                # 3. 1개월로 최종 시도
                hist = ticker.history(period="1mo")
            
            if hist.empty:
                if attempt < max_retries - 1:
                    time.sleep(1)  # 1초 대기 후 재시도
                    continue
                return None, None, None
            
            # 현재가
            current_price = hist['Close'].iloc[-1]
            
            # 종목 이름
            try:
                info = ticker.info
                name = info.get('longName', info.get('shortName', '알 수 없음'))
            except:
                name = "알 수 없음"
            
            return name, current_price, hist
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, None, None
    
    return None, None, None

# ========== UI 헤더 ==========
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🎯 승률 50% → 70% 향상 목표")
st.markdown("**✅ 수학적 검증 | ✅ 명확한 신호 | ✅ 실전 전략**")
st.markdown("---")

# ========== 종목 입력 섹션 ==========
if st.session_state.search_mode == 'input':
    # 일반 입력 모드
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input(
            "종목 입력 (예: 삼성전자, 005930, 카카오)", 
            key="ticker_input",
            placeholder="종목명 또는 코드를 입력하세요"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔍 검색", type="primary", use_container_width=True)

    if not ticker_input:
        st.info("👆 종목을 입력하고 검색 버튼을 클릭하세요")
        st.markdown("""
        ### 💡 사용 가능한 종목 예시
        - **한국 주식**: 삼성전자, 카카오, 네이버, 현대차, lg화학, 셀트리온 등
        - **미국 주식**: 테슬라, 애플, 엔비디아, 구글 등
        - **코드 직접 입력**: 005930, 035720 등
        """)
        st.stop()
    
    if search_btn:
        # 종목 검색
        user_input_lower = ticker_input.strip().lower()
        
        # 1. 정확한 매치 확인
        if user_input_lower in STOCK_MAP:
            ticker_code = STOCK_MAP[user_input_lower]
            st.session_state.current_ticker = ticker_code
            st.session_state.selected_suggestion = user_input_lower
            st.session_state.force_analyze = True
            st.rerun()
        
        # 2. 코드 입력 확인 (숫자만)
        elif ticker_input.strip().isdigit():
            ticker_code = ticker_input.strip() + ".KS"
            st.session_state.current_ticker = ticker_code
            st.session_state.force_analyze = True
            st.rerun()
        
        # 3. 미국 주식 티커 확인 (영문 대문자)
        elif ticker_input.strip().isupper() and ticker_input.strip().isalpha():
            ticker_code = ticker_input.strip()
            st.session_state.current_ticker = ticker_code
            st.session_state.force_analyze = True
            st.rerun()
        
        # 4. 유사 종목 검색
        else:
            similar_stocks, exact_match = find_similar_stocks(user_input_lower, n=3)
            
            if similar_stocks:
                st.session_state.suggested_stocks = similar_stocks
                st.session_state.search_mode = 'suggest'
                st.rerun()
            else:
                st.error(f"❌ '{ticker_input}' 종목을 찾을 수 없습니다.")
                st.markdown("**다시 시도해보세요:**")
                st.markdown("- 종목명을 정확히 입력")
                st.markdown("- 종목 코드로 입력 (예: 005930)")
                st.stop()

elif st.session_state.search_mode == 'suggest':
    # 유사 종목 추천 모드
    st.warning(f"⚠️ 입력한 종목을 찾을 수 없습니다. 유사한 종목을 추천합니다:")
    st.markdown("### 🔍 추천 종목 선택")
    
    # 추천 종목 표시
    for idx, stock_name in enumerate(st.session_state.suggested_stocks):
        stock_code = STOCK_MAP[stock_name]
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{idx+1}. 📊 {stock_name.upper()}** ({stock_code})")
        with col2:
            if st.button(f"선택", key=f"select_{idx}", use_container_width=True):
                st.session_state.current_ticker = stock_code
                st.session_state.selected_suggestion = stock_name
                st.session_state.search_mode = 'input'
                st.session_state.force_analyze = True
                st.rerun()
    
    st.markdown("---")
    if st.button("🔙 다시 검색하기"):
        reset_session()
        st.rerun()
    
    st.stop()

# ========== 분석 실행 ==========
if not st.session_state.force_analyze:
    st.stop()

# 강제 분석 모드
ticker_code = st.session_state.current_ticker
yf_ticker = ticker_code

# 이름 파싱
if st.session_state.selected_suggestion:
    parsed_name = st.session_state.selected_suggestion.upper()
else:
    parsed_name = CODE_TO_NAME.get(ticker_code, None)

# 데이터 로드
with st.spinner("📊 데이터 로딩 중... (최대 3번 재시도)"):
    company_name, current_price, hist = load_data_with_retry(yf_ticker, max_retries=3)

if hist is None or hist.empty:
    st.error(f"❌ 종목 '{ticker_code}' 데이터를 찾을 수 없습니다")
    st.markdown("**가능한 원인:**")
    st.markdown("- 잘못된 종목 코드")
    st.markdown("- 상장폐지된 종목")
    st.markdown("- 일시적인 데이터 서버 오류")
    st.markdown("---")
    if st.button("🔙 다시 검색하기"):
        st.session_state.force_analyze = False
        reset_session()
        st.rerun()
    st.stop()

display_name = parsed_name if parsed_name else company_name

# 분석 완료 후 force_analyze 플래그 해제
st.session_state.force_analyze = False

# ========== 종목 헤더 ==========
st.header(f"🏢 {display_name} ({ticker_code.replace('.KS', '')})")
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

# 새로운 검색 버튼
if st.button("🔄 다른 종목 검색"):
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

# 최신 이동평균 값
latest = hist.iloc[-1]
ma5 = latest['MA5']
ma20 = latest['MA20']
ma60 = latest['MA60']
ma120 = latest['MA120']

# 1-1. 캔들 패턴 인식
def detect_candle_pattern(hist):
    if len(hist) < 3:
        return "데이터 부족", 50
    
    last = hist.iloc[-1]
    prev = hist.iloc[-2]
    prev2 = hist.iloc[-3]
    
    last_body = abs(last['Close'] - last['Open'])
    prev_body = abs(prev['Close'] - prev['Open'])
    
    # Bullish Engulfing
    if (prev['Close'] < prev['Open'] and 
        last['Close'] > last['Open'] and 
        last_body > prev_body * 1.5):
        return "Bullish Engulfing 🟢", 80
    
    # Hammer
    lower_shadow = min(last['Open'], last['Close']) - last['Low']
    upper_shadow = last['High'] - max(last['Open'], last['Close'])
    if lower_shadow > last_body * 2 and upper_shadow < last_body * 0.5:
        return "Hammer 🟢", 75
    
    # Bearish Engulfing
    if (prev['Close'] > prev['Open'] and 
        last['Close'] < last['Open'] and 
        last_body > prev_body * 1.5):
        return "Bearish Engulfing 🔴", 20
    
    return "패턴 없음", 50

candle_pattern, candle_score = detect_candle_pattern(hist)

# 1-2. 이동평균 정배열
if pd.notna(ma5) and pd.notna(ma20) and pd.notna(ma60) and pd.notna(ma120):
    if ma5 > ma20 > ma60 > ma120:
        ma_alignment = "정배열 (강세) 🟢"
        ma_score = 85
    elif ma5 < ma20 < ma60 < ma120:
        ma_alignment = "역배열 (약세) 🔴"
        ma_score = 20
    else:
        ma_alignment = "혼조 (관망) 🟡"
        ma_score = 50
else:
    ma_alignment = "데이터 부족"
    ma_score = 50

# 1-3. 골든크로스/데드크로스
if len(hist) >= 2:
    prev_ma20 = hist['MA20'].iloc[-2]
    prev_ma60 = hist['MA60'].iloc[-2]
    
    if pd.notna(ma20) and pd.notna(ma60) and pd.notna(prev_ma20) and pd.notna(prev_ma60):
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

# 모듈1 종합 점수
module1_score = int((candle_score * 0.3 + ma_score * 0.4 + cross_score * 0.3))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("캔들 패턴", candle_pattern, f"{candle_score}점")
with col2:
    st.metric("이동평균 배열", ma_alignment, f"{ma_score}점")
with col3:
    st.metric("크로스 신호", cross, f"{cross_score}점")

st.success(f"**📊 모듈1 종합 점수: {module1_score}점**")
st.markdown("---")

# ========== 모듈 2: 거래량 & 공급 검증 ==========
st.subheader("📊 모듈 2: 거래량 & 공급 검증")

hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()

current_volume = hist['Volume'].iloc[-1]
avg_volume = hist['Volume_MA20'].iloc[-1]

if pd.notna(avg_volume) and avg_volume > 0:
    volume_ratio = current_volume / avg_volume
    
    if volume_ratio >= 2.0:
        breakout_confidence = "높음 🟢 (진성 돌파)"
        volume_score = 85
    elif volume_ratio >= 1.5:
        breakout_confidence = "중간 🟡 (주의 필요)"
        volume_score = 60
    else:
        breakout_confidence = "낮음 🔴 (거래량 부족)"
        volume_score = 30
else:
    volume_ratio = 0
    breakout_confidence = "데이터 부족"
    volume_score = 50

# 조정 건전성
if len(hist) >= 2:
    price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
    volume_change = (hist['Volume'].iloc[-1] - hist['Volume'].iloc[-2]) / hist['Volume'].iloc[-2]
    
    if price_change < 0 and volume_change < 0:
        adjustment_health = "건전 🟢"
        adjustment_score = 10
    elif price_change < 0 and volume_change > 0.5:
        adjustment_health = "위험 🔴"
        adjustment_score = -40
    else:
        adjustment_health = "보통"
        adjustment_score = 0
else:
    adjustment_health = "데이터 부족"
    adjustment_score = 0

# 유동성 필터
avg_trading_value = (hist['Close'] * hist['Volume']).tail(20).mean()
if avg_trading_value < 10_000_000_000:
    liquidity = "유동성 낮음 🔴"
    liquidity_score = -20
else:
    liquidity = "유동성 충분 🟢"
    liquidity_score = 0

module2_score = max(0, min(100, volume_score + adjustment_score + liquidity_score))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("거래량 배율", f"{volume_ratio:.2f}배", breakout_confidence)
with col2:
    st.metric("조정 건전성", adjustment_health, f"{adjustment_score:+d}점")
with col3:
    st.metric("유동성", liquidity, f"{avg_trading_value/100_000_000:.0f}억원")

st.success(f"**📊 모듈2 종합 점수: {module2_score}점**")
st.markdown("---")

# ========== 모듈 3: 매수 신호 ==========
st.subheader("🎯 모듈 3: 매수 신호 (4-AND 조건)")

cond1 = current_price > ma120 if pd.notna(ma120) else False
high_20d = hist['Close'].tail(20).max()
cond2 = current_price >= high_20d

if len(hist) >= 2:
    pct_change = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
    cond3 = 5 <= pct_change <= 15
else:
    pct_change = 0
    cond3 = False

cond4 = volume_ratio >= 2.0 if pd.notna(avg_volume) else False

satisfied_count = sum([cond1, cond2, cond3, cond4])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("120일선 상회", "✅" if cond1 else "❌")
with col2:
    st.metric("20일 신고가", "✅" if cond2 else "❌")
with col3:
    st.metric("최적 상승폭", "✅" if cond3 else "❌")
with col4:
    st.metric("거래량 급증", "✅" if cond4 else "❌")

if satisfied_count == 4:
    module3_score = 95
    st.success("🎉 **모든 매수 조건 충족!**")
elif satisfied_count == 3:
    module3_score = 70
    st.warning("⚠️ **3개 조건 충족**")
else:
    module3_score = 40
    st.error("❌ **매수 부적합**")

st.success(f"**📊 모듈3 종합 점수: {module3_score}점**")
st.markdown("---")

# ========== 모듈 4: 리스크 관리 ==========
st.subheader("🛡️ 모듈 4: 리스크 관리")

stop_loss_methods = {
    'entry_open': latest['Open'],
    'entry_low': latest['Low'],
    'pct_3': current_price * 0.97,
    'pct_5': current_price * 0.95,
    'ma20': ma20 if pd.notna(ma20) else current_price * 0.97
}

final_stop_loss = max(stop_loss_methods.values())
risk_pct = ((final_stop_loss - current_price) / current_price) * 100
target_price = current_price + abs(current_price - final_stop_loss) * 2
reward_pct = ((target_price - current_price) / current_price) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💼 진입가", f"{current_price:,.0f}원")
with col2:
    st.metric("🛑 손절가", f"{final_stop_loss:,.0f}원", f"{risk_pct:.2f}%")
with col3:
    st.metric("🎯 목표가", f"{target_price:,.0f}원", f"+{reward_pct:.2f}%")

st.success(f"**📊 모듈4: 리스크 관리 설정 완료**")
st.markdown("---")

# ========== 최종 평가 ==========
st.header("🏆 최종 종합 평가")

final_score = int(module1_score * 0.3 + module2_score * 0.3 + module3_score * 0.4)

if final_score >= 75:
    recommendation = "🟢 **강력 매수 추천**"
    recommendation_detail = "모든 지표가 긍정적입니다."
elif final_score >= 55:
    recommendation = "🟡 **신중 매수**"
    recommendation_detail = "리스크 관리를 철저히 하세요."
else:
    recommendation = "🔴 **매수 부적합**"
    recommendation_detail = "관망을 권장합니다."

st.markdown(f"### {recommendation}")
st.markdown(f"**종합 점수: {final_score}점 / 100점**")
st.markdown(f"_{recommendation_detail}_")

st.progress(final_score / 100)

# ========== 차트 ==========
st.subheader("📊 기술적 분석 차트")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=hist.index,
    open=hist['Open'],
    high=hist['High'],
    low=hist['Low'],
    close=hist['Close'],
    name='캔들'
))

fig.add_trace(go.Scatter(x=hist.index, y=hist['MA5'], mode='lines', name='MA5', line=dict(color='orange', width=1)))
fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1)))
fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], mode='lines', name='MA60', line=dict(color='green', width=1)))
fig.add_trace(go.Scatter(x=hist.index, y=hist['MA120'], mode='lines', name='MA120', line=dict(color='red', width=1)))

fig.add_hline(y=target_price, line_dash="dot", line_color="green", annotation_text=f"목표가: {target_price:,.0f}원")
fig.add_hline(y=final_stop_loss, line_dash="dot", line_color="red", annotation_text=f"손절가: {final_stop_loss:,.0f}원")

fig.update_layout(
    title=f"{display_name} 기술적 분석",
    xaxis_title="날짜",
    yaxis_title="가격 (원)",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("**🔔 본 분석은 투자 참고용이며, 최종 투자 결정은 본인의 책임입니다.**")
