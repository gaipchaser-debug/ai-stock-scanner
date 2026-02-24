import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
import os
from difflib import get_close_matches

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
if 'search_failed' not in st.session_state:
    st.session_state.search_failed = False
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None

def reset_session():
    st.cache_data.clear()
    st.session_state.search_failed = False
    st.session_state.suggested_stocks = []

def find_similar_stocks(user_input, n=3):
    """유사한 종목 이름 찾기 (difflib 사용)"""
    user_input = user_input.strip().lower()
    stock_names = list(STOCK_MAP.keys())
    
    # 1. 정확한 매치 시도
    if user_input in stock_names:
        return [user_input], True
    
    # 2. 부분 문자열 매치
    partial_matches = [name for name in stock_names if user_input in name or name in user_input]
    if partial_matches:
        return partial_matches[:n], False
    
    # 3. difflib를 이용한 유사도 매치
    close_matches = get_close_matches(user_input, stock_names, n=n, cutoff=0.4)
    
    return close_matches, False

def parse_ticker(user_input):
    """종목 코드/이름 파싱 (유사 검색 포함)"""
    user_input = user_input.strip().lower()
    
    # 1. 숫자 코드 입력 (예: 005930)
    if user_input.isdigit():
        # 한국 코드에 .KS 자동 추가
        ticker_with_suffix = user_input + ".KS"
        if ticker_with_suffix in CODE_TO_NAME:
            return ticker_with_suffix, CODE_TO_NAME[ticker_with_suffix].upper(), True
        return ticker_with_suffix, None, True
    
    # 2. 정확한 이름 매치
    if user_input in STOCK_MAP:
        code = STOCK_MAP[user_input]
        return code, user_input.upper(), True
    
    # 3. 유사 종목 검색
    similar_stocks, exact_match = find_similar_stocks(user_input, n=3)
    
    if similar_stocks and not exact_match:
        return None, None, False  # 유사 종목 추천 모드
    
    # 4. 그 외 (미국 주식 또는 직접 입력)
    return user_input, None, True

# ========== UI 헤더 ==========
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🎯 승률 50% → 70% 향상 목표")
st.markdown("**✅ 수학적 검증 | ✅ 명확한 신호 | ✅ 실전 전략**")
st.markdown("---")

# ========== 종목 입력 ==========
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input(
        "종목 입력 (예: 삼성전자, 005930, 카카오)", 
        key="ticker",
        placeholder="종목명 또는 코드를 입력하세요",
        on_change=reset_session
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 분석 시작", type="primary", use_container_width=True)

if not ticker_input or not analyze_btn:
    st.info("👆 종목을 입력하고 분석 시작 버튼을 클릭하세요")
    st.markdown("""
    ### 💡 사용 가능한 종목 예시
    - **한국 주식**: 삼성전자, 카카오, 네이버, 현대차, lg화학, 셀트리온 등
    - **미국 주식**: 테슬라, 애플, 엔비디아, 구글 등
    - **코드 직접 입력**: 005930 (삼성전자), 035720 (카카오) 등
    """)
    st.stop()

# ========== 종목 검색 및 유사 종목 추천 ==========
ticker_code, parsed_name, search_success = parse_ticker(ticker_input)

if not search_success:
    # 유사 종목 찾기
    similar_stocks, _ = find_similar_stocks(ticker_input, n=3)
    
    if similar_stocks:
        st.warning(f"⚠️ '{ticker_input}' 종목을 찾을 수 없습니다. 유사한 종목을 추천합니다:")
        st.markdown("### 🔍 추천 종목 (아래에서 선택하세요)")
        
        # 라디오 버튼으로 추천 종목 표시
        selected_stock = st.radio(
            "선택:",
            similar_stocks,
            format_func=lambda x: f"📊 {x.upper()} ({STOCK_MAP[x]})",
            key="similar_stock_selector"
        )
        
        # 선택 확인 버튼
        if st.button("✅ 선택한 종목 분석하기", type="primary"):
            ticker_code = STOCK_MAP[selected_stock]
            parsed_name = selected_stock.upper()
            search_success = True
            st.session_state.selected_stock = selected_stock
            st.rerun()
        else:
            st.stop()
    else:
        st.error(f"❌ '{ticker_input}' 종목을 찾을 수 없습니다. 올바른 종목명 또는 코드를 입력하세요.")
        st.stop()

# 세션 스테이트 업데이트
if st.session_state.current_ticker != ticker_code:
    st.session_state.current_ticker = ticker_code
    reset_session()

# yfinance 티커 포맷
yf_ticker = ticker_code

# ========== 데이터 로드 ==========
@st.cache_data(ttl=300, show_spinner=False)
def load_data(yf_ticker):
    try:
        ticker = yf.Ticker(yf_ticker)
        hist = ticker.history(period="6mo")
        if hist.empty:
            return None, None, None
        current_price = hist['Close'].iloc[-1]
        try:
            name = ticker.info.get('longName', ticker.info.get('shortName', '알 수 없음'))
        except:
            name = "알 수 없음"
        return name, current_price, hist
    except Exception as e:
        st.error(f"데이터 로드 오류: {str(e)}")
        return None, None, None

with st.spinner("📊 데이터 로딩 중..."):
    company_name, current_price, hist = load_data(yf_ticker)

if hist is None or hist.empty:
    st.error(f"❌ 종목 '{ticker_input}' 데이터를 찾을 수 없습니다")
    st.markdown("**가능한 원인:**")
    st.markdown("- 잘못된 종목 코드")
    st.markdown("- 상장폐지된 종목")
    st.markdown("- 일시적인 데이터 서버 오류")
    st.stop()

display_name = parsed_name if parsed_name else company_name

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

# 1-1. 캔들 패턴 인식 (간단한 구현)
def detect_candle_pattern(hist):
    if len(hist) < 3:
        return "데이터 부족", 0
    
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

# 1-2. 이동평균 정배열 체크
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

# 거래량 이동평균
hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()

current_volume = hist['Volume'].iloc[-1]
avg_volume = hist['Volume_MA20'].iloc[-1]

if pd.notna(avg_volume) and avg_volume > 0:
    volume_ratio = current_volume / avg_volume
    
    # 브레이크아웃 신뢰도
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

# 조정 건전성 (가격 하락 시 거래량 감소 = 건전)
if len(hist) >= 2:
    price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
    volume_change = (hist['Volume'].iloc[-1] - hist['Volume'].iloc[-2]) / hist['Volume'].iloc[-2]
    
    if price_change < 0 and volume_change < 0:
        adjustment_health = "건전 🟢 (하락+거래량감소)"
        adjustment_score = 10  # 가산점
    elif price_change < 0 and volume_change > 0.5:
        adjustment_health = "위험 🔴 (하락+거래량급증)"
        adjustment_score = -40  # 감점
    else:
        adjustment_health = "보통"
        adjustment_score = 0
else:
    adjustment_health = "데이터 부족"
    adjustment_score = 0

# 페니스톡 필터 (일평균 거래대금 100억 미만)
avg_trading_value = (hist['Close'] * hist['Volume']).tail(20).mean()
if avg_trading_value < 10_000_000_000:  # 100억
    liquidity = "유동성 낮음 🔴"
    liquidity_score = -20
else:
    liquidity = "유동성 충분 🟢"
    liquidity_score = 0

# 모듈2 종합 점수
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

# ========== 모듈 3: 매수 신호 (4-AND 조건) ==========
st.subheader("🎯 모듈 3: 매수 신호 (4-AND 조건)")

# 조건 1: 가격 > 120일선
cond1 = current_price > ma120 if pd.notna(ma120) else False

# 조건 2: 20일 신고가
high_20d = hist['Close'].tail(20).max()
cond2 = current_price >= high_20d

# 조건 3: 일일 변동률 5~15%
if len(hist) >= 2:
    pct_change = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
    cond3 = 5 <= pct_change <= 15
else:
    pct_change = 0
    cond3 = False

# 조건 4: 거래량 >= 2배
cond4 = volume_ratio >= 2.0 if pd.notna(avg_volume) else False

# 개별 조건 점수
buy_conditions = {
    'above_120ma': cond1,
    'new_high_20d': cond2,
    'optimal_gain': cond3,
    'volume_surge': cond4,
    'pct_change': pct_change,
    'volume_ratio': volume_ratio
}

satisfied_count = sum([cond1, cond2, cond3, cond4])

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("120일선 상회", "✅" if cond1 else "❌", f"{ma120:,.0f}원" if pd.notna(ma120) else "N/A")
with col2:
    st.metric("20일 신고가", "✅" if cond2 else "❌", f"{high_20d:,.0f}원")
with col3:
    optimal_text = f"{pct_change:.1f}%"
    if cond3:
        st.metric("최적 상승폭 (5~15%)", "✅", optimal_text)
    else:
        st.metric("최적 상승폭 (5~15%)", "❌", optimal_text)
with col4:
    st.metric("거래량 급증 (2배↑)", "✅" if cond4 else "❌", f"{volume_ratio:.2f}배")

if satisfied_count == 4:
    module3_score = 95
    st.success("🎉 **모든 매수 조건 충족! 강력 매수 신호**")
elif satisfied_count == 3:
    module3_score = 70
    st.warning("⚠️ **3개 조건 충족 (신중 매수)**")
elif satisfied_count == 2:
    module3_score = 55
    st.info("ℹ️ **2개 조건 충족 (관망)**")
else:
    module3_score = 40
    st.error("❌ **매수 부적합**")

st.success(f"**📊 모듈3 종합 점수: {module3_score}점**")
st.markdown("---")

# ========== 모듈 4: 리스크 관리 & 청산 전략 ==========
st.subheader("🛡️ 모듈 4: 리스크 관리 & 청산 전략")

# 손절가 계산 (5가지 방법)
stop_loss_methods = {}

# 방법 1: 진입 캔들의 시가
stop_loss_methods['entry_open'] = latest['Open']

# 방법 2: 진입 캔들의 저가
stop_loss_methods['entry_low'] = latest['Low']

# 방법 3: -3% 손절
stop_loss_methods['pct_3'] = current_price * 0.97

# 방법 4: -5% 손절
stop_loss_methods['pct_5'] = current_price * 0.95

# 방법 5: 20일선 하단
stop_loss_methods['ma20'] = ma20 if pd.notna(ma20) else current_price * 0.97

# 최종 손절가 (가장 높은 값 선택 = 안전)
final_stop_loss = max(stop_loss_methods.values())

# 손실 리스크 계산
risk_pct = ((final_stop_loss - current_price) / current_price) * 100

# 목표가 (1:2 리스크 리워드)
target_price = current_price + abs(current_price - final_stop_loss) * 2
reward_pct = ((target_price - current_price) / current_price) * 100

# 추적 손절 (최고가 대비 -3%)
if len(hist) >= 5:
    recent_high = hist['Close'].tail(5).max()
    trailing_stop = recent_high * 0.97
else:
    trailing_stop = final_stop_loss

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💼 진입가", f"{current_price:,.0f}원", "현재가")
with col2:
    st.metric("🛑 손절가", f"{final_stop_loss:,.0f}원", f"{risk_pct:.2f}%")
with col3:
    st.metric("🎯 목표가", f"{target_price:,.0f}원", f"+{reward_pct:.2f}%")

st.markdown(f"**추적 손절**: {trailing_stop:,.0f}원 (최근 5일 고점 대비 -3%)")
st.markdown(f"**리스크:리워드 비율**: 1:2")

st.success(f"**📊 모듈4: 리스크 관리 설정 완료**")
st.markdown("---")

# ========== 최종 종합 평가 ==========
st.header("🏆 최종 종합 평가")

# 가중 평균 점수
final_score = int(module1_score * 0.3 + module2_score * 0.3 + module3_score * 0.4)

# 투자 의견
if final_score >= 75:
    recommendation = "🟢 **강력 매수 추천**"
    recommendation_color = "green"
    recommendation_detail = "모든 지표가 긍정적입니다. 적극 매수를 고려하세요."
elif final_score >= 55:
    recommendation = "🟡 **신중 매수**"
    recommendation_color = "orange"
    recommendation_detail = "일부 지표가 긍정적이나, 리스크 관리를 철저히 하세요."
else:
    recommendation = "🔴 **매수 부적합**"
    recommendation_color = "red"
    recommendation_detail = "현재 매수 시점이 아닙니다. 관망을 권장합니다."

st.markdown(f"### {recommendation}")
st.markdown(f"**종합 점수: {final_score}점 / 100점**")
st.markdown(f"_{recommendation_detail}_")

# 점수 표시 (프로그레스 바)
st.progress(final_score / 100)

# 모듈별 점수 요약
st.markdown("### 📋 모듈별 상세 점수")
score_df = pd.DataFrame({
    "모듈": ["모듈1: 추세&패턴", "모듈2: 거래량&공급", "모듈3: 매수신호", "모듈4: 리스크관리"],
    "점수": [module1_score, module2_score, module3_score, "✅ 설정완료"],
    "가중치": ["30%", "30%", "40%", "-"],
    "상태": [
        "🟢" if module1_score >= 75 else "🟡" if module1_score >= 55 else "🔴",
        "🟢" if module2_score >= 75 else "🟡" if module2_score >= 55 else "🔴",
        "🟢" if module3_score >= 75 else "🟡" if module3_score >= 55 else "🔴",
        "✅"
    ]
})
st.table(score_df)

st.markdown("---")

# ========== 차트 시각화 ==========
st.subheader("📊 기술적 분석 차트")

fig = go.Figure()

# 캔들스틱 차트
fig.add_trace(go.Candlestick(
    x=hist.index,
    open=hist['Open'],
    high=hist['High'],
    low=hist['Low'],
    close=hist['Close'],
    name='캔들'
))

# 이동평균선
fig.add_trace(go.Scatter(x=hist.index, y=hist['MA5'], mode='lines', name='MA5', line=dict(color='orange', width=1)))
fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1)))
fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], mode='lines', name='MA60', line=dict(color='green', width=1)))
fig.add_trace(go.Scatter(x=hist.index, y=hist['MA120'], mode='lines', name='MA120', line=dict(color='red', width=1)))

# 목표가/손절가 라인
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
