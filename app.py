import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import json
import time
from bs4 import BeautifulSoup
import os

st.set_page_config(page_title="전문가급 주식 분석 시스템", page_icon="📊", layout="wide")

# API 키
def get_api_key(key_name):
    key = os.getenv(key_name)
    if key:
        return key
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return None

GEMINI_API_KEY = get_api_key("GEMINI_API_KEY")
NAVER_CLIENT_ID = get_api_key("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = get_api_key("NAVER_CLIENT_SECRET")

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
    except:
        GEMINI_MODEL = None
else:
    GEMINI_MODEL = None

STOCK_MAP = {
    "삼성전자": "005930", "sk하이닉스": "000660", "카카오": "035720",
    "네이버": "035420", "lg화학": "051910", "현대차": "005380",
    "키움증권": "039490", "미래에셋증권": "006800"
}

if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None

def reset_session():
    st.cache_data.clear()

def parse_ticker(user_input):
    user_input = user_input.strip().lower()
    if user_input.isdigit():
        return user_input, None
    if user_input in STOCK_MAP:
        code = STOCK_MAP[user_input]
        name = next((k for k, v in STOCK_MAP.items() if v == code), None)
        return code, name.upper() if name else None
    return user_input, None

# === 헤더 ===
st.title("📊 전문가급 주식 분석 시스템 (4대 모듈)")
st.markdown("**승률 50% → 70% 상승을 위한 정밀 분석**")
st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input("종목 입력", key="ticker", on_change=reset_session, placeholder="예: 삼성전자, 005930")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 분석", type="primary", use_container_width=True)

if not ticker_input or not analyze_btn:
    st.info("👆 종목을 입력하세요")
    st.stop()

ticker_code, parsed_name = parse_ticker(ticker_input)
if st.session_state.current_ticker != ticker_code:
    st.session_state.current_ticker = ticker_code
    reset_session()

yf_ticker = ticker_code + ".KS" if ticker_code.isdigit() else ticker_code

# === 데이터 로드 ===
@st.cache_data(ttl=300, show_spinner=False)
def load_stock_data(yf_ticker, period="6mo"):
    try:
        ticker = yf.Ticker(yf_ticker)
        hist = ticker.history(period=period)
        if hist.empty:
            return None, None, None
        current_price = hist['Close'].iloc[-1]
        try:
            info = ticker.info
            name = info.get('longName', info.get('shortName', '알 수 없음'))
        except:
            name = "알 수 없음"
        return name, current_price, hist
    except:
        return None, None, None

with st.spinner("📊 데이터 로딩..."):
    company_name, current_price, hist = load_stock_data(yf_ticker)

if not hist or hist.empty:
    st.error(f"❌ 종목 '{ticker_input}' 데이터를 찾을 수 없습니다")
    st.stop()

display_name = parsed_name if parsed_name else company_name
st.header(f"🏢 {display_name} ({ticker_code})")
st.subheader(f"💰 현재가: {current_price:,.0f}원")

if len(hist) >= 2:
    prev = hist['Close'].iloc[-2]
    change = current_price - prev
    pct = (change / prev) * 100
    if change > 0:
        st.markdown(f"📈 전일 대비: +{change:,.0f}원 (+{pct:.2f}%)")
    elif change < 0:
        st.markdown(f"📉 전일 대비: {change:,.0f}원 ({pct:.2f}%)")

st.markdown("---")

# ==========================================
# 모듈 1: 추세 및 패턴 인식 (Trend & Pattern)
# ==========================================
st.subheader("📐 모듈 1: 추세 및 패턴 인식")

def calculate_ma(prices, period):
    """이동평균선 계산"""
    return prices.rolling(window=period).mean()

def detect_pattern(hist):
    """캔들스틱 패턴 감지"""
    if len(hist) < 3:
        return None, 0
    
    last = hist.iloc[-1]
    prev = hist.iloc[-2]
    
    open_p = last['Open']
    close_p = last['Close']
    high_p = last['High']
    low_p = last['Low']
    
    body = abs(close_p - open_p)
    upper_shadow = high_p - max(open_p, close_p)
    lower_shadow = min(open_p, close_p) - low_p
    
    # 상승 장악형 (Bullish Engulfing)
    if close_p > open_p and prev['Close'] < prev['Open']:
        if close_p > prev['Open'] and open_p < prev['Close']:
            return "상승 장악형 (강세)", 80
    
    # 망치형 (Hammer)
    if lower_shadow > body * 2 and upper_shadow < body * 0.3:
        return "망치형 (바닥 반전)", 75
    
    # 샛별형 (Morning Star) - 간단 버전
    if len(hist) >= 3:
        prev2 = hist.iloc[-3]
        if prev2['Close'] < prev2['Open'] and close_p > open_p:
            if prev['Close'] < prev['Open']:
                return "샛별형 (반전)", 70
    
    # 하락 장악형
    if close_p < open_p and prev['Close'] > prev['Open']:
        if close_p < prev['Open'] and open_p > prev['Close']:
            return "하락 장악형 (약세)", 20
    
    # 유성형 (Shooting Star)
    if upper_shadow > body * 2 and lower_shadow < body * 0.3:
        return "유성형 (고점 저항)", 25
    
    return None, 50

def check_ma_alignment(ma5, ma20, ma60, ma120):
    """이동평균 정배열/역배열 확인"""
    # 정배열: 단기 > 중기 > 장기
    if ma5 > ma20 > ma60 > ma120:
        return "정배열 (강세)", 85
    # 역배열
    elif ma5 < ma20 < ma60 < ma120:
        return "역배열 (약세)", 20
    else:
        return "혼조", 50

def check_golden_cross(ma20_current, ma20_prev, ma60_current, ma60_prev):
    """골든크로스 확인"""
    if ma20_prev < ma60_prev and ma20_current > ma60_current:
        return True, 90
    elif ma20_prev > ma60_prev and ma20_current < ma60_current:
        return True, 10  # 데드크로스
    return False, 50

# 이동평균선 계산
prices = hist['Close']
ma5 = calculate_ma(prices, 5)
ma20 = calculate_ma(prices, 20)
ma60 = calculate_ma(prices, 60)
ma120 = calculate_ma(prices, 120)

# 패턴 감지
pattern_name, pattern_score = detect_pattern(hist)

# 정배열/역배열
alignment, alignment_score = check_ma_alignment(
    ma5.iloc[-1], ma20.iloc[-1], ma60.iloc[-1], ma120.iloc[-1]
)

# 골든크로스
golden_cross, gc_score = check_golden_cross(
    ma20.iloc[-1], ma20.iloc[-2],
    ma60.iloc[-1], ma60.iloc[-2]
)

# 모듈1 종합 점수
module1_score = (pattern_score * 0.4 + alignment_score * 0.4 + gc_score * 0.2)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("캔들 패턴", pattern_name if pattern_name else "패턴 없음", f"{pattern_score}점")
with col2:
    st.metric("이동평균 정렬", alignment, f"{alignment_score}점")
with col3:
    st.metric("골든크로스", "발생" if golden_cross else "미발생", f"{gc_score}점")

if module1_score >= 70:
    st.success(f"✅ 모듈1 통과 ({module1_score:.0f}점) - 추세 상승 환경")
elif module1_score <= 40:
    st.error(f"❌ 모듈1 실패 ({module1_score:.0f}점) - 투자 부적합")
else:
    st.warning(f"⚠️ 모듈1 보통 ({module1_score:.0f}점) - 신중 판단")

st.markdown("---")

# ==========================================
# 모듈 2: 거래량 및 수급 검증
# ==========================================
st.subheader("📊 모듈 2: 거래량 및 수급 검증")

def calculate_volume_metrics(hist):
    """거래량 지표 계산"""
    volumes = hist['Volume']
    vma20 = volumes.rolling(window=20).mean()
    
    today_vol = volumes.iloc[-1]
    avg_vol = vma20.iloc[-1]
    
    # 돌파 신뢰도: 오늘 거래량이 평균의 200% 이상?
    vol_ratio = today_vol / avg_vol
    breakthrough = vol_ratio >= 2.0
    
    # 조정 건전성: 주가 하락 시 거래량 감소?
    price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
    vol_change = (today_vol - volumes.iloc[-2]) / volumes.iloc[-2]
    
    healthy_correction = price_change < 0 and vol_change < 0
    panic_selling = price_change < -0.03 and vol_change > 0.5
    
    # 일평균 거래대금
    avg_value = (hist['Close'].iloc[-20:] * hist['Volume'].iloc[-20:]).mean()
    penny_stock = avg_value < 10_000_000_000  # 100억 미만
    
    return {
        'vol_ratio': vol_ratio,
        'breakthrough': breakthrough,
        'healthy_correction': healthy_correction,
        'panic_selling': panic_selling,
        'avg_value': avg_value,
        'penny_stock': penny_stock
    }

vol_metrics = calculate_volume_metrics(hist)

# 모듈2 점수
module2_score = 50
if vol_metrics['breakthrough']:
    module2_score += 30
if vol_metrics['healthy_correction']:
    module2_score += 10
if vol_metrics['panic_selling']:
    module2_score -= 40
if vol_metrics['penny_stock']:
    module2_score -= 20

module2_score = max(0, min(100, module2_score))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("거래량 비율", f"{vol_metrics['vol_ratio']:.1f}배", 
              "돌파 신뢰" if vol_metrics['breakthrough'] else "보통")
with col2:
    st.metric("조정 상태", 
              "건전" if vol_metrics['healthy_correction'] else "패닉" if vol_metrics['panic_selling'] else "정상")
with col3:
    st.metric("일평균 거래대금", f"{vol_metrics['avg_value']/1e8:.0f}억원",
              "⚠️ 페니스탁" if vol_metrics['penny_stock'] else "적정")

if module2_score >= 70:
    st.success(f"✅ 모듈2 통과 ({module2_score:.0f}점) - 진성 신호")
elif module2_score <= 40:
    st.error(f"❌ 모듈2 실패 ({module2_score:.0f}점) - 거짓 신호(Fakeout)")
else:
    st.warning(f"⚠️ 모듈2 보통 ({module2_score:.0f}점)")

st.markdown("---")

# ==========================================
# 모듈 3: 매수 시그널 발생
# ==========================================
st.subheader("🎯 모듈 3: 매수 시그널 발생")

def check_buy_signal(current_price, hist, ma120):
    """매수 시그널 조건 검증"""
    conditions = {}
    
    # 조건1: 120일선 위
    cond1 = current_price > ma120.iloc[-1]
    conditions['above_ma120'] = cond1
    
    # 조건2: 20일 최고가 경신
    high_20d = hist['High'].iloc[-20:].max()
    cond2 = current_price >= high_20d
    conditions['new_high_20d'] = cond2
    
    # 조건3: 등락률 +5% ~ +15%
    prev_close = hist['Close'].iloc[-2]
    pct_change = (current_price - prev_close) / prev_close * 100
    cond3 = 5 <= pct_change <= 15
    conditions['optimal_gain'] = cond3
    conditions['pct_change'] = pct_change
    
    # 조건4: 거래량 2배 이상
    today_vol = hist['Volume'].iloc[-1]
    vma20 = hist['Volume'].iloc[-20:].mean()
    cond4 = today_vol >= vma20 * 2
    conditions['volume_2x'] = cond4
    
    # 전체 조건
    all_pass = cond1 and cond2 and cond3 and cond4
    conditions['signal'] = all_pass
    
    return conditions

buy_conditions = check_buy_signal(current_price, hist, ma120)

st.markdown("### 📋 매수 조건 체크")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"1️⃣ 120일선 위치: {'✅ 통과' if buy_conditions['above_ma120'] else '❌ 실패'}")
    st.markdown(f"2️⃣ 20일 최고가 경신: {'✅ 통과' if buy_conditions['new_high_20d'] else '❌ 실패'}")
with col2:
    st.markdown(f"3️⃣ 등락률 적정: {'✅ 통과' if buy_conditions['optimal_gain'] else f'❌ 실패 ({buy_conditions['pct_change']:.1f}%)'}")
    st.markdown(f"4️⃣ 거래량 2배: {'✅ 통과' if buy_conditions['volume_2x'] else '❌ 실패'}")

if buy_conditions['signal']:
    module3_score = 95
    st.success(f"🎯 **매수 시그널 발생!** ({module3_score}점)")
else:
    passed = sum([buy_conditions['above_ma120'], buy_conditions['new_high_20d'], 
                  buy_conditions['optimal_gain'], buy_conditions['volume_2x']])
    module3_score = 25 + passed * 15
    st.warning(f"⏳ 매수 시그널 대기 중 ({module3_score}점) - {passed}/4 조건 충족")

st.markdown("---")

# ==========================================
# 모듈 4: 리스크 관리 및 청산
# ==========================================
st.subheader("🛡️ 모듈 4: 리스크 관리 및 청산")

def calculate_risk_management(current_price, hist, ma20):
    """손절가/익절가 계산"""
    # 오늘 캔들
    today = hist.iloc[-1]
    
    # 손절가 계산
    stop_loss_options = []
    
    # 1. 진입 기준봉의 시가/저가
    stop_loss_options.append(('기준봉 시가', today['Open']))
    stop_loss_options.append(('기준봉 저가', today['Low']))
    
    # 2. 현재가 대비 -3% ~ -5%
    stop_loss_options.append(('-3% 손절', current_price * 0.97))
    stop_loss_options.append(('-5% 손절', current_price * 0.95))
    
    # 3. 20일 이평선
    stop_loss_options.append(('20일선', ma20.iloc[-1]))
    
    # 최종 손절가: 가장 가까운 하단 가격
    stop_loss = max([price for name, price in stop_loss_options if price < current_price], default=current_price * 0.95)
    
    # 익절가 계산 (손익비 1:2)
    risk = current_price - stop_loss
    take_profit = current_price + risk * 2
    
    # 트레일링 스탑 (고점 -3%)
    recent_high = hist['High'].iloc[-5:].max()
    trailing_stop = recent_high * 0.97
    
    return {
        'stop_loss': stop_loss,
        'stop_loss_pct': (stop_loss - current_price) / current_price * 100,
        'take_profit': take_profit,
        'take_profit_pct': (take_profit - current_price) / current_price * 100,
        'trailing_stop': trailing_stop,
        'risk_reward': 2.0
    }

risk_mgmt = calculate_risk_management(current_price, hist, ma20)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💀 손절가", f"{risk_mgmt['stop_loss']:,.0f}원", 
              f"{risk_mgmt['stop_loss_pct']:.1f}%")
with col2:
    st.metric("💰 익절가", f"{risk_mgmt['take_profit']:,.0f}원",
              f"+{risk_mgmt['take_profit_pct']:.1f}%")
with col3:
    st.metric("📉 트레일링 스탑", f"{risk_mgmt['trailing_stop']:,.0f}원")

st.info(f"📐 **손익비**: 1:{risk_mgmt['risk_reward']:.1f} (리스크 1원당 {risk_mgmt['risk_reward']:.1f}원 수익 기대)")

module4_score = 100  # 리스크 관리는 항상 적용 가능

st.markdown("---")

# ==========================================
# 최종 종합 판단
# ==========================================
st.subheader("🎯 최종 투자 판단")

# 가중 평균 (모듈1 30%, 모듈2 30%, 모듈3 40%)
final_score = (module1_score * 0.3 + module2_score * 0.3 + module3_score * 0.4)

col1, col2 = st.columns([2, 1])
with col1:
    if final_score >= 75:
        st.success(f"""
## 🟢 강력 매수 추천 ({final_score:.0f}점)

**투자 전략**:
- 진입가: {current_price:,.0f}원 (현재가)
- 손절가: {risk_mgmt['stop_loss']:,.0f}원 ({risk_mgmt['stop_loss_pct']:.1f}%)
- 익절가: {risk_mgmt['take_profit']:,.0f}원 (+{risk_mgmt['take_profit_pct']:.1f}%)

**근거**:
- 모듈1 (추세): {module1_score:.0f}점 - {alignment}
- 모듈2 (거래량): {module2_score:.0f}점 - {'진성 신호' if vol_metrics['breakthrough'] else '검증 필요'}
- 모듈3 (시그널): {module3_score:.0f}점 - {'매수 발생' if buy_conditions['signal'] else '대기'}
""")
    elif final_score >= 55:
        st.warning(f"""
## 🟡 신중 매수 ({final_score:.0f}점)

**투자 전략**:
- 분할 매수 추천 (50% 진입)
- 손절가: {risk_mgmt['stop_loss']:,.0f}원
- 목표가: {risk_mgmt['take_profit']:,.0f}원

**주의사항**:
- 일부 조건 미충족
- 추가 확인 필요
""")
    else:
        st.error(f"""
## 🔴 매수 비추천 ({final_score:.0f}점)

**판단 근거**:
- 모듈1: {module1_score:.0f}점 - 추세 약세
- 모듈2: {module2_score:.0f}점 - 거래량 부족
- 모듈3: {module3_score:.0f}점 - 시그널 미발생

**조언**: 추세 전환 시까지 관망
""")

with col2:
    # 점수 게이지
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "종합 점수"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkgreen" if final_score >= 75 else "orange" if final_score >= 55 else "darkred"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

# 차트 표시
st.markdown("---")
st.subheader("📈 종합 차트")

fig = go.Figure()

# 캔들스틱
fig.add_trace(go.Candlestick(
    x=hist.index[-60:],
    open=hist['Open'].iloc[-60:],
    high=hist['High'].iloc[-60:],
    low=hist['Low'].iloc[-60:],
    close=hist['Close'].iloc[-60:],
    name='캔들'
))

# 이동평균선
fig.add_trace(go.Scatter(x=hist.index[-60:], y=ma20.iloc[-60:], name='20일선', line=dict(color='orange', width=1)))
fig.add_trace(go.Scatter(x=hist.index[-60:], y=ma60.iloc[-60:], name='60일선', line=dict(color='blue', width=1)))
fig.add_trace(go.Scatter(x=hist.index[-60:], y=ma120.iloc[-60:], name='120일선', line=dict(color='purple', width=1)))

# 손절/익절가 표시
fig.add_hline(y=risk_mgmt['stop_loss'], line_dash="dash", line_color="red", annotation_text="손절가")
fig.add_hline(y=risk_mgmt['take_profit'], line_dash="dash", line_color="green", annotation_text="익절가")

fig.update_layout(
    title="기술적 분석 차트 (최근 60일)",
    xaxis_title="날짜",
    yaxis_title="가격 (원)",
    height=500,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(f"💡 분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("⚠️ 본 분석은 투자 참고용이며, 최종 결정은 본인의 책임입니다.")
