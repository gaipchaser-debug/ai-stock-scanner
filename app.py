import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
import os

st.set_page_config(page_title="전문가급 주식 분석 시스템", page_icon="📊", layout="wide")

# 종목 매핑
STOCK_MAP = {
    "삼성전자": "005930", "sk하이닉스": "000660", "카카오": "035720",
    "네이버": "035420", "lg화학": "051910", "현대차": "005380",
    "셀트리온": "068270", "삼성바이오로직스": "207940", "포스코홀딩스": "005490",
    "kb금융": "105560", "신한지주": "055550", "키움증권": "039490"
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
        name = [k for k, v in STOCK_MAP.items() if v == code][0]
        return code, name.upper()
    return user_input, None

# === 헤더 ===
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🎯 승률 50% → 70% 향상 목표")
st.markdown("**✅ 수학적 검증 | ✅ 명확한 신호 | ✅ 실전 전략**")
st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input("종목 입력 (예: 삼성전자, 005930)", key="ticker", on_change=reset_session)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 분석 시작", type="primary", use_container_width=True)

if not ticker_input or not analyze_btn:
    st.info("👆 종목을 입력하고 분석 시작 버튼을 클릭하세요")
    st.stop()

ticker_code, parsed_name = parse_ticker(ticker_input)
if st.session_state.current_ticker != ticker_code:
    st.session_state.current_ticker = ticker_code
    reset_session()

yf_ticker = ticker_code + ".KS" if ticker_code.isdigit() else ticker_code

# === 데이터 로드 ===
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
    except:
        return None, None, None

with st.spinner("📊 데이터 로딩 중..."):
    company_name, current_price, hist = load_data(yf_ticker)

if hist is None or hist.empty:
    st.error(f"❌ 종목 '{ticker_input}' 데이터를 찾을 수 없습니다")
    st.stop()

display_name = parsed_name if parsed_name else company_name

st.header(f"🏢 {display_name} ({ticker_code})")
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

# ==========================================
# 모듈 1: 추세 및 패턴 인식 (승률 50~55% 베이스라인)
# ==========================================
st.header("📐 모듈 1: 추세 및 패턴 인식")
st.caption("캔들스틱 패턴 + 이동평균선 정배열 + 골든크로스")

def detect_candlestick_pattern(hist):
    """캔들스틱 패턴 감지"""
    if len(hist) < 3:
        return "패턴 없음", 50
    
    last = hist.iloc[-1]
    prev = hist.iloc[-2]
    prev2 = hist.iloc[-3]
    
    o, c, h, l = last['Open'], last['Close'], last['High'], last['Low']
    po, pc = prev['Open'], prev['Close']
    
    body = abs(c - o)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    
    # 상승 장악형 (Bullish Engulfing)
    if c > o and pc < po:
        if c > po and o < pc:
            return "상승 장악형 (강세)", 80
    
    # 망치형 (Hammer) - 바닥권 반전
    if lower_shadow > body * 2 and upper_shadow < body * 0.3:
        if c > o:  # 양봉이면 더 신뢰
            return "망치형 (바닥 반전)", 75
    
    # 샛별형 (Morning Star) - 3개 캔들 패턴
    if prev2['Close'] < prev2['Open']:  # 첫째 음봉
        if abs(prev['Close'] - prev['Open']) < body * 0.5:  # 둘째 작은 몸통
            if c > o and c > prev2['Open']:  # 셋째 양봉
                return "샛별형 (반전)", 70
    
    # 하락 장악형 (Bearish Engulfing)
    if c < o and pc > po:
        if c < po and o > pc:
            return "하락 장악형 (약세)", 20
    
    # 유성형 (Shooting Star) - 고점 저항
    if upper_shadow > body * 2 and lower_shadow < body * 0.3:
        if c < o:  # 음봉이면 더 확실
            return "유성형 (고점 저항)", 25
    
    return "패턴 없음", 50

def calculate_ma(prices, period):
    """이동평균선 계산"""
    return prices.rolling(window=period).mean()

def check_ma_alignment(ma5, ma20, ma60, ma120):
    """정배열/역배열 확인"""
    if pd.isna(ma5) or pd.isna(ma20) or pd.isna(ma60) or pd.isna(ma120):
        return "데이터 부족", 50
    
    # 정배열: MA5 > MA20 > MA60 > MA120
    if ma5 > ma20 > ma60 > ma120:
        return "정배열 (강세 추세)", 85
    # 역배열: MA5 < MA20 < MA60 < MA120
    elif ma5 < ma20 < ma60 < ma120:
        return "역배열 (약세 추세) ⚠️", 20
    else:
        return "혼조 (중립)", 50

def check_golden_cross(ma20_curr, ma20_prev, ma60_curr, ma60_prev):
    """골든크로스/데드크로스 확인"""
    if pd.isna(ma20_curr) or pd.isna(ma60_curr):
        return "확인 불가", 50
    
    # 골든크로스: 20일선이 60일선을 하→상 돌파
    if ma20_prev < ma60_prev and ma20_curr > ma60_curr:
        return "골든크로스 발생 ✨", 90
    # 데드크로스: 20일선이 60일선을 상→하 돌파
    elif ma20_prev > ma60_prev and ma20_curr < ma60_curr:
        return "데드크로스 발생 ⚠️", 10
    # 현재 상태 유지
    elif ma20_curr > ma60_curr:
        return "20일선 > 60일선 (양호)", 70
    else:
        return "20일선 < 60일선 (약세)", 30

# 이동평균선 계산
prices = hist['Close']
ma5 = calculate_ma(prices, 5)
ma20 = calculate_ma(prices, 20)
ma60 = calculate_ma(prices, 60)
ma120 = calculate_ma(prices, 120)

# 1) 캔들 패턴 감지
pattern_name, pattern_score = detect_candlestick_pattern(hist)

# 2) 정배열/역배열
alignment_text, alignment_score = check_ma_alignment(
    ma5.iloc[-1], ma20.iloc[-1], ma60.iloc[-1], ma120.iloc[-1]
)

# 3) 골든크로스
if len(ma20) >= 2 and len(ma60) >= 2:
    gc_text, gc_score = check_golden_cross(
        ma20.iloc[-1], ma20.iloc[-2],
        ma60.iloc[-1], ma60.iloc[-2]
    )
else:
    gc_text, gc_score = "데이터 부족", 50

# 모듈1 종합 점수 (가중평균: 패턴 40%, 정배열 40%, 골든크로스 20%)
module1_score = pattern_score * 0.4 + alignment_score * 0.4 + gc_score * 0.2

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🕯️ 캔들 패턴", pattern_name, f"{pattern_score}점")
with col2:
    st.metric("📊 이동평균 정렬", alignment_text, f"{alignment_score}점")
with col3:
    st.metric("✨ 골든/데드크로스", gc_text, f"{gc_score}점")

# 모듈1 판정
if module1_score >= 70:
    st.success(f"✅ **모듈1 통과** ({module1_score:.0f}점) - 추세 상승 환경 확인")
elif module1_score <= 40:
    st.error(f"❌ **모듈1 실패** ({module1_score:.0f}점) - **투자 부적합** (역배열 또는 약세 패턴)")
else:
    st.warning(f"⚠️ **모듈1 보통** ({module1_score:.0f}점) - 신중 판단 필요")

st.markdown("---")

# ==========================================
# 모듈 2: 거래량 및 수급 검증 (Fakeout 필터링)
# ==========================================
st.header("📊 모듈 2: 거래량 및 수급 검증")
st.caption("진성 신호 vs 거짓 신호 구분 + 페니스탁 필터")

def analyze_volume(hist):
    """거래량 지표 분석"""
    volumes = hist['Volume']
    prices = hist['Close']
    
    # 1) 20일 평균 거래량
    vma20 = volumes.rolling(window=20).mean().iloc[-1]
    today_vol = volumes.iloc[-1]
    vol_ratio = today_vol / vma20 if vma20 > 0 else 0
    
    # 돌파 신뢰도: 거래량 2배 이상?
    breakthrough = vol_ratio >= 2.0
    
    # 2) 조정 건전성
    if len(prices) >= 2 and len(volumes) >= 2:
        price_change = (prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]
        vol_change = (volumes.iloc[-1] - volumes.iloc[-2]) / volumes.iloc[-2]
        
        # 주가 하락 + 거래량 감소 → 건전한 조정
        healthy_correction = (price_change < 0) and (vol_change < 0)
        # 주가 하락 + 거래량 폭증 → 패닉 셀링
        panic_selling = (price_change < -0.03) and (vol_change > 0.5)
    else:
        healthy_correction = False
        panic_selling = False
    
    # 3) 일평균 거래대금 (페니스탁 필터)
    avg_value = (prices.iloc[-20:] * volumes.iloc[-20:]).mean()
    penny_stock = avg_value < 10_000_000_000  # 100억 원 미만
    
    return {
        'vol_ratio': vol_ratio,
        'breakthrough': breakthrough,
        'healthy_correction': healthy_correction,
        'panic_selling': panic_selling,
        'avg_value': avg_value,
        'penny_stock': penny_stock
    }

vol_metrics = analyze_volume(hist)

# 모듈2 점수 계산
module2_score = 50
if vol_metrics['breakthrough']:
    module2_score += 30  # 진성 신호
if vol_metrics['healthy_correction']:
    module2_score += 10  # 건전한 조정
if vol_metrics['panic_selling']:
    module2_score -= 40  # 패닉 셀링
if vol_metrics['penny_stock']:
    module2_score -= 20  # 페니스탁 리스크

module2_score = max(0, min(100, module2_score))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📈 거래량 비율", 
              f"{vol_metrics['vol_ratio']:.2f}배",
              "✅ 돌파 신뢰" if vol_metrics['breakthrough'] else "보통")
with col2:
    if vol_metrics['panic_selling']:
        st.metric("⚠️ 조정 상태", "패닉 셀링", "위험")
    elif vol_metrics['healthy_correction']:
        st.metric("✅ 조정 상태", "건전한 조정", "양호")
    else:
        st.metric("📊 조정 상태", "정상 범위", "-")
with col3:
    st.metric("💰 일평균 거래대금", 
              f"{vol_metrics['avg_value']/1e8:.0f}억원",
              "⚠️ 페니스탁" if vol_metrics['penny_stock'] else "적정 유동성")

# 모듈2 판정
if module2_score >= 70:
    st.success(f"✅ **모듈2 통과** ({module2_score:.0f}점) - **진성 신호** (거래량 뒷받침)")
elif module2_score <= 40:
    st.error(f"❌ **모듈2 실패** ({module2_score:.0f}점) - **거짓 신호(Fakeout)** 또는 페닉 셀링")
else:
    st.warning(f"⚠️ **모듈2 보통** ({module2_score:.0f}점) - 거래량 검증 필요")

st.markdown("---")

# ==========================================
# 모듈 3: 매수 시그널 발생 (4대 AND 조건)
# ==========================================
st.header("🎯 모듈 3: 매수 시그널 발생")
st.caption("4가지 조건 동시 충족 시 매수 시그널")

def check_buy_signal(current_price, hist, ma120):
    """4대 매수 조건 검증"""
    conditions = {}
    
    # 조건1: 종가 > 120일선
    if not pd.isna(ma120.iloc[-1]):
        cond1 = current_price > ma120.iloc[-1]
    else:
        cond1 = False
    conditions['above_ma120'] = cond1
    
    # 조건2: 현재가 = 20일 최고가 경신
    high_20d = hist['High'].iloc[-20:].max()
    cond2 = current_price >= high_20d * 0.995  # 0.5% 이내 허용
    conditions['new_high_20d'] = cond2
    
    # 조건3: 등락률 +5% ~ +15%
    if len(hist) >= 2:
        prev_close = hist['Close'].iloc[-2]
        pct_change = (current_price - prev_close) / prev_close * 100
        cond3 = 5 <= pct_change <= 15
        conditions['pct_change'] = pct_change
    else:
        cond3 = False
        conditions['pct_change'] = 0
    conditions['optimal_gain'] = cond3
    
    # 조건4: 거래량 ≥ 평균 × 2배
    vma20 = hist['Volume'].iloc[-20:].mean()
    today_vol = hist['Volume'].iloc[-1]
    cond4 = today_vol >= vma20 * 2
    conditions['volume_2x'] = cond4
    
    # 전체 조건 충족 여부
    conditions['all_pass'] = cond1 and cond2 and cond3 and cond4
    conditions['pass_count'] = sum([cond1, cond2, cond3, cond4])
    
    return conditions

buy_conditions = check_buy_signal(current_price, hist, ma120)

st.markdown("### 📋 4대 매수 조건 체크")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**1️⃣ 120일선 위 위치**: {'✅ 통과' if buy_conditions['above_ma120'] else '❌ 실패'}")
    st.markdown(f"**2️⃣ 20일 최고가 경신**: {'✅ 통과' if buy_conditions['new_high_20d'] else '❌ 실패'}")
with col2:
    st.markdown(f"**3️⃣ 등락률 +5~15%**: {'✅ 통과' if buy_conditions['optimal_gain'] else f'❌ 실패 ({buy_conditions[\"pct_change\"]:.1f}%)'}")
    st.markdown(f"**4️⃣ 거래량 2배 이상**: {'✅ 통과' if buy_conditions['volume_2x'] else '❌ 실패'}")

# 모듈3 점수
if buy_conditions['all_pass']:
    module3_score = 95
    st.success(f"🎯 **매수 시그널 발생!** ({module3_score}점) - 4/4 조건 모두 충족")
else:
    passed = buy_conditions['pass_count']
    module3_score = 25 + passed * 15
    st.warning(f"⏳ **매수 시그널 대기 중** ({module3_score}점) - {passed}/4 조건 충족")

st.markdown("---")

# ==========================================
# 모듈 4: 리스크 관리 및 청산 (손익비 1:2)
# ==========================================
st.header("🛡️ 모듈 4: 리스크 관리 및 청산")
st.caption("손절가 / 익절가 / 트레일링 스탑 제시")

def calculate_risk_management(current_price, hist, ma20):
    """손절가/익절가 계산"""
    today = hist.iloc[-1]
    
    # 손절가 후보 5가지
    stop_candidates = []
    stop_candidates.append(('기준봉 시가', today['Open']))
    stop_candidates.append(('기준봉 저가', today['Low']))
    stop_candidates.append(('현재가 -3%', current_price * 0.97))
    stop_candidates.append(('현재가 -5%', current_price * 0.95))
    if not pd.isna(ma20.iloc[-1]):
        stop_candidates.append(('20일 이평선', ma20.iloc[-1]))
    
    # 최종 손절가: 현재가보다 낮은 것 중 가장 가까운 값
    valid_stops = [price for name, price in stop_candidates if price < current_price]
    if valid_stops:
        stop_loss = max(valid_stops)
    else:
        stop_loss = current_price * 0.95
    
    # 익절가: 손익비 1:2 적용
    risk = current_price - stop_loss
    take_profit = current_price + risk * 2
    
    # 트레일링 스탑: 최근 5일 최고가 -3%
    recent_high = hist['High'].iloc[-5:].max()
    trailing_stop = recent_high * 0.97
    
    return {
        'stop_loss': stop_loss,
        'stop_loss_pct': (stop_loss - current_price) / current_price * 100,
        'take_profit': take_profit,
        'take_profit_pct': (take_profit - current_price) / current_price * 100,
        'trailing_stop': trailing_stop,
        'risk': risk,
        'reward': risk * 2,
        'risk_reward_ratio': 2.0
    }

risk_mgmt = calculate_risk_management(current_price, hist, ma20)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💀 손절가", 
              f"{risk_mgmt['stop_loss']:,.0f}원",
              f"{risk_mgmt['stop_loss_pct']:.1f}%")
with col2:
    st.metric("💰 익절가 (목표가)",
              f"{risk_mgmt['take_profit']:,.0f}원",
              f"+{risk_mgmt['take_profit_pct']:.1f}%")
with col3:
    st.metric("📉 트레일링 스탑",
              f"{risk_mgmt['trailing_stop']:,.0f}원",
              "고점 -3%")

st.info(f"📐 **손익비**: 1:{risk_mgmt['risk_reward_ratio']:.1f} (리스크 {risk_mgmt['risk']:,.0f}원 대비 수익 {risk_mgmt['reward']:,.0f}원 기대)")

st.markdown("---")

# ==========================================
# 최종 종합 판단 (가중 평균)
# ==========================================
st.header("🎯 최종 투자 판단")

# 가중 평균: 모듈1(30%) + 모듈2(30%) + 모듈3(40%)
final_score = (module1_score * 0.3 + module2_score * 0.3 + module3_score * 0.4)

st.markdown(f"""
**📊 모듈별 점수**:
- 모듈1 (추세 및 패턴): **{module1_score:.0f}점** × 0.3 = {module1_score * 0.3:.1f}
- 모듈2 (거래량 검증): **{module2_score:.0f}점** × 0.3 = {module2_score * 0.3:.1f}
- 모듈3 (매수 시그널): **{module3_score:.0f}점** × 0.4 = {module3_score * 0.4:.1f}

**🎯 최종 점수**: **{final_score:.0f}점**
""")

col1, col2 = st.columns([2, 1])

with col1:
    if final_score >= 75:
        st.success(f"""
## 🟢 강력 매수 추천 ({final_score:.0f}점)

### 💼 투자 전략
- **진입가**: {current_price:,.0f}원 (현재가)
- **손절가**: {risk_mgmt['stop_loss']:,.0f}원 ({risk_mgmt['stop_loss_pct']:.1f}%)
- **익절가**: {risk_mgmt['take_profit']:,.0f}원 (+{risk_mgmt['take_profit_pct']:.1f}%)
- **손익비**: 1:{risk_mgmt['risk_reward_ratio']:.1f}

### 📈 판단 근거
- ✅ 모듈1: {module1_score:.0f}점 - {alignment_text}
- ✅ 모듈2: {module2_score:.0f}점 - {'진성 신호' if vol_metrics['breakthrough'] else '거래량 검증'}
- ✅ 모듈3: {module3_score:.0f}점 - {'매수 시그널 발생' if buy_conditions['all_pass'] else f'{buy_conditions["pass_count"]}/4 조건 충족'}

### 💡 투자 조언
지금은 **추세, 거래량, 시그널** 모두 긍정적입니다.  
목표가 **{risk_mgmt['take_profit']:,.0f}원**을 노리며 진입하되,  
**{risk_mgmt['stop_loss']:,.0f}원** 아래로 이탈 시 손절하세요.
""")
    
    elif final_score >= 55:
        st.warning(f"""
## 🟡 신중 매수 ({final_score:.0f}점)

### 💼 투자 전략
- **분할 매수 추천** (50% 진입)
- **손절가**: {risk_mgmt['stop_loss']:,.0f}원
- **목표가**: {risk_mgmt['take_profit']:,.0f}원

### 📊 판단 근거
- 모듈1: {module1_score:.0f}점 - {alignment_text}
- 모듈2: {module2_score:.0f}점
- 모듈3: {module3_score:.0f}점

### ⚠️ 주의사항
일부 조건이 미충족 상태입니다.  
**분할 매수**로 리스크를 낮추고, 추가 확인 후 진입을 늘리세요.
""")
    
    else:
        st.error(f"""
## 🔴 매수 비추천 ({final_score:.0f}점)

### ❌ 판단 근거
- 모듈1: **{module1_score:.0f}점** - 추세 약세
- 모듈2: **{module2_score:.0f}점** - 거래량 부족
- 모듈3: **{module3_score:.0f}점** - 시그널 미발생

### 💡 투자 조언
현재는 **투자 부적합** 상태입니다.  
추세 전환 또는 골든크로스 발생 시까지 **관망**하세요.
""")

with col2:
    # 점수 게이지
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "종합 점수", 'font': {'size': 20}},
        number={'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkgreen" if final_score >= 75 else "orange" if final_score >= 55 else "darkred"},
            'steps': [
                {'range': [0, 55], 'color': "rgba(255,0,0,0.2)"},
                {'range': [55, 75], 'color': "rgba(255,165,0,0.2)"},
                {'range': [75, 100], 'color': "rgba(0,128,0,0.2)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ==========================================
# 종합 차트
# ==========================================
st.header("📈 종합 기술적 분석 차트")

fig = go.Figure()

# 캔들스틱 (최근 60일)
recent_hist = hist.iloc[-60:]
fig.add_trace(go.Candlestick(
    x=recent_hist.index,
    open=recent_hist['Open'],
    high=recent_hist['High'],
    low=recent_hist['Low'],
    close=recent_hist['Close'],
    name='캔들'
))

# 이동평균선
fig.add_trace(go.Scatter(
    x=recent_hist.index,
    y=ma20.iloc[-60:],
    name='20일선',
    line=dict(color='orange', width=2)
))
fig.add_trace(go.Scatter(
    x=recent_hist.index,
    y=ma60.iloc[-60:],
    name='60일선',
    line=dict(color='blue', width=2)
))
fig.add_trace(go.Scatter(
    x=recent_hist.index,
    y=ma120.iloc[-60:],
    name='120일선',
    line=dict(color='purple', width=2)
))

# 손절가 / 익절가 수평선
fig.add_hline(
    y=risk_mgmt['stop_loss'],
    line_dash="dash",
    line_color="red",
    line_width=2,
    annotation_text=f"손절가 {risk_mgmt['stop_loss']:,.0f}원",
    annotation_position="right"
)
fig.add_hline(
    y=risk_mgmt['take_profit'],
    line_dash="dash",
    line_color="green",
    line_width=2,
    annotation_text=f"익절가 {risk_mgmt['take_profit']:,.0f}원",
    annotation_position="right"
)

fig.update_layout(
    title=f"{display_name} 기술적 분석 (최근 60일)",
    xaxis_title="날짜",
    yaxis_title="가격 (원)",
    height=600,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(f"💡 분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("⚠️ 본 분석은 투자 참고용이며, 최종 투자 결정은 본인의 책임입니다.")
st.caption("📊 시스템: 승률 50% → 70% 향상 목표 | 수학적 검증 + 명확한 신호 + 실전 전략")
