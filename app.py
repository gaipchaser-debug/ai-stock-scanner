import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

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
                code = row['Code']
                name = row['Name'].lower().strip()
                market = row['Market']
                
                if market == 'KOSPI':
                    ticker = f"{code}.KS"
                else:
                    ticker = f"{code}.KQ"
                
                stock_dict[name] = ticker
                stock_dict[code] = ticker
            
            return stock_dict, all_stocks
        else:
            return {}, None
    
    except Exception as e:
        st.error(f"종목 리스트 로드 실패: {str(e)}")
        return {}, None

def search_stock(query, stock_dict, all_stocks_df):
    """종목 검색"""
    query = query.strip().lower()
    
    if query in stock_dict:
        return stock_dict[query], None
    
    if all_stocks_df is not None:
        matches = all_stocks_df[
            all_stocks_df['Name'].str.lower().str.contains(query, na=False)
        ]
        
        if len(matches) == 1:
            code = matches.iloc[0]['Code']
            market = matches.iloc[0]['Market']
            ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
            return ticker, None
        
        elif len(matches) > 1:
            return None, matches.head(10)
    
    return None, None

def load_stock_data(ticker, max_retries=3):
    """주식 데이터 로드"""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            
            for period in ["6mo", "3mo", "2mo", "1mo"]:
                hist = stock.history(period=period)
                if not hist.empty and len(hist) >= 20:
                    try:
                        info = stock.info
                        name = info.get('longName', info.get('shortName', ticker))
                    except:
                        name = ticker
                    
                    current_price = hist['Close'].iloc[-1]
                    return True, name, current_price, hist
            
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

# ========== UI ==========
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🇰🇷 한국 주식 전체 검색 (FinanceDataReader)")
st.markdown("**✅ 코스피 + 코스닥 전체 | ✅ 2,500+ 종목 | ✅ 4대 모듈 시각화**")
st.markdown("---")

# 종목 리스트 로드
if st.session_state.stock_list is None:
    with st.spinner("📡 전체 종목 리스트 로딩 중..."):
        stock_dict, all_stocks_df = load_all_korean_stocks()
        
        if stock_dict:
            st.session_state.stock_list = (stock_dict, all_stocks_df)
            st.success(f"✅ {len(stock_dict)//2}개 종목 로드 완료")
        else:
            st.error("❌ FinanceDataReader 설치 필요: pip install finance-datareader")
            st.stop()

stock_dict, all_stocks_df = st.session_state.stock_list

# 검색
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "🔍 종목 검색",
        placeholder="종목명 또는 코드 입력 (예: 삼성전자, 005930)"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🔎 검색", type="primary", use_container_width=True)

if not query:
    st.info("👆 종목을 입력하세요")
    
    with st.expander("💡 검색 가능 종목 예시"):
        if all_stocks_df is not None:
            sample = all_stocks_df.head(20)[['Code', 'Name', 'Market']]
            st.dataframe(sample, use_container_width=True)
    
    st.stop()

if not search_btn:
    st.stop()

# 검색 실행
with st.spinner("🔍 검색 중..."):
    ticker, matches = search_stock(query, stock_dict, all_stocks_df)
    
    if ticker:
        st.success(f"✅ 발견: {ticker}")
        final_ticker = ticker
    
    elif matches is not None and len(matches) > 0:
        st.warning(f"⚠️ {len(matches)}개 종목 발견")
        st.markdown("### 📋 선택하세요")
        
        for idx, row in matches.iterrows():
            code = row['Code']
            name = row['Name']
            market = row['Market']
            ticker_code = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{name}** ({code}) - {market}")
            with col2:
                if st.button("선택", key=f"sel_{idx}", use_container_width=True):
                    st.session_state.current_ticker = ticker_code
                    st.rerun()
        
        st.stop()
    
    else:
        st.error(f"❌ '{query}' 종목 없음")
        st.stop()

# 데이터 로드
st.markdown("---")
with st.spinner("📊 데이터 로딩..."):
    is_valid, company_name, current_price, hist = load_stock_data(final_ticker, max_retries=3)

if not is_valid:
    st.error(f"❌ 데이터 로드 실패: {final_ticker}")
    st.stop()

# ========== 종목 정보 ==========
st.header(f"🏢 {company_name}")
st.subheader(f"📊 {final_ticker}")

# 현재가 표시
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💰 현재가", f"{current_price:,.0f}원")

if len(hist) >= 2:
    prev = hist['Close'].iloc[-2]
    change = current_price - prev
    pct = (change / prev) * 100
    
    with col2:
        if change > 0:
            st.metric("전일 대비", f"+{change:,.0f}원", f"+{pct:.2f}%", delta_color="normal")
        elif change < 0:
            st.metric("전일 대비", f"{change:,.0f}원", f"{pct:.2f}%", delta_color="inverse")
        else:
            st.metric("전일 대비", "보합", "0.00%")

with col3:
    st.metric("데이터 기간", f"{len(hist)}일")

if st.button("🔄 다시 검색", use_container_width=True):
    reset_session()
    st.rerun()

st.markdown("---")

# ========== 모듈 계산 ==========

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

# 모듈 1
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

# 타입 안전성 확보
candle_score = int(candle_score)
ma_score = int(ma_score)
cross_score = int(cross_score)

module1_score = int((candle_score * 0.3 + ma_score * 0.4 + cross_score * 0.3))

# 모듈 2
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
    breakout = "부족"
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
    adj_health = "부족"
    adj_score = 0

volume_score = int(volume_score)
adj_score = int(adj_score)
module2_score = max(0, min(100, int(volume_score + adj_score)))

# 모듈 3
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

if satisfied == 4:
    module3_score = 95
elif satisfied == 3:
    module3_score = 70
else:
    module3_score = 40

module3_score = int(module3_score)

# 모듈 4
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

# 최종 점수
final_score = int(module1_score * 0.3 + module2_score * 0.3 + module3_score * 0.4)

# ========== 모듈 시각화 ==========
st.subheader("📊 4대 모듈 분석 결과")

# 모듈 점수 게이지 차트
fig_modules = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
           [{'type': 'indicator'}, {'type': 'indicator'}]],
    subplot_titles=("모듈 1: 추세 & 패턴", "모듈 2: 거래량 검증", 
                    "모듈 3: 매수 신호", "모듈 4: 리스크 관리")
)

# 모듈 1
fig_modules.add_trace(go.Indicator(
    mode="gauge+number+delta",
    value=module1_score,
    title={'text': f"{module1_score}점"},
    delta={'reference': 75},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 55], 'color': "lightgray"},
            {'range': [55, 75], 'color': "yellow"},
            {'range': [75, 100], 'color': "lightgreen"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 75
        }
    }
), row=1, col=1)

# 모듈 2
fig_modules.add_trace(go.Indicator(
    mode="gauge+number+delta",
    value=module2_score,
    title={'text': f"{module2_score}점"},
    delta={'reference': 75},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkorange"},
        'steps': [
            {'range': [0, 55], 'color': "lightgray"},
            {'range': [55, 75], 'color': "yellow"},
            {'range': [75, 100], 'color': "lightgreen"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 75
        }
    }
), row=1, col=2)

# 모듈 3
fig_modules.add_trace(go.Indicator(
    mode="gauge+number+delta",
    value=module3_score,
    title={'text': f"{module3_score}점"},
    delta={'reference': 75},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkgreen"},
        'steps': [
            {'range': [0, 55], 'color': "lightgray"},
            {'range': [55, 75], 'color': "yellow"},
            {'range': [75, 100], 'color': "lightgreen"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 75
        }
    }
), row=2, col=1)

# 모듈 4
fig_modules.add_trace(go.Indicator(
    mode="number+delta",
    value=100,  # 리스크 관리는 항상 완료
    title={'text': "설정 완료"},
    delta={'reference': 100},
    number={'suffix': "%", 'font': {'size': 40}}
), row=2, col=2)

fig_modules.update_layout(
    height=600,
    showlegend=False,
    title_text="4대 모듈 점수 (75점 이상: 강력 매수, 55~74점: 신중 매수, 55점 미만: 부적합)"
)

st.plotly_chart(fig_modules, use_container_width=True)

st.markdown("---")

# ========== 모듈 상세 ==========
tab1, tab2, tab3, tab4 = st.tabs(["📈 모듈 1", "📊 모듈 2", "🎯 모듈 3", "🛡️ 모듈 4"])

with tab1:
    st.subheader("📈 모듈 1: 추세 & 패턴 인식")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("캔들 패턴", candle_pattern, f"{candle_score}점")
    with col2:
        st.metric("이동평균", ma_alignment, f"{ma_score}점")
    with col3:
        st.metric("크로스", cross, f"{cross_score}점")
    
    st.success(f"**모듈1 종합: {module1_score}점**")
    
    # 이동평균선 정보
    if pd.notna(ma5) and pd.notna(ma20) and pd.notna(ma60) and pd.notna(ma120):
        st.markdown("**이동평균선 현황:**")
        ma_df = pd.DataFrame({
            '이동평균': ['MA5', 'MA20', 'MA60', 'MA120'],
            '가격': [f"{ma5:,.0f}원", f"{ma20:,.0f}원", f"{ma60:,.0f}원", f"{ma120:,.0f}원"]
        })
        st.dataframe(ma_df, use_container_width=True)

with tab2:
    st.subheader("📊 모듈 2: 거래량 & 공급 검증")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("거래량 배율", f"{volume_ratio:.2f}배", breakout)
    with col2:
        st.metric("조정 건전성", adj_health, f"{adj_score:+d}점")
    
    st.success(f"**모듈2 종합: {module2_score}점**")
    
    # 거래량 차트
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(
        x=hist.index[-20:],
        y=hist['Volume'].tail(20),
        name='거래량',
        marker_color='lightblue'
    ))
    fig_volume.add_trace(go.Scatter(
        x=hist.index[-20:],
        y=hist['Volume_MA20'].tail(20),
        name='20일 평균',
        line=dict(color='red', width=2)
    ))
    fig_volume.update_layout(
        title="최근 20일 거래량",
        xaxis_title="날짜",
        yaxis_title="거래량",
        height=400
    )
    st.plotly_chart(fig_volume, use_container_width=True)

with tab3:
    st.subheader("🎯 모듈 3: 매수 신호 (4-AND 조건)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("120일선 상회", "✅" if cond1 else "❌")
    with col2:
        st.metric("20일 신고가", "✅" if cond2 else "❌")
    with col3:
        st.metric("최적 상승폭", "✅" if cond3 else "❌")
    with col4:
        st.metric("거래량 급증", "✅" if cond4 else "❌")
    
    if satisfied == 4:
        st.success("🎉 **모든 조건 충족: 강력 매수 신호**")
    elif satisfied == 3:
        st.warning("⚠️ **3개 조건 충족: 신중 매수**")
    else:
        st.error("❌ **조건 미달: 매수 부적합**")
    
    st.success(f"**모듈3 종합: {module3_score}점** (충족: {satisfied}/4)")

with tab4:
    st.subheader("🛡️ 모듈 4: 리스크 관리 & 청산 전략")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💼 진입가", f"{current_price:,.0f}원", "현재가")
    with col2:
        st.metric("🛑 손절가", f"{final_sl:,.0f}원", f"{risk:.2f}%")
    with col3:
        st.metric("🎯 목표가", f"{target:,.0f}원", f"+{reward:.2f}%")
    
    st.markdown(f"**리스크:리워드 비율**: 1:2")
    st.success("**모듈4: 리스크 관리 설정 완료**")
    
    # 손익 시뮬레이션
    st.markdown("**📈 손익 시뮬레이션**")
    price_range = np.linspace(final_sl * 0.95, target * 1.05, 100)
    profit_loss = [(p - current_price) / current_price * 100 for p in price_range]
    
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=price_range,
        y=profit_loss,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(0,176,240,0.2)',
        line=dict(color='blue', width=2),
        name='손익률'
    ))
    fig_sim.add_vline(x=current_price, line_dash="dash", line_color="black", annotation_text="현재가")
    fig_sim.add_vline(x=final_sl, line_dash="dot", line_color="red", annotation_text="손절가")
    fig_sim.add_vline(x=target, line_dash="dot", line_color="green", annotation_text="목표가")
    fig_sim.add_hline(y=0, line_dash="solid", line_color="gray")
    
    fig_sim.update_layout(
        title="가격별 손익률",
        xaxis_title="주가 (원)",
        yaxis_title="손익률 (%)",
        height=400
    )
    st.plotly_chart(fig_sim, use_container_width=True)

st.markdown("---")

# ========== 최종 평가 ==========
st.header("🏆 최종 종합 평가")

# 최종 점수 게이지
fig_final = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=final_score,
    title={'text': "최종 점수", 'font': {'size': 24}},
    delta={'reference': 75, 'font': {'size': 20}},
    number={'font': {'size': 60}},
    gauge={
        'axis': {'range': [0, 100], 'tickwidth': 1},
        'bar': {'color': "darkblue" if final_score >= 75 else "orange" if final_score >= 55 else "red"},
        'bgcolor': "white",
        'steps': [
            {'range': [0, 55], 'color': 'rgba(255,0,0,0.2)'},
            {'range': [55, 75], 'color': 'rgba(255,255,0,0.2)'},
            {'range': [75, 100], 'color': 'rgba(0,255,0,0.2)'}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 75
        }
    }
))

fig_final.update_layout(height=400)
st.plotly_chart(fig_final, use_container_width=True)

# 투자 의견
if final_score >= 75:
    st.success("### 🟢 강력 매수 추천")
    st.markdown("**모든 지표가 긍정적입니다. 적극적인 매수를 고려하세요.**")
elif final_score >= 55:
    st.warning("### 🟡 신중 매수")
    st.markdown("**일부 지표가 긍정적입니다. 리스크 관리를 철저히 하세요.**")
else:
    st.error("### 🔴 매수 부적합")
    st.markdown("**현재 매수 시점이 아닙니다. 관망을 권장합니다.**")

# 모듈별 기여도
st.markdown("### 📊 모듈별 기여도")
contribution_df = pd.DataFrame({
    '모듈': ['모듈1: 추세&패턴', '모듈2: 거래량', '모듈3: 매수신호'],
    '점수': [module1_score, module2_score, module3_score],
    '가중치': ['30%', '30%', '40%'],
    '기여도': [
        f"{module1_score * 0.3:.1f}점",
        f"{module2_score * 0.3:.1f}점",
        f"{module3_score * 0.4:.1f}점"
    ]
})
st.dataframe(contribution_df, use_container_width=True)

st.markdown("---")

# ========== 가격 차트 ==========
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

if pd.notna(ma5).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA5'], mode='lines', name='MA5', line=dict(color='orange', width=1)))
if pd.notna(ma20).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1)))
if pd.notna(ma60).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], mode='lines', name='MA60', line=dict(color='green', width=1)))
if pd.notna(ma120).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA120'], mode='lines', name='MA120', line=dict(color='red', width=1)))

fig.add_hline(y=target, line_dash="dot", line_color="green", annotation_text=f"목표가: {target:,.0f}원")
fig.add_hline(y=final_sl, line_dash="dot", line_color="red", annotation_text=f"손절가: {final_sl:,.0f}원")

fig.update_layout(
    title=f"{company_name} 기술적 분석",
    xaxis_title="날짜",
    yaxis_title="가격 (원)",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("🔔 본 분석은 참고용이며, 최종 투자 결정은 본인의 책임입니다.")
