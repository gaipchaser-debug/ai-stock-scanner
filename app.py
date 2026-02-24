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
                code = str(row['Code'])
                name = str(row['Name']).lower().strip()
                market = str(row['Market'])
                
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
    query = str(query).strip().lower()
    
    if query in stock_dict:
        return stock_dict[query], None
    
    if all_stocks_df is not None:
        matches = all_stocks_df[
            all_stocks_df['Name'].str.lower().str.contains(query, na=False)
        ]
        
        if len(matches) == 1:
            code = str(matches.iloc[0]['Code'])
            market = str(matches.iloc[0]['Market'])
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
                    
                    current_price = float(hist['Close'].iloc[-1])
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

def create_pattern_reference(pattern_type):
    """캔들 패턴 참조 이미지 생성"""
    fig = go.Figure()
    
    if pattern_type == "Bullish Engulfing":
        # 하락 캔들 (이전)
        fig.add_trace(go.Candlestick(
            x=[1],
            open=[105], high=[110], low=[95], close=[98],
            name='이전 캔들',
            increasing_line_color='red',
            decreasing_line_color='red'
        ))
        # 상승 캔들 (현재) - 더 큰 몸통
        fig.add_trace(go.Candlestick(
            x=[2],
            open=[97], high=[125], low=[95], close=[123],
            name='현재 캔들',
            increasing_line_color='green',
            decreasing_line_color='green'
        ))
        title = "Bullish Engulfing 패턴"
        
    elif pattern_type == "Hammer":
        # 망치형 캔들
        fig.add_trace(go.Candlestick(
            x=[1],
            open=[102], high=[105], low=[85], close=[103],
            name='망치형',
            increasing_line_color='green',
            decreasing_line_color='green'
        ))
        title = "Hammer 패턴"
        
    elif pattern_type == "Bearish Engulfing":
        # 상승 캔들 (이전)
        fig.add_trace(go.Candlestick(
            x=[1],
            open=[95], high=[105], low=[93], close=[102],
            name='이전 캔들',
            increasing_line_color='green',
            decreasing_line_color='green'
        ))
        # 하락 캔들 (현재) - 더 큰 몸통
        fig.add_trace(go.Candlestick(
            x=[2],
            open=[103], high=[108], low=[80], close=[82],
            name='현재 캔들',
            increasing_line_color='red',
            decreasing_line_color='red'
        ))
        title = "Bearish Engulfing 패턴"
    
    else:
        # 일반 캔들
        fig.add_trace(go.Candlestick(
            x=[1],
            open=[100], high=[105], low=[95], close=[102],
            name='일반 캔들',
            increasing_line_color='gray',
            decreasing_line_color='gray'
        ))
        title = "일반 캔들"
    
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="가격",
        height=300,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis_showticklabels=False
    )
    
    return fig

def create_actual_candle_chart(hist, num_candles=5):
    """실제 최근 캔들 차트"""
    recent_hist = hist.tail(num_candles)
    
    fig = go.Figure(data=[go.Candlestick(
        x=list(range(len(recent_hist))),
        open=recent_hist['Open'],
        high=recent_hist['High'],
        low=recent_hist['Low'],
        close=recent_hist['Close']
    )])
    
    fig.update_layout(
        title="실제 최근 캔들 (최근 5일)",
        xaxis_title="",
        yaxis_title="가격",
        height=300,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis_showticklabels=False
    )
    
    return fig

# ========== UI ==========
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🇰🇷 한국 주식 전체 검색 + 캔들 패턴 비교")
st.markdown("**✅ 코스피 + 코스닥 | ✅ 패턴 일치도 시각화**")
st.markdown("---")

# 종목 리스트 로드
if st.session_state.stock_list is None:
    with st.spinner("📡 전체 종목 리스트 로딩 중..."):
        stock_dict, all_stocks_df = load_all_korean_stocks()
        
        if stock_dict:
            st.session_state.stock_list = (stock_dict, all_stocks_df)
            st.success(f"✅ {len(stock_dict)//2}개 종목 로드 완료")
        else:
            st.error("❌ FinanceDataReader 설치 필요")
            st.stop()

stock_dict, all_stocks_df = st.session_state.stock_list

# 검색
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "🔍 종목 검색",
        placeholder="종목명 또는 코드 (예: 삼성전자, 005930)"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🔎 검색", type="primary", use_container_width=True)

if not query:
    st.info("👆 종목을 입력하세요")
    with st.expander("💡 검색 가능 종목"):
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
        st.markdown("### 📋 선택")
        
        for idx, row in matches.iterrows():
            code = str(row['Code'])
            name = str(row['Name'])
            market = str(row['Market'])
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
        st.error(f"❌ '{query}' 없음")
        st.stop()

# 데이터 로드
st.markdown("---")
with st.spinner("📊 데이터 로딩..."):
    is_valid, company_name, current_price, hist = load_stock_data(final_ticker, max_retries=3)

if not is_valid:
    st.error(f"❌ 로드 실패: {final_ticker}")
    st.stop()

# ========== 종목 정보 ==========
st.header(f"🏢 {company_name}")
st.subheader(f"📊 {final_ticker}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💰 현재가", f"{current_price:,.0f}원")

if len(hist) >= 2:
    prev = float(hist['Close'].iloc[-2])
    change = current_price - prev
    pct = (change / prev) * 100
    
    with col2:
        if change > 0:
            st.metric("전일대비", f"+{change:,.0f}원", f"+{pct:.2f}%")
        elif change < 0:
            st.metric("전일대비", f"{change:,.0f}원", f"{pct:.2f}%")
        else:
            st.metric("전일대비", "보합", "0.00%")

with col3:
    st.metric("데이터", f"{len(hist)}일")

if st.button("🔄 다시 검색"):
    reset_session()
    st.rerun()

st.markdown("---")

# ========== 계산 ==========

def calculate_ma(data, period):
    return data['Close'].rolling(window=period).mean()

hist['MA5'] = calculate_ma(hist, 5)
hist['MA20'] = calculate_ma(hist, 20)
hist['MA60'] = calculate_ma(hist, 60)
hist['MA120'] = calculate_ma(hist, 120)

latest = hist.iloc[-1]
ma5 = float(latest['MA5']) if pd.notna(latest['MA5']) else current_price
ma20 = float(latest['MA20']) if pd.notna(latest['MA20']) else current_price
ma60 = float(latest['MA60']) if pd.notna(latest['MA60']) else current_price
ma120 = float(latest['MA120']) if pd.notna(latest['MA120']) else current_price

def detect_candle_pattern(hist):
    if len(hist) < 3:
        return "데이터 부족", 50, 0
    
    last = hist.iloc[-1]
    prev = hist.iloc[-2]
    
    last_body = abs(float(last['Close']) - float(last['Open']))
    prev_body = abs(float(prev['Close']) - float(prev['Open']))
    
    # Bullish Engulfing
    if (prev['Close'] < prev['Open'] and 
        last['Close'] > last['Open'] and 
        last_body > prev_body * 1.5):
        match_score = 90
        return "Bullish Engulfing 🟢", 80, match_score
    
    # Hammer
    lower_shadow = min(float(last['Open']), float(last['Close'])) - float(last['Low'])
    upper_shadow = float(last['High']) - max(float(last['Open']), float(last['Close']))
    if last_body > 0 and lower_shadow > last_body * 2 and upper_shadow < last_body * 0.5:
        match_score = 85
        return "Hammer 🟢", 75, match_score
    
    # Bearish Engulfing
    if (prev['Close'] > prev['Open'] and 
        last['Close'] < last['Open'] and 
        last_body > prev_body * 1.5):
        match_score = 90
        return "Bearish Engulfing 🔴", 20, match_score
    
    return "패턴 없음", 50, 0

candle_pattern, candle_score, pattern_match = detect_candle_pattern(hist)

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

candle_score = int(candle_score)
ma_score = int(ma_score)
cross_score = int(cross_score)

module1_score = int(candle_score * 0.3 + ma_score * 0.4 + cross_score * 0.3)

# 모듈 2
hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()
current_volume = float(hist['Volume'].iloc[-1])
avg_volume = float(hist['Volume_MA20'].iloc[-1]) if pd.notna(hist['Volume_MA20'].iloc[-1]) else 1

if avg_volume > 0:
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

module2_score = int(volume_score)

# 모듈 3
cond1 = current_price > ma120
high_20d = float(hist['Close'].tail(20).max())
cond2 = current_price >= high_20d

if len(hist) >= 2:
    pct_chg = ((current_price - float(hist['Close'].iloc[-2])) / float(hist['Close'].iloc[-2])) * 100
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

# 모듈 4
sl_methods = {
    'open': float(latest['Open']),
    'low': float(latest['Low']),
    '3pct': current_price * 0.97,
    '5pct': current_price * 0.95,
    'ma20': ma20
}

final_sl = max(sl_methods.values())
risk = ((final_sl - current_price) / current_price) * 100
target = current_price + abs(current_price - final_sl) * 2
reward = ((target - current_price) / current_price) * 100

# 최종
final_score = int(module1_score * 0.3 + module2_score * 0.3 + module3_score * 0.4)

# ========== 캔들 패턴 비교 ==========
st.subheader("🕯️ 캔들 패턴 분석 & 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📚 표준 패턴")
    if "Bullish Engulfing" in candle_pattern:
        pattern_fig = create_pattern_reference("Bullish Engulfing")
    elif "Hammer" in candle_pattern:
        pattern_fig = create_pattern_reference("Hammer")
    elif "Bearish Engulfing" in candle_pattern:
        pattern_fig = create_pattern_reference("Bearish Engulfing")
    else:
        pattern_fig = create_pattern_reference("Normal")
    
    st.plotly_chart(pattern_fig, use_container_width=True)

with col2:
    st.markdown("### 📊 실제 차트")
    actual_fig = create_actual_candle_chart(hist, num_candles=5)
    st.plotly_chart(actual_fig, use_container_width=True)

# 일치도
st.markdown("### 🎯 패턴 일치도")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("감지 패턴", candle_pattern)
with col2:
    st.metric("일치도", f"{pattern_match}%")
with col3:
    if pattern_match >= 80:
        st.success("✅ 강한 신호")
    elif pattern_match >= 60:
        st.warning("⚠️ 중간 신호")
    else:
        st.info("ℹ️ 약한 신호")

st.markdown("---")

# ========== 4대 모듈 게이지 ==========
st.subheader("📊 4대 모듈 분석")

fig_modules = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
           [{'type': 'indicator'}, {'type': 'indicator'}]],
    subplot_titles=("모듈1: 추세&패턴", "모듈2: 거래량", "모듈3: 매수신호", "모듈4: 리스크")
)

fig_modules.add_trace(go.Indicator(
    mode="gauge+number",
    value=module1_score,
    title={'text': f"{module1_score}점"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 55], 'color': "lightgray"},
            {'range': [55, 75], 'color': "yellow"},
            {'range': [75, 100], 'color': "lightgreen"}
        ]
    }
), row=1, col=1)

fig_modules.add_trace(go.Indicator(
    mode="gauge+number",
    value=module2_score,
    title={'text': f"{module2_score}점"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkorange"},
        'steps': [
            {'range': [0, 55], 'color': "lightgray"},
            {'range': [55, 75], 'color': "yellow"},
            {'range': [75, 100], 'color': "lightgreen"}
        ]
    }
), row=1, col=2)

fig_modules.add_trace(go.Indicator(
    mode="gauge+number",
    value=module3_score,
    title={'text': f"{module3_score}점"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkgreen"},
        'steps': [
            {'range': [0, 55], 'color': "lightgray"},
            {'range': [55, 75], 'color': "yellow"},
            {'range': [75, 100], 'color': "lightgreen"}
        ]
    }
), row=2, col=1)

fig_modules.add_trace(go.Indicator(
    mode="number",
    value=100,
    title={'text': "완료"},
), row=2, col=2)

fig_modules.update_layout(height=600, showlegend=False)
st.plotly_chart(fig_modules, use_container_width=True)

st.markdown("---")

# ========== 최종 평가 ==========
st.header("🏆 최종 평가")

fig_final = go.Figure(go.Indicator(
    mode="gauge+number",
    value=final_score,
    title={'text': "최종 점수"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue" if final_score >= 75 else "orange" if final_score >= 55 else "red"},
        'steps': [
            {'range': [0, 55], 'color': 'rgba(255,0,0,0.2)'},
            {'range': [55, 75], 'color': 'rgba(255,255,0,0.2)'},
            {'range': [75, 100], 'color': 'rgba(0,255,0,0.2)'}
        ]
    }
))

fig_final.update_layout(height=400)
st.plotly_chart(fig_final, use_container_width=True)

if final_score >= 75:
    st.success("### 🟢 강력 매수")
elif final_score >= 55:
    st.warning("### 🟡 신중 매수")
else:
    st.error("### 🔴 부적합")

# 기여도
st.markdown("### 📊 기여도")
contrib_df = pd.DataFrame({
    '모듈': ['모듈1', '모듈2', '모듈3'],
    '점수': [module1_score, module2_score, module3_score],
    '가중치': ['30%', '30%', '40%'],
    '기여도': [
        f"{module1_score * 0.3:.1f}",
        f"{module2_score * 0.3:.1f}",
        f"{module3_score * 0.4:.1f}"
    ]
})
st.dataframe(contrib_df, use_container_width=True)

st.markdown("---")

# ========== 차트 ==========
st.subheader("📊 가격 차트")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=hist.index,
    open=hist['Open'],
    high=hist['High'],
    low=hist['Low'],
    close=hist['Close'],
    name='캔들'
))

if pd.notna(hist['MA5']).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA5'], mode='lines', name='MA5', line=dict(color='orange', width=1)))
if pd.notna(hist['MA20']).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1)))

fig.add_hline(y=target, line_dash="dot", line_color="green", annotation_text=f"목표: {target:,.0f}")
fig.add_hline(y=final_sl, line_dash="dot", line_color="red", annotation_text=f"손절: {final_sl:,.0f}")

fig.update_layout(
    title=company_name,
    xaxis_title="날짜",
    yaxis_title="가격",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

st.caption("🔔 참고용입니다.")
