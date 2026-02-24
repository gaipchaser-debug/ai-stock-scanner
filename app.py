import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
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
    """한국 주식 전체 리스트 로드 (FinanceDataReader)"""
    try:
        if FDR_AVAILABLE:
            # 코스피
            kospi = fdr.StockListing('KOSPI')
            kospi['Market'] = 'KOSPI'
            
            # 코스닥
            kosdaq = fdr.StockListing('KOSDAQ')
            kosdaq['Market'] = 'KOSDAQ'
            
            # 병합
            all_stocks = pd.concat([kospi, kosdaq], ignore_index=True)
            
            # 종목명 → 코드 딕셔너리
            stock_dict = {}
            for _, row in all_stocks.iterrows():
                code = row['Code']
                name = row['Name'].lower().strip()
                market = row['Market']
                
                # yfinance 티커 형식
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
    
    # 1. 정확한 매치
    if query in stock_dict:
        return stock_dict[query], None
    
    # 2. 부분 매치
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
st.markdown("**✅ 코스피 + 코스닥 전체 | ✅ 2,500+ 종목**")
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
st.subheader(f"💰 {current_price:,.0f}원")

if len(hist) >= 2:
    prev = hist['Close'].iloc[-2]
    change = current_price - prev
    pct = (change / prev) * 100
    
    if change > 0:
        st.markdown(f"📈 전일대비: +{change:,.0f}원 (+{pct:.2f}%)")
    elif change < 0:
        st.markdown(f"📉 전일대비: {change:,.0f}원 ({pct:.2f}%)")

if st.button("🔄 다시 검색"):
    reset_session()
    st.rerun()

st.markdown("---")

# ========== 모듈 1 ==========
st.subheader("📈 모듈 1: 추세 & 패턴")

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
    st.metric("캔들", candle_pattern, f"{candle_score}점")
with col2:
    st.metric("이동평균", ma_alignment, f"{ma_score}점")
with col3:
    st.metric("크로스", cross, f"{cross_score}점")

st.success(f"**모듈1: {module1_score}점**")
st.markdown("---")

# ========== 모듈 2 ==========
st.subheader("📊 모듈 2: 거래량")

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

module2_score = volume_score

col1, col2 = st.columns(2)
with col1:
    st.metric("거래량", f"{volume_ratio:.2f}배", breakout)
with col2:
    st.metric("점수", f"{volume_score}점")

st.success(f"**모듈2: {module2_score}점**")
st.markdown("---")

# ========== 모듈 3 ==========
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
    st.metric("120일선", "✅" if cond1 else "❌")
with col2:
    st.metric("20일고", "✅" if cond2 else "❌")
with col3:
    st.metric("상승", "✅" if cond3 else "❌")
with col4:
    st.metric("거래량", "✅" if cond4 else "❌")

if satisfied == 4:
    module3_score = 95
    st.success("🎉 강력 매수")
elif satisfied == 3:
    module3_score = 70
    st.warning("⚠️ 신중 매수")
else:
    module3_score = 40
    st.error("❌ 부적합")

st.success(f"**모듈3: {module3_score}점**")
st.markdown("---")

# ========== 모듈 4 ==========
st.subheader("🛡️ 모듈 4: 리스크")

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

st.success("**모듈4: 완료**")
st.markdown("---")

# ========== 최종 ==========
st.header("🏆 최종 평가")

final_score = int(module1_score * 0.3 + module2_score * 0.3 + module3_score * 0.4)

if final_score >= 75:
    rec = "🟢 강력 매수"
elif final_score >= 55:
    rec = "🟡 신중 매수"
else:
    rec = "🔴 부적합"

st.markdown(f"### {rec}")
st.markdown(f"**{final_score}점**")
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
    title=company_name,
    xaxis_title="날짜",
    yaxis_title="가격",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

st.caption("🔔 본 분석은 참고용입니다.")
