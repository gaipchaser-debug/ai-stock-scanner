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
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'scan_mode' not in st.session_state:
    st.session_state.scan_mode = None

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

def load_stock_data(ticker, max_retries=2):
    """주식 데이터 로드 (스캔용 간소화 버전)"""
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            
            for period in ["3mo", "2mo"]:
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
                time.sleep(0.5)
                continue
            
            return False, None, None, None
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            return False, None, None, None
    
    return False, None, None, None

def calculate_rsi(data, period=14):
    """RSI 계산"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stock_score(hist, current_price):
    """종목 점수 계산 (빠른 버전)"""
    try:
        if len(hist) < 20:
            return 0, {}
        
        # 이동평균
        hist['MA5'] = hist['Close'].rolling(window=5).mean()
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA60'] = hist['Close'].rolling(window=60).mean()
        hist['MA120'] = hist['Close'].rolling(window=120).mean()
        
        latest = hist.iloc[-1]
        ma5 = float(latest['MA5']) if pd.notna(latest['MA5']) else current_price
        ma20 = float(latest['MA20']) if pd.notna(latest['MA20']) else current_price
        ma60 = float(latest['MA60']) if pd.notna(latest['MA60']) else current_price
        ma120 = float(latest['MA120']) if pd.notna(latest['MA120']) else current_price
        
        # 모듈 1: 추세
        if ma5 > ma20 > ma60 > ma120:
            ma_score = 85
        elif ma5 < ma20 < ma60 < ma120:
            ma_score = 20
        else:
            ma_score = 50
        
        # 골든크로스
        if len(hist) >= 2:
            prev_ma20 = hist['MA20'].iloc[-2]
            prev_ma60 = hist['MA60'].iloc[-2]
            if pd.notna(prev_ma20) and pd.notna(prev_ma60):
                if ma20 > ma60 and prev_ma20 <= prev_ma60:
                    cross_score = 90
                elif ma20 < ma60 and prev_ma20 >= prev_ma60:
                    cross_score = 10
                else:
                    cross_score = 50
            else:
                cross_score = 50
        else:
            cross_score = 50
        
        module1_score = int(ma_score * 0.6 + cross_score * 0.4)
        
        # 모듈 2: 거래량
        hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()
        current_volume = float(hist['Volume'].iloc[-1])
        avg_volume = float(hist['Volume_MA20'].iloc[-1]) if pd.notna(hist['Volume_MA20'].iloc[-1]) else 1
        
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            if volume_ratio >= 2.0:
                volume_score = 85
            elif volume_ratio >= 1.5:
                volume_score = 60
            else:
                volume_score = 30
        else:
            volume_ratio = 0
            volume_score = 50
        
        module2_score = int(volume_score)
        
        # 모듈 3: 매수 조건
        cond1 = current_price > ma120
        high_20d = float(hist['Close'].tail(20).max())
        cond2 = current_price >= high_20d * 0.95  # 20일 고점 근처
        
        if len(hist) >= 2:
            pct_chg = ((current_price - float(hist['Close'].iloc[-2])) / float(hist['Close'].iloc[-2])) * 100
            cond3 = -5 <= pct_chg <= 15
        else:
            pct_chg = 0
            cond3 = False
        
        cond4 = volume_ratio >= 1.5
        
        satisfied = sum([cond1, cond2, cond3, cond4])
        
        if satisfied >= 3:
            module3_score = 85
        elif satisfied >= 2:
            module3_score = 60
        else:
            module3_score = 35
        
        # 모듈 4: 리스크
        sl_methods = {
            'open': float(latest['Open']),
            'low': float(latest['Low']),
            '3pct': current_price * 0.97,
            'ma20': ma20
        }
        
        final_sl = max(sl_methods.values())
        risk_pct = ((final_sl - current_price) / current_price) * 100
        target = current_price + abs(current_price - final_sl) * 2
        reward_pct = ((target - current_price) / current_price) * 100
        
        risk_reward_ratio = abs(reward_pct / risk_pct) if risk_pct != 0 else 0
        
        if risk_reward_ratio >= 2.5:
            module4_score = 95
        elif risk_reward_ratio >= 2.0:
            module4_score = 85
        elif risk_reward_ratio >= 1.5:
            module4_score = 70
        else:
            module4_score = 50
        
        # 최종 점수
        final_score = int(module1_score * 0.25 + module2_score * 0.25 + module3_score * 0.25 + module4_score * 0.25)
        
        details = {
            'module1': module1_score,
            'module2': module2_score,
            'module3': module3_score,
            'module4': module4_score,
            'volume_ratio': volume_ratio,
            'conditions': satisfied,
            'rr_ratio': risk_reward_ratio
        }
        
        return final_score, details
    
    except Exception as e:
        return 0, {}

def scan_stocks(stock_list, mode='quick'):
    """종목 스캔"""
    results = []
    
    if mode == 'quick':
        # 주요 대형주 100개만 스캔
        stocks_to_scan = stock_list.head(100)
        st.info("🔍 주요 대형주 100개 종목 스캔 중...")
    else:
        # 전체 종목 스캔
        stocks_to_scan = stock_list
        st.warning(f"🔍 전체 {len(stock_list)}개 종목 스캔 중... 약 10-20분 소요됩니다.")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(stocks_to_scan)
    
    for idx, row in stocks_to_scan.iterrows():
        code = str(row['Code'])
        name = str(row['Name'])
        market = str(row['Market'])
        
        ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
        
        # 진행률 업데이트
        progress = (idx + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"분석 중: {name} ({code}) - {idx+1}/{total}")
        
        # 데이터 로드
        is_valid, company_name, current_price, hist = load_stock_data(ticker, max_retries=1)
        
        if not is_valid:
            continue
        
        # 점수 계산
        score, details = calculate_stock_score(hist, current_price)
        
        if score >= 55:  # 55점 이상만 저장
            results.append({
                'ticker': ticker,
                'code': code,
                'name': name,
                'market': market,
                'price': current_price,
                'score': score,
                'module1': details.get('module1', 0),
                'module2': details.get('module2', 0),
                'module3': details.get('module3', 0),
                'module4': details.get('module4', 0),
                'volume_ratio': details.get('volume_ratio', 0),
                'conditions': details.get('conditions', 0),
                'rr_ratio': details.get('rr_ratio', 0)
            })
        
        # API 과부하 방지
        time.sleep(0.1)
    
    progress_bar.empty()
    status_text.empty()
    
    # 점수순 정렬
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('score', ascending=False).head(10)
    
    return results_df

def create_pattern_reference(pattern_type):
    """캔들 패턴 참조 이미지"""
    fig = go.Figure()
    
    if pattern_type == "Bullish Engulfing":
        fig.add_trace(go.Candlestick(
            x=[1], open=[105], high=[110], low=[95], close=[98],
            increasing_line_color='red', decreasing_line_color='red'
        ))
        fig.add_trace(go.Candlestick(
            x=[2], open=[97], high=[125], low=[95], close=[123],
            increasing_line_color='green', decreasing_line_color='green'
        ))
        title = "Bullish Engulfing (상승 장악형)"
        
    elif pattern_type == "Hammer":
        fig.add_trace(go.Candlestick(
            x=[1], open=[102], high=[105], low=[85], close=[103],
            increasing_line_color='green', decreasing_line_color='green'
        ))
        title = "Hammer (망치형)"
        
    elif pattern_type == "Bearish Engulfing":
        fig.add_trace(go.Candlestick(
            x=[1], open=[95], high=[105], low=[93], close=[102],
            increasing_line_color='green', decreasing_line_color='green'
        ))
        fig.add_trace(go.Candlestick(
            x=[2], open=[103], high=[108], low=[80], close=[82],
            increasing_line_color='red', decreasing_line_color='red'
        ))
        title = "Bearish Engulfing (하락 장악형)"
    
    else:
        fig.add_trace(go.Candlestick(
            x=[1], open=[100], high=[105], low=[95], close=[102],
            increasing_line_color='gray', decreasing_line_color='gray'
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

def detect_candle_pattern_advanced(hist):
    """정교한 캔들 패턴 분석 (RSI + 거래량 포함)"""
    if len(hist) < 15:
        return "데이터 부족", 50, 0, {}
    
    # RSI 계산
    hist['RSI'] = calculate_rsi(hist, period=14)
    
    last = hist.iloc[-1]
    prev = hist.iloc[-2]
    
    last_body = abs(float(last['Close']) - float(last['Open']))
    prev_body = abs(float(prev['Close']) - float(prev['Open']))
    
    # 거래량 비율
    avg_volume = hist['Volume'].tail(20).mean()
    volume_ratio = float(last['Volume']) / avg_volume if avg_volume > 0 else 1
    
    # RSI
    current_rsi = float(last['RSI']) if pd.notna(last['RSI']) else 50
    
    # 가격 위치 (20일 범위 내)
    high_20 = hist['Close'].tail(20).max()
    low_20 = hist['Close'].tail(20).min()
    price_position = (float(last['Close']) - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
    
    pattern_details = {
        'rsi': current_rsi,
        'volume_ratio': volume_ratio,
        'price_position': price_position * 100
    }
    
    # === Bullish Engulfing ===
    if (prev['Close'] < prev['Open'] and 
        last['Close'] > last['Open'] and 
        last_body > prev_body * 1.5):
        
        # 일치도 계산 (100점 만점)
        body_score = min(30, (last_body / prev_body) * 10)  # 최대 30점
        rsi_score = 20 if 30 <= current_rsi <= 50 else 10 if current_rsi < 70 else 0  # RSI 과매도~중립
        volume_score = min(25, volume_ratio * 12.5)  # 거래량 2배 = 25점
        position_score = 25 if price_position < 0.5 else 15  # 하단부 = 25점
        
        match_score = int(body_score + rsi_score + volume_score + position_score)
        
        return "Bullish Engulfing 🟢", 80, match_score, pattern_details
    
    # === Hammer ===
    lower_shadow = min(float(last['Open']), float(last['Close'])) - float(last['Low'])
    upper_shadow = float(last['High']) - max(float(last['Open']), float(last['Close']))
    
    if last_body > 0 and lower_shadow > last_body * 2 and upper_shadow < last_body * 0.5:
        
        shadow_score = min(35, (lower_shadow / last_body) * 12)  # 최대 35점
        rsi_score = 25 if current_rsi < 35 else 15 if current_rsi < 50 else 5
        volume_score = min(20, volume_ratio * 10)
        position_score = 20 if price_position < 0.3 else 10
        
        match_score = int(shadow_score + rsi_score + volume_score + position_score)
        
        return "Hammer 🟢", 75, match_score, pattern_details
    
    # === Bearish Engulfing ===
    if (prev['Close'] > prev['Open'] and 
        last['Close'] < last['Open'] and 
        last_body > prev_body * 1.5):
        
        body_score = min(30, (last_body / prev_body) * 10)
        rsi_score = 20 if 50 <= current_rsi <= 70 else 10 if current_rsi > 30 else 0
        volume_score = min(25, volume_ratio * 12.5)
        position_score = 25 if price_position > 0.5 else 15
        
        match_score = int(body_score + rsi_score + volume_score + position_score)
        
        return "Bearish Engulfing 🔴", 20, match_score, pattern_details
    
    return "패턴 없음", 50, 0, pattern_details

# ========== UI ==========
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🇰🇷 한국 주식 전체 검색 + AI 기반 4대 모듈 분석")
st.markdown("**✅ 정교한 패턴 분석 (RSI+거래량) | ✅ 리스크 관리 시각화 | ✅ 투자 적합 종목 자동 추천**")
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

# ========== 투자 적합 종목 추천 섹션 ==========
st.subheader("🎯 투자 적합 종목 추천")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 주요 대형주 빠른 추천 (100개)", type="primary", use_container_width=True):
        st.session_state.scan_mode = 'quick'
        st.session_state.scan_results = None

with col2:
    if st.button("🔍 전체 종목 검색 (약 10-20분 소요)", use_container_width=True):
        st.session_state.scan_mode = 'full'
        st.session_state.scan_results = None

# 스캔 실행
if st.session_state.scan_mode and st.session_state.scan_results is None:
    with st.spinner(f"{'빠른' if st.session_state.scan_mode == 'quick' else '전체'} 스캔 진행 중..."):
        results = scan_stocks(all_stocks_df, mode=st.session_state.scan_mode)
        st.session_state.scan_results = results
        st.rerun()

# 결과 표시
if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
    st.success(f"✅ 상위 {len(st.session_state.scan_results)}개 종목 발견!")
    
    # 결과 테이블
    display_df = st.session_state.scan_results[['name', 'code', 'market', 'price', 'score', 'module1', 'module2', 'module3', 'module4']].copy()
    display_df.columns = ['종목명', '코드', '시장', '현재가', '최종점수', 'M1', 'M2', 'M3', 'M4']
    display_df['현재가'] = display_df['현재가'].apply(lambda x: f"{x:,.0f}원")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # 상세 정보
    with st.expander("📊 상세 점수 정보"):
        for idx, row in st.session_state.scan_results.iterrows():
            st.markdown(f"""
            **{idx+1}. {row['name']}** ({row['code']}) - {row['market']}
            - 💰 현재가: {row['price']:,.0f}원
            - 🏆 최종 점수: **{row['score']}점**
            - 📊 모듈별: M1={row['module1']}점 | M2={row['module2']}점 | M3={row['module3']}점 | M4={row['module4']}점
            - 📈 거래량 배율: {row['volume_ratio']:.2f}배 | 조건 충족: {row['conditions']}/4 | R:R={row['rr_ratio']:.2f}:1
            """)
            st.markdown("---")
    
    # 차트
    st.markdown("### 📊 상위 10개 종목 점수 비교")
    
    fig_compare = go.Figure()
    
    fig_compare.add_trace(go.Bar(
        x=st.session_state.scan_results['name'],
        y=st.session_state.scan_results['score'],
        text=st.session_state.scan_results['score'],
        textposition='outside',
        marker_color=['green' if s >= 75 else 'orange' if s >= 55 else 'red' 
                      for s in st.session_state.scan_results['score']]
    ))
    
    fig_compare.add_hline(y=75, line_dash="dash", line_color="green", 
                          annotation_text="강력 매수 (75점)", annotation_position="right")
    fig_compare.add_hline(y=55, line_dash="dash", line_color="orange", 
                          annotation_text="신중 매수 (55점)", annotation_position="right")
    
    fig_compare.update_layout(
        title="투자 적합도 점수 비교",
        xaxis_title="종목명",
        yaxis_title="점수",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # 초기화 버튼
    if st.button("🔄 다시 검색"):
        st.session_state.scan_mode = None
        st.session_state.scan_results = None
        st.rerun()

elif st.session_state.scan_mode and st.session_state.scan_results is not None and st.session_state.scan_results.empty:
    st.warning("⚠️ 조건에 맞는 종목이 없습니다. 기준을 낮춰보세요.")

st.markdown("---")

# ========== 개별 종목 검색 ==========
st.subheader("🔍 개별 종목 분석")

# 검색
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "종목 검색",
        placeholder="종목명 또는 코드 (예: 삼성전자, 005930)"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🔎 검색", type="secondary", use_container_width=True)

if not query:
    st.info("👆 종목을 입력하거나 위의 추천 종목을 확인하세요")
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

# 데이터 로드 (상세 분석용)
st.markdown("---")
with st.spinner("📊 상세 데이터 로딩..."):
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

if st.button("🔄 다른 종목 검색"):
    reset_session()
    st.rerun()

st.markdown("---")

# ========== 상세 계산 ==========

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

candle_pattern, candle_score, pattern_match, pattern_details = detect_candle_pattern_advanced(hist)

# 모듈 1 계산
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
risk_pct = ((final_sl - current_price) / current_price) * 100
target = current_price + abs(current_price - final_sl) * 2
reward_pct = ((target - current_price) / current_price) * 100

# 모듈 4 점수 계산
risk_reward_ratio = abs(reward_pct / risk_pct) if risk_pct != 0 else 0

if risk_reward_ratio >= 2.5:
    module4_score = 95
elif risk_reward_ratio >= 2.0:
    module4_score = 85
elif risk_reward_ratio >= 1.5:
    module4_score = 70
elif risk_reward_ratio >= 1.0:
    module4_score = 50
else:
    module4_score = 30

module4_score = int(module4_score)

# **최종 점수**
final_score = int(module1_score * 0.25 + module2_score * 0.25 + module3_score * 0.25 + module4_score * 0.25)

# ========== 상세 분석 UI (이전과 동일) ==========
st.subheader("🕯️ 모듈 1: 추세 & 패턴 인식")

st.info("""
**📚 모듈 1이란?**  
주가의 **추세 방향**과 **캔들 패턴**을 분석하여 향후 상승/하락 가능성을 예측합니다.

- **캔들 패턴**: 단기 반전 신호 포착
- **이동평균 배열**: 장기 추세 확인
- **골든/데드크로스**: 중장기 추세 전환 신호
""")

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

st.markdown("### 🎯 패턴 일치도 분석")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("감지 패턴", candle_pattern)
with col2:
    st.metric("종합 일치도", f"{pattern_match}%")
with col3:
    st.metric("RSI", f"{pattern_details.get('rsi', 50):.1f}")
with col4:
    st.metric("거래량 배율", f"{pattern_details.get('volume_ratio', 1):.2f}배")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("캔들 패턴", f"{candle_score}점", candle_pattern)
with col2:
    st.metric("이동평균", f"{ma_score}점", ma_alignment)
with col3:
    st.metric("크로스", f"{cross_score}점", cross)

st.success(f"**📊 모듈1 종합: {module1_score}점**")

st.markdown("---")

# 모듈 2
st.subheader("📊 모듈 2: 거래량 & 공급 검증")

st.info("""
**📊 모듈 2란?**  
거래량을 분석하여 가격 움직임의 신뢰도를 검증합니다.
""")

col1, col2 = st.columns(2)
with col1:
    st.metric("거래량 배율", f"{volume_ratio:.2f}배", breakout)
with col2:
    st.metric("모듈2 점수", f"{module2_score}점")

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
    title="최근 20일 거래량 추이",
    height=400
)
st.plotly_chart(fig_volume, use_container_width=True)

st.success(f"**📊 모듈2 종합: {module2_score}점**")

st.markdown("---")

# 모듈 3
st.subheader("🎯 모듈 3: 매수 신호")

st.info("""
**🎯 모듈 3이란?**  
4가지 핵심 조건을 충족해야 강력한 매수 신호로 판단합니다.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("120일선 상회", "✅" if cond1 else "❌")
with col2:
    st.metric("20일 신고가", "✅" if cond2 else "❌")
with col3:
    st.metric("최적 상승폭", "✅" if cond3 else "❌")
with col4:
    st.metric("거래량 급증", "✅" if cond4 else "❌")

st.success(f"**📊 모듈3 종합: {module3_score}점** (충족: {satisfied}/4)")

st.markdown("---")

# 모듈 4
st.subheader("🛡️ 모듈 4: 리스크 & 수익 관리")

st.info("""
**🛡️ 모듈 4란?**  
투자 시 손실을 제한하고 수익을 극대화하기 위한 진입/청산 가격을 제시합니다.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💼 진입가", f"{current_price:,.0f}원")
with col2:
    st.metric("🛑 손절가", f"{final_sl:,.0f}원", f"{risk_pct:.2f}%")
with col3:
    st.metric("🎯 목표가", f"{target:,.0f}원", f"+{reward_pct:.2f}%")
with col4:
    st.metric("📊 모듈4 점수", f"{module4_score}점", f"{risk_reward_ratio:.2f}:1")

fig_rr = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_reward_ratio,
    title={'text': "리스크:리워드", 'font': {'size': 20}},
    number={'suffix': ":1", 'font': {'size': 40}},
    gauge={
        'axis': {'range': [0, 4]},
        'bar': {'color': "darkgreen" if risk_reward_ratio >= 2.0 else "orange"},
        'steps': [
            {'range': [0, 1.5], 'color': 'rgba(255,0,0,0.2)'},
            {'range': [1.5, 2.0], 'color': 'rgba(255,255,0,0.2)'},
            {'range': [2.0, 4], 'color': 'rgba(0,255,0,0.2)'}
        ],
        'threshold': {'line': {'color': "green", 'width': 4}, 'value': 2.0}
    }
))
fig_rr.update_layout(height=350)
st.plotly_chart(fig_rr, use_container_width=True)

st.success(f"**📊 모듈4 종합: {module4_score}점**")

st.markdown("---")

# 4대 모듈 게이지
st.subheader("📊 4대 모듈 종합 점수")

fig_modules = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
           [{'type': 'indicator'}, {'type': 'indicator'}]],
    subplot_titles=("모듈1: 추세&패턴", "모듈2: 거래량", "모듈3: 매수신호", "모듈4: 리스크")
)

for i, (score, row, col, color) in enumerate([
    (module1_score, 1, 1, "darkblue"),
    (module2_score, 1, 2, "darkorange"),
    (module3_score, 2, 1, "darkgreen"),
    (module4_score, 2, 2, "darkviolet")
]):
    fig_modules.add_trace(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': f"{score}점"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 55], 'color': "lightgray"},
                {'range': [55, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 75}
        }
    ), row=row, col=col)

fig_modules.update_layout(height=600, showlegend=False)
st.plotly_chart(fig_modules, use_container_width=True)

st.markdown("---")

# 최종 평가
st.header("🏆 최종 종합 평가")

fig_final = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=final_score,
    title={'text': "최종 점수", 'font': {'size': 24}},
    delta={'reference': 75, 'font': {'size': 20}},
    number={'font': {'size': 60}},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "darkblue" if final_score >= 75 else "orange" if final_score >= 55 else "red"},
        'steps': [
            {'range': [0, 55], 'color': 'rgba(255,0,0,0.2)'},
            {'range': [55, 75], 'color': 'rgba(255,255,0,0.2)'},
            {'range': [75, 100], 'color': 'rgba(0,255,0,0.2)'}
        ],
        'threshold': {'line': {'color': "red", 'width': 4}, 'value': 75}
    }
))

fig_final.update_layout(height=400)
st.plotly_chart(fig_final, use_container_width=True)

if final_score >= 75:
    st.success("### 🟢 강력 매수 추천")
elif final_score >= 55:
    st.warning("### 🟡 신중 매수")
else:
    st.error("### 🔴 매수 부적합")

# 기여도
st.markdown("### 📊 모듈별 기여도")
contrib_df = pd.DataFrame({
    '모듈': ['모듈1', '모듈2', '모듈3', '모듈4'],
    '점수': [module1_score, module2_score, module3_score, module4_score],
    '가중치': ['25%', '25%', '25%', '25%'],
    '기여도': [
        f"{module1_score * 0.25:.1f}점",
        f"{module2_score * 0.25:.1f}점",
        f"{module3_score * 0.25:.1f}점",
        f"{module4_score * 0.25:.1f}점"
    ]
})
st.dataframe(contrib_df, use_container_width=True)

st.markdown(f"""
**최종 점수**: ({module1_score} × 0.25) + ({module2_score} × 0.25) + ({module3_score} × 0.25) + ({module4_score} × 0.25) = **{final_score}점**
""")

st.markdown("---")

# 가격 차트
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

if pd.notna(hist['MA5']).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA5'], mode='lines', 
                             name='MA5', line=dict(color='orange', width=1)))
if pd.notna(hist['MA20']).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], mode='lines', 
                             name='MA20', line=dict(color='blue', width=2)))
if pd.notna(hist['MA60']).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], mode='lines', 
                             name='MA60', line=dict(color='green', width=2)))
if pd.notna(hist['MA120']).any():
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA120'], mode='lines', 
                             name='MA120', line=dict(color='red', width=2)))

fig.add_hline(y=target, line_dash="dot", line_color="green", 
              annotation_text=f"목표가 {target:,.0f}원")
fig.add_hline(y=final_sl, line_dash="dot", line_color="red", 
              annotation_text=f"손절가 {final_sl:,.0f}원")

fig.update_layout(
    title=f"{company_name} 기술적 분석",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info("**📌 주의**: 이 분석은 참고용이며 투자 권유가 아닙니다.")
