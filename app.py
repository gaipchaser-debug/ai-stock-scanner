import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import difflib

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
if 'radar_results' not in st.session_state:
    st.session_state.radar_results = None
if 'radar_scan_mode' not in st.session_state:
    st.session_state.radar_scan_mode = None
if 'radar_kospi_change' not in st.session_state:
    st.session_state.radar_kospi_change = None
if 'radar_kospi_current' not in st.session_state:
    st.session_state.radar_kospi_current = None
if 'div_scan_results' not in st.session_state:
    st.session_state.div_scan_results = None

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
                ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
                stock_dict[name] = ticker
                stock_dict[code] = ticker
            return stock_dict, all_stocks
        return {}, None
    except Exception as e:
        st.error(f"종목 리스트 로드 실패: {str(e)}")
        return {}, None

def search_stock(query, stock_dict, all_stocks_df):
    query_raw = str(query).strip()
    query_lower = query_raw.lower()
    if query_raw.isdigit():
        code_padded = query_raw.zfill(6)
        if code_padded in stock_dict: return stock_dict[code_padded], None, 'exact'
        if query_raw in stock_dict: return stock_dict[query_raw], None, 'exact'
    q_clean = query_lower.replace(" ", "").replace("(", "").replace(")", "")
    if query_lower in stock_dict: return stock_dict[query_lower], None, 'exact'
    if q_clean in stock_dict: return stock_dict[q_clean], None, 'exact'
    if all_stocks_df is not None:
        names_lower = all_stocks_df['Name'].str.lower().str.strip()
        mask_partial = names_lower.str.contains(query_lower, na=False, regex=False)
        partial_matches = all_stocks_df[mask_partial]
        if len(partial_matches) == 1:
            code = str(partial_matches.iloc[0]['Code'])
            market = str(partial_matches.iloc[0]['Market'])
            return f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ", None, 'exact'
        elif len(partial_matches) > 1:
            return None, partial_matches.head(10).reset_index(drop=True), 'partial'
        for cutoff in [0.55, 0.40, 0.30]:
            close_names = difflib.get_close_matches(query_lower, names_lower.tolist(), n=10, cutoff=cutoff)
            if close_names: break
        if close_names:
            sim_mask = names_lower.isin(close_names)
            similar_df = all_stocks_df[sim_mask].copy()
            similar_df['_sim'] = similar_df['Name'].str.lower().apply(lambda n: difflib.SequenceMatcher(None, query_lower, n).ratio())
            return None, similar_df.sort_values('_sim', ascending=False).drop(columns=['_sim']).head(10).reset_index(drop=True), 'similar'
    return None, None, 'notfound' 

def load_stock_data(ticker, max_retries=2):
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            for period in ["3mo", "2mo"]:
                hist = stock.history(period=period)
                if not hist.empty and len(hist) >= 20:
                    return True, stock.info.get('longName', ticker), float(hist['Close'].iloc[-1]), hist
        except: time.sleep(0.5)
    return False, None, None, None

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + (gain / loss)))

def detect_candle_pattern_advanced(hist):
    if len(hist) < 15: return "데이터 부족", 50, 0, {}
    hist['RSI'] = calculate_rsi(hist, period=14)
    c = hist['Close'].iloc[-5:].values
    o = hist['Open'].iloc[-5:].values
    h = hist['High'].iloc[-5:].values
    l = hist['Low'].iloc[-5:].values
    v = hist['Volume'].iloc[-5:].values
    
    last_body, prev_body = abs(c[-1] - o[-1]), abs(c[-2] - o[-2])
    avg_volume = hist['Volume'].tail(20).mean()
    volume_ratio = float(v[-1]) / avg_volume if avg_volume > 0 else 1
    current_rsi = float(hist['RSI'].iloc[-1]) if pd.notna(hist['RSI'].iloc[-1]) else 50
    high_20, low_20 = hist['Close'].tail(20).max(), hist['Close'].tail(20).min()
    price_pos = (c[-1] - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
    details = {'rsi': current_rsi, 'volume_ratio': volume_ratio, 'price_position': price_pos * 100}

    if (c[-3]>o[-3] and c[-2]>o[-2] and c[-1]>o[-1] and c[-3]<c[-2]<c[-1] and o[-3]<o[-2]<c[-3] and o[-2]<o[-1]<c[-2]):
        return "적삼병 (사카타5법) 🚀", 95, int(min(100, (volume_ratio*20) + (100-current_rsi)*0.5 + 40)), details
    if (c[-3]<o[-3] and c[-2]<o[-2] and c[-1]<o[-1] and c[-3]>c[-2]>c[-1] and c[-3]<o[-2]<o[-3] and c[-2]<o[-1]<o[-2]):
        return "흑삼병 (사카타5법) ⚠️", 10, int(min(100, current_rsi*0.8 + 40)), details
    if (c[-5]>o[-5] and c[-4]<o[-4] and c[-3]<o[-3] and c[-2]<o[-2] and c[-5]>max(c[-4],c[-3],c[-2]) and o[-5]<min(c[-4],c[-3],c[-2]) and c[-1]>o[-1] and c[-1]>c[-5]):
        return "상승 삼법 (사카타5법) 📈", 90, 85, details
    if (c[-2]<o[-2] and c[-1]>o[-1] and last_body>prev_body*1.5 and c[-1]>o[-2] and o[-1]<c[-2]):
        return "상승 장악형 (Bullish Engulfing) 🟢", 80, int(min(30, (last_body/prev_body)*10) + (20 if 30<=current_rsi<=50 else 0) + min(25, volume_ratio*12.5)), details
    
    lower_shadow, upper_shadow = min(o[-1], c[-1]) - l[-1], h[-1] - max(o[-1], c[-1])
    if last_body > 0 and lower_shadow > last_body*2 and upper_shadow < last_body*0.5:
        return "해머형 (망치형) 🟢", 75, int(min(35, (lower_shadow/last_body)*12) + (25 if current_rsi<35 else 5) + min(20, volume_ratio*10)), details
    return "일반 캔들 (패턴 없음)", 50, 0, details

def calculate_stock_score(hist, current_price, vs_kospi=None, verdict=None):
    try:
        if len(hist) < 20: return 0, {}
        hist = hist.copy()
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        hist['MA60'] = hist['Close'].rolling(60).mean()
        hist['MA120'] = hist['Close'].rolling(120).mean()
        
        latest = hist.iloc[-1]
        ma5 = float(latest['MA5']) if pd.notna(latest['MA5']) else current_price
        ma20 = float(latest['MA20']) if pd.notna(latest['MA20']) else current_price
        ma60 = float(latest['MA60']) if pd.notna(latest['MA60']) else current_price
        ma120 = float(latest['MA120']) if pd.notna(latest['MA120']) else current_price

        pattern_name, candle_score, _, _ = detect_candle_pattern_advanced(hist)
        align_count = sum([ma5>ma20, ma20>ma60, ma60>ma120, ma5>ma60, ma20>ma120])
        ma_score = {5:92, 4:78, 3:62, 2:46, 1:32, 0:15}[align_count]
        
        # 오류가 났던 부분을 여러 줄로 안전하게 수정
        cross_score = 50
        if len(hist) >= 5:
            prev_ma20 = hist['MA20'].iloc[-2] if pd.notna(hist['MA20'].iloc[-2]) else ma20
            prev_ma60 = hist['MA60'].iloc[-2] if pd.notna(hist['MA60'].iloc[-2]) else ma60
            if ma20 > ma60 and prev_ma20 <= prev_ma60:
                cross_score = 95
            elif ma20 < ma60 and prev_ma20 >= prev_ma60:
                cross_score = 8
                
        module1_score = int(ma_score * 0.4 + cross_score * 0.3 + candle_score * 0.3)

        volume_ratio = float(hist['Volume'].iloc[-1]) / float(hist['Volume'].rolling(20).mean().iloc[-1]) if float(hist['Volume'].rolling(20).mean().iloc[-1])>0 else 1
        module2_score = int(min(95, max(20, volume_ratio * 38 + 10)))

        conds = [current_price > ma120, current_price >= float(hist['Close'].tail(20).max()) * 0.95, -2 <= ((current_price - float(hist['Close'].iloc[-2])) / float(hist['Close'].iloc[-2])) * 100 <= 15 if len(hist)>=2 else False, volume_ratio >= 1.5]
        module3_score = {4:85, 3:65, 2:45, 1:30, 0:15}[sum(conds)]

        final_sl = max(float(latest['Open']), float(latest['Low']), current_price * 0.97, ma20)
        risk_pct = ((final_sl - current_price) / current_price) * 100
        reward_pct = (((current_price + abs(current_price - final_sl) * 2) - current_price) / current_price) * 100
        module4_score = min(95, max(20, int(abs(reward_pct / risk_pct) * 28 + 18) if risk_pct != 0 else 95))

        module5_score = None
        if vs_kospi is not None:
            module5_score = 100 if verdict == "⭐ 역주행" else 85 if vs_kospi > 2.0 else 65 if vs_kospi > 0 else 45 if vs_kospi > -1.0 else 20

        final_score = int(module1_score*0.2 + module2_score*0.2 + module3_score*0.2 + module4_score*0.2 + module5_score*0.2) if module5_score else int(module1_score*0.25 + module2_score*0.25 + module3_score*0.25 + module4_score*0.25)
        return final_score, {'module1': module1_score, 'module2': module2_score, 'module3': module3_score, 'module4': module4_score, 'module5': module5_score, 'pattern_name': pattern_name}
    except: return 0, {}

def scan_stocks(stock_list, mode='quick'):
    results, radar_lookup = [], {r['ticker']: {'vs_kospi': r.get('vs_kospi'), 'verdict': r.get('verdict')} for _, r in st.session_state.get('radar_results', pd.DataFrame()).iterrows()} if st.session_state.get('radar_results') is not None else {}
    stocks_to_scan = stock_list.head(100) if mode == 'quick' else stock_list
    progress_bar, status_text = st.progress(0), st.empty()
    for idx, (df_idx, row) in enumerate(stocks_to_scan.iterrows()):
        ticker = f"{row['Code']}.KS" if row['Market'] == 'KOSPI' else f"{row['Code']}.KQ"
        progress_bar.progress((idx + 1) / len(stocks_to_scan))
        status_text.text(f"분석 중: {row['Name']}")
        is_valid, _, price, hist = load_stock_data(ticker, 1)
        if not is_valid: continue
        score, details = calculate_stock_score(hist, price, radar_lookup.get(ticker, {}).get('vs_kospi'), radar_lookup.get(ticker, {}).get('verdict'))
        if score >= 50: results.append({'ticker': ticker, 'code': row['Code'], 'name': row['Name'], 'score': score, 'pattern': details.get('pattern_name', ''), **details})
    progress_bar.empty(); status_text.empty()
    return pd.DataFrame(results).sort_values('score', ascending=False).head(10) if results else pd.DataFrame()

# ===== 시장 레이더 함수들 =====
TOP50_FALLBACK = [
    ("005930", "삼성전자", "KOSPI"), ("000660", "SK하이닉스", "KOSPI"), ("207940", "삼성바이오로직스", "KOSPI"), 
    ("005380", "현대차", "KOSPI"), ("105560", "KB금융", "KOSPI"), ("055550", "신한지주", "KOSPI"), 
    ("035420", "NAVER", "KOSPI"), ("035720", "카카오", "KOSPI"), ("068270", "셀트리온", "KOSPI")
]
@st.cache_data(ttl=300)
def get_kospi_status():
    try:
        hist = yf.Ticker("^KS11").history(period="5d")
        if len(hist)>=2: return float(hist['Close'].iloc[-1]), (float(hist['Close'].iloc[-1])-float(hist['Close'].iloc[-2]))/float(hist['Close'].iloc[-2])*100, float(hist['Close'].iloc[-1])-float(hist['Close'].iloc[-2]), hist
    except: return None, 0, 0, None

def run_radar_scan(stock_list):
    k_cur, k_chg, _, _ = get_kospi_status()
    results = []
    for c, n, m in stock_list:
        ticker = f"{c}.KS" if m == "KOSPI" else f"{c}.KQ"
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist)>=2:
                price = float(hist['Close'].iloc[-1])
                chg = (price - float(hist['Close'].iloc[-2])) / float(hist['Close'].iloc[-2]) * 100
                vs_k = chg - k_chg
                verdict = "⭐ 역주행" if chg>0 and k_chg<0 else "✅ 강한 방어" if vs_k>1.0 else "🔴 이탈"
                results.append({"ticker": ticker, "name": n, "price": price, "vs_kospi": round(vs_k, 2), "verdict": verdict})
        except: continue
    return pd.DataFrame(results).sort_values("vs_kospi", ascending=False).reset_index(drop=True), k_cur, k_chg, None

# =========================================================================
# ★ TAB 4: 배당주 스캔 및 백테스팅 엔진 함수 ★
# =========================================================================
DIVIDEND_CANDIDATES = [
    ("105560", "KB금융", "KOSPI"), ("055550", "신한지주", "KOSPI"), ("086790", "하나금융지주", "KOSPI"), 
    ("316140", "우리금융지주", "KOSPI"), ("032640", "LG유플러스", "KOSPI"), ("017670", "SK텔레콤", "KOSPI"), 
    ("030200", "KT", "KOSPI"), ("033780", "KT&G", "KOSPI"), ("024110", "기업은행", "KOSPI"), 
    ("005930", "삼성전자", "KOSPI"), ("090430", "아모레퍼시픽", "KOSPI"), ("005380", "현대차", "KOSPI")
]

def scan_all_dividend_stocks_for_today(investment_amount):
    """오늘 날짜를 기준으로 주요 배당주의 과거 흐름을 시뮬레이션"""
    results = []
    today = datetime.now()
    target_sell_month, target_sell_day = 12, 26 # 통상적인 배당락일 전 매도
    
    prog, stat = st.progress(0), st.empty()
    for idx, (code, name, market) in enumerate(DIVIDEND_CANDIDATES):
        prog.progress((idx + 1) / len(DIVIDEND_CANDIDATES))
        stat.text(f"과거 데이터 시뮬레이션 중: {name}")
        ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
        
        try:
            hist = yf.Ticker(ticker).history(period="10y")
            if hist.empty or len(hist) < 252: continue
            
            hist['Month'] = hist.index.month
            hist['Day'] = hist.index.day
            hist['Year'] = hist.index.year
            
            today_month, today_day = today.month, today.day
            buy_prices, sell_prices = [], []
            
            for year in hist['Year'].unique():
                if year == today.year: continue # 올해 연말 데이터 제외
                
                # 해당 연도의 오늘 기준 매수 시점 찾기
                buy_mask = (hist['Year'] == year) & (hist['Month'] == today_month) & (hist['Day'] >= today_day)
                sell_mask = (hist['Year'] == year) & (hist['Month'] == target_sell_month) & (hist['Day'] <= target_sell_day)
                
                if not hist[buy_mask].empty and not hist[sell_mask].empty:
                    b_price = hist[buy_mask].iloc[0]['Close']
                    s_price = hist[sell_mask].iloc[-1]['Close']
                    buy_prices.append(b_price)
                    sell_prices.append(s_price)
            
            if not buy_prices: continue
            
            returns = [(s - b) / b * 100 for s, b in zip(sell_prices, buy_prices)]
            avg_return = np.mean(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
            expected_profit = investment_amount * (avg_return / 100)
            
            if avg_return >= 3.0 and win_rate >= 70: action = "🔥 적극 매수 (Buy Now)"
            elif avg_return > 0: action = "🟡 관망/분할 매수 (Wait)"
            else: action = "🔴 매수 부적합 (Too Late)"
            
            results.append({
                'ticker': ticker, 'name': name,
                'current_price': hist.iloc[-1]['Close'],
                'avg_return': avg_return,
                'win_rate': win_rate,
                'expected_profit': expected_profit,
                'action': action
            })
        except: continue
        
    prog.empty(); stat.empty()
    if results:
        df = pd.DataFrame(results).sort_values(by='avg_return', ascending=False).reset_index(drop=True)
        return df
    return pd.DataFrame()


# ========== 메인 UI ==========
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🇰🇷 사카타 5법 캔들 분석 + 4대 모듈 + 배당 백테스팅")

if st.session_state.stock_list is None:
    with st.spinner("📡 전체 종목 리스트 로딩 중..."):
        stock_dict, all_stocks_df = load_all_korean_stocks()
        if stock_dict: st.session_state.stock_list = (stock_dict, all_stocks_df)
        else: st.stop()

stock_dict, all_stocks_df = st.session_state.stock_list

tab1, tab2, tab3, tab4 = st.tabs(["📡 시장 레이더", "🎯 투자 적합 종목 추천", "🔍 개별 종목 분석", "🎁 배당주 투자 가이드"])

# ----- TAB 1: 시장 레이더 -----
with tab1:
    st.subheader("📡 오늘의 시장 레이더")
    if st.button("레이더 스캔 실행 (시총 50위)", type="primary"):
        with st.spinner("스캔 중..."):
            df, kc, kchg, _ = run_radar_scan(TOP50_FALLBACK)
            st.session_state.radar_results, st.session_state.radar_kospi_change = df, kchg
            st.rerun()
    if st.session_state.radar_results is not None: st.dataframe(st.session_state.radar_results)

# ----- TAB 2: 투자 적합 종목 추천 -----
with tab2:
    st.subheader("🎯 투자 적합 종목 추천")
    if st.button("🚀 전체 종목 스캔 (사카타 5법 모듈 반영)", type="primary"):
        with st.spinner("스캔 중..."):
            st.session_state.scan_results = scan_stocks(all_stocks_df, mode='quick')
            st.rerun()
    if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
        df_show = st.session_state.scan_results[['name', 'score', 'pattern', 'module1', 'module2', 'module3']]
        st.dataframe(df_show, use_container_width=True)

# ----- TAB 3: 개별 종목 분석 -----
with tab3:
    st.subheader("🔍 개별 종목 분석")
    query = st.text_input("종목명 또는 코드 입력", placeholder="예: 삼성전자, 005930")
    if st.button("🔎 검색"):
        ticker, _, _ = search_stock(query, stock_dict, all_stocks_df)
        if ticker:
            st.session_state.current_ticker = ticker
            st.rerun()
    if st.session_state.current_ticker:
        is_valid, name, price, hist = load_stock_data(st.session_state.current_ticker)
        if is_valid:
            pattern_name, candle_score, _, _ = detect_candle_pattern_advanced(hist)
            st.markdown(f"### 📊 분석 결과: {name} ({st.session_state.current_ticker})")
            st.metric("감지된 캔들 패턴 (사카타 5법)", pattern_name, f"패턴 점수: {candle_score}점")
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
            fig.update_layout(height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# ----- TAB 4: 배당주 투자 가이드 (일괄 스캐너) -----
with tab4:
    st.subheader("🎁 배당주 전체 스캐너 (10년 데이터 백테스팅)")
    st.markdown("""
    > 💡 **'오늘'을 기준으로 계산합니다.** > 국내 주요 고배당주들의 과거 10년 주가 흐름을 시뮬레이션하여, **오늘 날짜에 매수**해서 배당락일 직전(12월 26일경)에 매도했을 때의 평균 성과를 분석합니다.
    """)
    
    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        investment_krw = st.number_input("💰 예상 투자금액 (원)", min_value=1000000, max_value=1000000000, value=10000000, step=1000000)
    with col_input2:
        st.write("") # 버튼 위치 조정
        st.write("")
        run_div_scan = st.button("🚀 국내 주요 배당주 오늘 기준 타이밍 분석", type="primary", use_container_width=True)
        
    if run_div_scan:
        with st.spinner("⏳ 주요 배당주의 과거 10년 치 주가 궤적을 연산하고 있습니다..."):
            div_df = scan_all_dividend_stocks_for_today(investment_krw)
            st.session_state.div_scan_results = div_df
            
    if st.session_state.div_scan_results is not None and not st.session_state.div_scan_results.empty:
        div_df = st.session_state.div_scan_results
        
        st.markdown("---")
        st.markdown("### 🏆 '오늘' 기준 배당주 매매 전략 순위")
        st.caption(f"기준일: {datetime.now().strftime('%Y년 %m월 %d일')} 매수 → 12월 26일 매도 기준 과거 10년 통계")
        
        # DataFrame 표시용 포맷팅
        display_div_df = div_df.copy()
        display_div_df['current_price'] = display_div_df['current_price'].apply(lambda x: f"{x:,.0f}원")
        display_div_df['avg_return'] = display_div_df['avg_return'].apply(lambda x: f"{x:+.2f}%")
        display_div_df['win_rate'] = display_div_df['win_rate'].apply(lambda x: f"{x:.0f}%")
        display_div_df['expected_profit'] = display_div_df['expected_profit'].apply(lambda x: f"{x:,.0f}원")
        
        display_div_df.columns = ['종목코드', '종목명', '현재가', '과거 10년 평균수익률(%)', '과거 승률', f'예상수익금 ({investment_krw:,}원 기준)', '현재 타이밍 진단']
        
        # 데이터프레임 렌더링 (판정 결과에 따라 색상 강조)
        def color_action(val):
            if 'Buy Now' in str(val): return 'color: #e74c3c; font-weight: bold'
            elif 'Wait' in str(val): return 'color: #f39c12'
            else: return 'color: #7f8c8d'

        st.dataframe(display_div_df.style.applymap(color_action, subset=['현재 타이밍 진단']), use_container_width=True)
        
        st.info("💡 **전략 가이드**: 과거 10년 기준, 승률 70% 이상 및 평균 수익률 3% 이상인 종목을 'Buy Now(적극 매수)'로 추천합니다. 배당락일 약 2일 전에 매도하여 안전하게 시세차익만 챙기는 전략입니다.")
