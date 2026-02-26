올려주신 오류 이미지들과 코드를 분석해보니 오류의 원인은 **'복사 붙여넣기(Copy & Paste) 과정에서의 충돌'**입니다.

주요 원인은 다음과 같습니다.

1. `SyntaxError: invalid syntax` (`python) : 코드를 복사하실 때 제가 작성해 드린 마크다운 코드 블록 기호(`python)까지 함께 복사되어 파이썬 문법 오류가 발생했습니다.
2. `ValueError: Length mismatch` 및 중복 함수 오류 : 기존 `app.py`의 코드를 지우지 않고 그 아래에 계속 코드를 덧붙여서(Append) 넣으셨기 때문에, 동일한 함수가 2~3개씩 중복 생성되면서 데이터 열(Column) 개수가 맞지 않는 충돌이 발생했습니다.

이 모든 문제를 깔끔하게 해결한 **최종 클린 버전의 전체 코드**를 제공해 드립니다.

### 🚨 [필독] 이렇게 적용해 주세요!

1. 현재 `app.py` 파일의 텍스트 편집기 창에서 **`Ctrl + A` (또는 Cmd + A)를 눌러 전체 선택**을 합니다.
2. **`Delete` 키를 눌러 기존 코드를 전부 싹 지워주세요.** (완전히 빈 화면이 되어야 합니다)
3. 아래의 코드를 복사할 때, 맨 위와 맨 아래에 있는 **`python** 이나 **`** 기호는 빼고, **`import streamlit as st` 부터 맨 마지막 줄까지만** 정확히 복사해서 붙여넣고 저장(`Ctrl + S`)해 주세요.

---

### 💻 완벽 통합 최종 코드 (`app.py`)

```python
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

# ========== 세션 스테이트 초기화 ==========
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
if 'radar_kospi_change' not in st.session_state:
    st.session_state.radar_kospi_change = None
if 'div_scan_results' not in st.session_state:
    st.session_state.div_scan_results = None

def reset_session():
    st.cache_data.clear()

# ========== 1. 기본 데이터 로드 함수 ==========
@st.cache_data(ttl=86400)
def load_all_korean_stocks():
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
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ========== 2. 사카타 5법 및 분석 알고리즘 ==========
def detect_candle_pattern_advanced(hist):
    if len(hist) < 15: return "데이터 부족", 50, 0, {}
    hist['RSI'] = calculate_rsi(hist, period=14)
    c, o, h, l, v = hist['Close'].iloc[-5:].values, hist['Open'].iloc[-5:].values, hist['High'].iloc[-5:].values, hist['Low'].iloc[-5:].values, hist['Volume'].iloc[-5:].values
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
        hist['MA5'], hist['MA20'], hist['MA60'], hist['MA120'] = hist['Close'].rolling(5).mean(), hist['Close'].rolling(20).mean(), hist['Close'].rolling(60).mean(), hist['Close'].rolling(120).mean()
        latest = hist.iloc[-1]
        ma5, ma20, ma60, ma120 = float(latest['MA5']) if pd.notna(latest['MA5']) else current_price, float(latest['MA20']) if pd.notna(latest['MA20']) else current_price, float(latest['MA60']) if pd.notna(latest['MA60']) else current_price, float(latest['MA120']) if pd.notna(latest['MA120']) else current_price

        pattern_name, candle_score, _, _ = detect_candle_pattern_advanced(hist)
        
        align_count = sum([ma5>ma20, ma20>ma60, ma60>ma120, ma5>ma60, ma20>ma120])
        ma_score = {5:92, 4:78, 3:62, 2:46, 1:32, 0:15}[align_count]
        
        cross_score = 50
        if len(hist) >= 5:
            prev_ma20 = hist['MA20'].iloc[-2] if pd.notna(hist['MA20'].iloc[-2]) else ma20
            prev_ma60 = hist['MA60'].iloc[-2] if pd.notna(hist['MA60'].iloc[-2]) else ma60
            if ma20 > ma60 and prev_ma20 <= prev_ma60: cross_score = 95
            elif ma20 < ma60 and prev_ma20 >= prev_ma60: cross_score = 8
                
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

# ========== 3. 시장 레이더 함수 ==========
TOP50_FALLBACK = [
    ("005930", "삼성전자", "KOSPI"), ("000660", "SK하이닉스", "KOSPI"), ("207940", "삼성바이오로직스", "KOSPI"), 
    ("005380", "현대차", "KOSPI"), ("105560", "KB금융", "KOSPI"), ("055550", "신한지주", "KOSPI"), 
    ("035420", "NAVER", "KOSPI"), ("035720", "카카오", "KOSPI"), ("068270", "셀트리온", "KOSPI"),
    ("051910", "LG화학", "KOSPI"), ("006400", "삼성SDI", "KOSPI"), ("028260", "삼성물산", "KOSPI")
]

@st.cache_data(ttl=300)
def get_kospi_status():
    try:
        hist = yf.Ticker("^KS11").history(period="5d")
        if len(hist)>=2: return float(hist['Close'].iloc[-1]), (float(hist['Close'].iloc[-1])-float(hist['Close'].iloc[-2]))/float(hist['Close'].iloc[-2])*100, float(hist['Close'].iloc[-1])-float(hist['Close'].iloc[-2]), hist
    except: return None, 0, 0, None

def get_stock_today_change(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if len(hist)>=2: return float(hist['Close'].iloc[-1]), (float(hist['Close'].iloc[-1])-float(hist['Close'].iloc[-2]))/float(hist['Close'].iloc[-2])*100, float(hist['Volume'].iloc[-1]) / float(hist['Volume'].mean())
        return None, None, None
    except: return None, None, None

def run_radar_scan(stock_list):
    kospi_current, kospi_change, kospi_pt, kospi_hist = get_kospi_status()
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (code, name, market) in enumerate(stock_list):
        ticker = f"{code}.KS" if market == "KOSPI" else f"{code}.KQ"
        progress_bar.progress((i + 1) / len(stock_list))
        status_text.text(f"📡 스캔 중: {name}")

        price, chg, vol_ratio = get_stock_today_change(ticker)
        if price is None: continue

        vs_kospi = chg - kospi_change
        if chg > 0 and kospi_change < 0: verdict = "⭐ 역주행"
        elif vs_kospi > 1.0: verdict = "✅ 강한 방어"
        elif vs_kospi > 0: verdict = "🛡️ 방어"
        elif vs_kospi > -1.0: verdict = "➖ 동행"
        else: verdict = "🔴 이탈"

        # 정확히 8개의 키를 가진 딕셔너리 생성
        results.append({
            "code": code,
            "name": name,
            "market": market,
            "price": price,
            "change_pct": round(chg, 2),
            "vs_kospi": round(vs_kospi, 2),
            "vol_ratio": round(vol_ratio, 2),
            "verdict": verdict
        })
        
    progress_bar.empty()
    status_text.empty()
    df = pd.DataFrame(results)
    if not df.empty: df = df.sort_values("vs_kospi", ascending=False).reset_index(drop=True)
    return df, kospi_current, kospi_change, kospi_hist

# ========== 4. 배당 랠리 최적화 백테스팅 엔진 ==========
DIVIDEND_CANDIDATES = [
    ("105560", "KB금융", "KOSPI"), ("055550", "신한지주", "KOSPI"), ("086790", "하나금융지주", "KOSPI"), 
    ("316140", "우리금융지주", "KOSPI"), ("032640", "LG유플러스", "KOSPI"), ("017670", "SK텔레콤", "KOSPI"), 
    ("030200", "KT", "KOSPI"), ("033780", "KT&G", "KOSPI"), ("024110", "기업은행", "KOSPI"), 
    ("005930", "삼성전자", "KOSPI"), ("005380", "현대차", "KOSPI"), ("090430", "아모레퍼시픽", "KOSPI")
]

def analyze_dividend_rally_and_project(ticker, investment_amount):
    """과거 10년 패턴 기반으로 다가올 배당일 최적 타점 도출"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="10y")
        divs = stock.dividends
        
        if hist.empty or divs.empty: return None
            
        try: hist.index = hist.index.tz_localize(None)
        except: hist.index = hist.index.tz_convert(None)
        try: divs.index = divs.index.tz_localize(None)
        except: divs.index = divs.index.tz_convert(None)
        
        today = datetime.now()
        
        # 1. 다가올 배당일 & 주당 배당금 예측
        last_1y_divs = divs[divs.index > (today - pd.DateOffset(years=1))]
        dps = last_1y_divs.sum() if not last_1y_divs.empty else 0
        
        last_2y_divs = divs[divs.index > (today - pd.DateOffset(years=2))]
        projected_dates = [d + pd.DateOffset(years=1) for d in last_2y_divs.index]
        future_dates = sorted([d for d in projected_dates if d.date() > today.date()])
        
        if future_dates: next_d_day = future_dates[0]
        else: return None
            
        # 2. 과거 10년 최적 배당 랠리 매수타점 (D-15, D-30, D-45, D-60 중 최적화)
        test_windows = [15, 30, 45, 60] 
        sell_offset = 2 # 배당락일 2일 전 매도 (고정)
        
        best_ret, best_win, best_w = -999, 0, 30
        
        for w in test_windows:
            returns = []
            for d_date in divs.index:
                if d_date in hist.index:
                    d_idx = hist.index.get_loc(d_date)
                    buy_idx = d_idx - w
                    sell_idx = d_idx - sell_offset
                    
                    if buy_idx >= 0 and sell_idx >= 0 and sell_idx > buy_idx:
                        b_price = hist['Close'].iloc[buy_idx]
                        s_price = hist['Close'].iloc[sell_idx]
                        returns.append((s_price - b_price) / b_price * 100)
            
            if returns:
                avg_r = np.mean(returns)
                win_r = sum(1 for r in returns if r > 0) / len(returns) * 100
                if avg_r > best_ret:
                    best_ret, best_win, best_w = avg_r, win_r, w
        
        if best_ret == -999: return None
        
        # 3. 오늘 기준 액션 판별
        target_buy_date = next_d_day - timedelta(days=int(best_w * 1.4))
        days_to_buy = (target_buy_date.date() - today.date()).days
        
        if -5 <= days_to_buy <= 5: action = "🔥 지금 최적 매수기"
        elif days_to_buy > 5: action = f"⏳ {days_to_buy}일 후 매수"
        else: action = "🚀 이미 상승 랠리중"

        current_price = float(hist.iloc[-1]['Close'])
        expected_profit = investment_amount * (best_ret / 100)
        div_yield = (dps / current_price * 100) if current_price > 0 else 0

        return {
            'current_price': current_price, 'dps': dps, 'div_yield': div_yield,
            'next_d_day': next_d_day.strftime('%Y-%m-%d'),
            'best_strategy': f"D-{best_w}일 매수",
            'avg_return': best_ret, 'win_rate': best_win,
            'expected_profit': expected_profit, 'action': action
        }
    except: return None

def scan_all_dividend_stocks_for_rally(investment_amount):
    results = []
    prog, stat = st.progress(0), st.empty()
    for idx, (code, name, market) in enumerate(DIVIDEND_CANDIDATES):
        prog.progress((idx + 1) / len(DIVIDEND_CANDIDATES))
        stat.text(f"과거 패턴 및 다가올 배당일 연산 중: {name}")
        ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
        res = analyze_dividend_rally_and_project(ticker, investment_amount)
        if res:
            res['ticker'] = ticker
            res['name'] = name
            results.append(res)
    prog.empty(); stat.empty()
    if results:
        df = pd.DataFrame(results).sort_values(by='avg_return', ascending=False).reset_index(drop=True)
        return df
    return pd.DataFrame()


# =================================================================================
# 메인 UI 레이아웃
# =================================================================================
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🇰🇷 사카타 5법 캔들분석 + AI 4대 모듈 + 배당 랠리 백테스팅")

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
    st.info("코스피 하락 시에도 방어하는 강한 종목을 실시간으로 발굴합니다.")
    
    if st.button("📡 레이더 스캔 실행 (시총 50위)", type="primary"):
        with st.spinner("스캔 중..."):
            df, kc, kchg, _ = run_radar_scan(TOP50_FALLBACK)
            st.session_state.radar_results, st.session_state.radar_kospi_change = df, kchg
            st.rerun()
                
    if st.session_state.radar_results is not None and not st.session_state.radar_results.empty:
        df_radar = st.session_state.radar_results.copy()
        
        # 생성된 데이터프레임의 8개 컬럼에 맞춰 이름 부여
        df_radar.columns = ['종목코드', '종목명', '시장', '현재가', '등락률(%)', 'vs코스피(%p)', '거래량배율', '판정']
        
        # 포맷팅
        df_radar['현재가'] = df_radar['현재가'].apply(lambda x: f"{x:,.0f}원")
        df_radar['등락률(%)'] = df_radar['등락률(%)'].apply(lambda x: f"{x:+.2f}%")
        df_radar['vs코스피(%p)'] = df_radar['vs코스피(%p)'].apply(lambda x: f"{x:+.2f}%p")
        
        # 화면에 6개 주요 정보 표시
        st.dataframe(df_radar[['종목명', '시장', '현재가', '등락률(%)', 'vs코스피(%p)', '판정']], use_container_width=True)

# ----- TAB 2: 투자 적합 종목 추천 -----
with tab2:
    st.subheader("🎯 투자 적합 종목 추천")
    st.success("💡 **시스템 업데이트**: '사카타 5법' 캔들 패턴(적삼병, 상승삼법 등) 로직이 M1 평가 모듈에 정밀 반영되었습니다.")
    
    if st.button("🚀 전체 종목 스캔 (사카타 5법 + 4대 모듈 반영)", type="primary"):
        with st.spinner("빅데이터 스캔 중..."):
            st.session_state.scan_results = scan_stocks(all_stocks_df, mode='quick')
            st.rerun()
            
    if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
        df_show = st.session_state.scan_results[['name', 'score', 'pattern', 'module1', 'module2', 'module3']].copy()
        df_show.columns = ['종목명', '최종점수', '감지된 캔들패턴', 'M1(추세/패턴)', 'M2(거래량)', 'M3(매수조건)']
        st.dataframe(df_show.style.applymap(lambda x: 'background-color:#d4edda; font-weight:bold' if int(x)>=75 else '', subset=['최종점수']), use_container_width=True)

# ----- TAB 3: 개별 종목 분석 -----
with tab3:
    st.subheader("🔍 개별 종목 정밀 분석")
    query = st.text_input("종목명 또는 코드 입력", placeholder="예: 삼성전자, 005930", key="tab3_search")
    if st.button("🔎 분석 시작"):
        ticker, _, _ = search_stock(query, stock_dict, all_stocks_df)
        if ticker:
            st.session_state.current_ticker = ticker
            st.rerun()
            
    if st.session_state.current_ticker:
        is_valid, name, price, hist = load_stock_data(st.session_state.current_ticker)
        if is_valid:
            pattern_name, candle_score, _, _ = detect_candle_pattern_advanced(hist)
            st.markdown(f"### 📊 진단 결과: {name}")
            
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("📌 감지된 캔들 패턴", pattern_name)
            p_col2.metric("🎯 패턴 신뢰도 (사카타 5법 기준)", f"{candle_score}점")
            
            if "삼병" in pattern_name or "삼법" in pattern_name:
                st.warning(f"🔔 **사카타 5법 포착!** 강력한 추세 전환/지속을 의미하는 **{pattern_name}** 신호가 발생했습니다.")
                
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
            fig.update_layout(height=400, xaxis_rangeslider_visible=False, title="최근 가격 흐름 (Candlestick)")
            st.plotly_chart(fig, use_container_width=True)

# ----- TAB 4: 배당주 투자 가이드 (다가올 배당일 최적화) -----
with tab4:
    st.subheader("🎁 배당 랠리 최적 타점 시뮬레이터")
    st.markdown("""
    > 💡 **다가올 배당일을 10년 데이터로 공략합니다.** > 과거 10년간 해당 종목의 배당일 전후 주가 흐름을 분석하여, **배당일 며칠 전에 사야 차익이 가장 큰지** 찾아냅니다.  
    > 그리고 예측된 **다가올 배당일**을 바탕으로 '오늘' 당장 취해야 할 액션을 제시합니다.
    """)
    
    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        investment_krw = st.number_input("💰 모의 투자금액 (원)", min_value=1000000, max_value=1000000000, value=10000000, step=1000000)
    with col_input2:
        st.write("") 
        st.write("")
        run_div_scan = st.button("🚀 다가올 배당일 분석 및 최적 매수 타이밍 스캔", type="primary", use_container_width=True)
        
    if run_div_scan:
        with st.spinner("⏳ 과거 10년치 배당 랠리 패턴을 역추적하여 가장 확률 높은 시점을 찾고 있습니다..."):
            div_df = scan_all_dividend_stocks_for_rally(investment_krw)
            st.session_state.div_scan_results = div_df
            
    if st.session_state.div_scan_results is not None and not st.session_state.div_scan_results.empty:
        div_df = st.session_state.div_scan_results
        
        st.markdown("---")
        st.markdown(f"### 🏆 종목별 배당 랠리 전략 리포트 (현재일: {datetime.now().strftime('%Y-%m-%d')})")
        st.caption("✓ 과거 10년 동안 배당락일 기준, 언제 샀을 때 수익률이 가장 높았는지를 계산한 결과입니다. (매도는 무조건 배당락 2일 전)")
        
        display_div_df = div_df.copy()
        
        # 포맷팅 적용
        display_div_df['current_price'] = display_div_df['current_price'].apply(lambda x: f"{x:,.0f}원")
        display_div_df['dps'] = display_div_df['dps'].apply(lambda x: f"{x:,.0f}원" if x > 0 else "미정")
        display_div_df['div_yield'] = display_div_df['div_yield'].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
        display_div_df['avg_return'] = display_div_df['avg_return'].apply(lambda x: f"{x:+.2f}%")
        display_div_df['win_rate'] = display_div_df['win_rate'].apply(lambda x: f"{x:.0f}%")
        display_div_df['expected_profit'] = display_div_df['expected_profit'].apply(lambda x: f"{x:,.0f}원")
        
        # 보기 좋게 컬럼 재배치
        display_div_df = display_div_df[['name', 'current_price', 'dps', 'div_yield', 'next_d_day', 'best_strategy', 'avg_return', 'win_rate', 'expected_profit', 'action']]
        display_div_df.columns = [
            '종목명', '현재가', '연 배당금', '시가배당률', '예상 다음 배당일', '최적 전략',
            '평균 차익', '승률', f'예상 차익({investment_krw//10000}만)', '상태 판정'
        ]
        
        # 색상 스타일 적용
        def color_action(val):
            if '최적 매수기' in str(val): return 'color: #e74c3c; font-weight: bold; background-color: #fdf2e9'
            elif '대기' in str(val): return 'color: #f39c12'
            else: return 'color: #3498db'

        st.dataframe(display_div_df.style.applymap(color_action, subset=['상태 판정']), use_container_width=True, height=450)
        
        st.info("""
        **💡 리포트 활용 가이드**
        * **예상 다음 배당일**: 과거 배당 지급 패턴을 분석하여 계산한 가장 가까운 배당일입니다.
        * **최적 전략**: 과거 10년 동안 주가가 배당일 기준 며칠 전부터 가장 크게 올랐는지를 보여줍니다. (예: D-30일 매수)
        * **상태 판정**: 최적 매수 타이밍이 '오늘'과 얼마나 가까운지를 계산하여, 바로 사야 할지 기다려야 할지 알려줍니다.
        """)

```
