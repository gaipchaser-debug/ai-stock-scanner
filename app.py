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
    query_raw = str(query).strip()
    query_lower = query_raw.lower()

    if query_raw.isdigit():
        code_padded = query_raw.zfill(6)
        if code_padded in stock_dict:
            return stock_dict[code_padded], None, 'exact'
        if query_raw in stock_dict:
            return stock_dict[query_raw], None, 'exact'
        if all_stocks_df is not None:
            code_matches = all_stocks_df[all_stocks_df['Code'].astype(str).str.zfill(6) == code_padded]
            if len(code_matches) == 1:
                code = str(code_matches.iloc[0]['Code'])
                market = str(code_matches.iloc[0]['Market'])
                ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
                return ticker, None, 'exact'

    q_clean = query_lower.replace(" ", "").replace("(", "").replace(")", "")
    if query_lower in stock_dict:
        return stock_dict[query_lower], None, 'exact'
    if q_clean in stock_dict:
        return stock_dict[q_clean], None, 'exact'

    if all_stocks_df is not None:
        names_lower = all_stocks_df['Name'].str.lower().str.strip()
        names_clean = names_lower.str.replace(r"[\s\(\)\.\-]", "", regex=True)

        mask_partial = names_lower.str.contains(query_lower, na=False, regex=False)
        partial_matches = all_stocks_df[mask_partial]

        if len(partial_matches) == 1:
            code = str(partial_matches.iloc[0]['Code'])
            market = str(partial_matches.iloc[0]['Market'])
            ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
            return ticker, None, 'exact'
        elif len(partial_matches) > 1:
            return None, partial_matches.head(10).reset_index(drop=True), 'partial'

        mask_clean = names_clean.str.contains(q_clean, na=False, regex=False)
        clean_matches = all_stocks_df[mask_clean]
        if len(clean_matches) == 1:
            code = str(clean_matches.iloc[0]['Code'])
            market = str(clean_matches.iloc[0]['Market'])
            ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
            return ticker, None, 'exact'
        elif len(clean_matches) > 1:
            return None, clean_matches.head(10).reset_index(drop=True), 'partial'

        all_names_list = names_lower.tolist()
        for cutoff in [0.55, 0.40, 0.30]:
            close_names = difflib.get_close_matches(query_lower, all_names_list, n=10, cutoff=cutoff)
            if close_names:
                break

        if close_names:
            sim_mask = names_lower.isin(close_names)
            similar_df = all_stocks_df[sim_mask].copy()
            similar_df['_sim'] = similar_df['Name'].str.lower().apply(
                lambda n: difflib.SequenceMatcher(None, query_lower, n).ratio()
            )
            similar_df = similar_df.sort_values('_sim', ascending=False).drop(columns=['_sim'])
            return None, similar_df.head(10).reset_index(drop=True), 'similar'

    return None, None, 'notfound' 

def load_stock_data(ticker, max_retries=2):
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
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ★ 사카타 5법이 추가된 정교한 캔들 분석 함수 ★
def detect_candle_pattern_advanced(hist):
    if len(hist) < 15:
        return "데이터 부족", 50, 0, {}
    
    hist['RSI'] = calculate_rsi(hist, period=14)
    
    # 5일간 데이터 추출 (사카타 5법용)
    c = hist['Close'].iloc[-5:].values
    o = hist['Open'].iloc[-5:].values
    h = hist['High'].iloc[-5:].values
    l = hist['Low'].iloc[-5:].values
    v = hist['Volume'].iloc[-5:].values
    
    last_body = abs(c[-1] - o[-1])
    prev_body = abs(c[-2] - o[-2])
    
    avg_volume = hist['Volume'].tail(20).mean()
    volume_ratio = float(v[-1]) / avg_volume if avg_volume > 0 else 1
    current_rsi = float(hist['RSI'].iloc[-1]) if pd.notna(hist['RSI'].iloc[-1]) else 50
    
    high_20 = hist['Close'].tail(20).max()
    low_20 = hist['Close'].tail(20).min()
    price_position = (c[-1] - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
    
    pattern_details = {'rsi': current_rsi, 'volume_ratio': volume_ratio, 'price_position': price_position * 100}
    
    # --- 사카타 5법 1: 적삼병 (Three White Soldiers) ---
    if (c[-3] > o[-3] and c[-2] > o[-2] and c[-1] > o[-1] and 
        c[-3] < c[-2] < c[-1] and 
        o[-3] < o[-2] < c[-3] and o[-2] < o[-1] < c[-2]):
        match_score = int(min(100, (volume_ratio * 20) + (100 - current_rsi) * 0.5 + 40))
        return "적삼병 (사카타5법) 🚀", 95, match_score, pattern_details

    # --- 사카타 5법 2: 흑삼병 (Three Black Crows) ---
    if (c[-3] < o[-3] and c[-2] < o[-2] and c[-1] < o[-1] and 
        c[-3] > c[-2] > c[-1] and 
        c[-3] < o[-2] < o[-3] and c[-2] < o[-1] < o[-2]):
        match_score = int(min(100, current_rsi * 0.8 + 40))
        return "흑삼병 (사카타5법) ⚠️", 10, match_score, pattern_details

    # --- 사카타 5법 3: 상승 삼법 (Rising Three Methods) ---
    if (c[-5] > o[-5] and 
        c[-4] < o[-4] and c[-3] < o[-3] and c[-2] < o[-2] and 
        c[-5] > max(c[-4], c[-3], c[-2]) and o[-5] < min(c[-4], c[-3], c[-2]) and 
        c[-1] > o[-1] and c[-1] > c[-5]):
        return "상승 삼법 (사카타5법) 📈", 90, 85, pattern_details

    # --- 일반 상승 장악형 ---
    if (c[-2] < o[-2] and c[-1] > o[-1] and last_body > prev_body * 1.5):
        body_score = min(30, (last_body / prev_body) * 10)
        rsi_score = 20 if 30 <= current_rsi <= 50 else 10 if current_rsi < 70 else 0
        volume_score = min(25, volume_ratio * 12.5)
        position_score = 25 if price_position < 0.5 else 15
        return "상승 장악형 🟢", 80, int(body_score + rsi_score + volume_score + position_score), pattern_details
    
    # --- 일반 망치형 ---
    lower_shadow = min(o[-1], c[-1]) - l[-1]
    upper_shadow = h[-1] - max(o[-1], c[-1])
    if last_body > 0 and lower_shadow > last_body * 2 and upper_shadow < last_body * 0.5:
        shadow_score = min(35, (lower_shadow / last_body) * 12)
        rsi_score = 25 if current_rsi < 35 else 15 if current_rsi < 50 else 5
        volume_score = min(20, volume_ratio * 10)
        position_score = 20 if price_position < 0.3 else 10
        return "해머형(망치형) 🟢", 75, int(shadow_score + rsi_score + volume_score + position_score), pattern_details
    
    return "일반 캔들 (패턴 없음)", 50, 0, pattern_details

def calculate_stock_score(hist, current_price, vs_kospi=None, verdict=None):
    try:
        if len(hist) < 20:
            return 0, {}

        hist = hist.copy()
        hist['MA5']   = hist['Close'].rolling(window=5).mean()
        hist['MA20']  = hist['Close'].rolling(window=20).mean()
        hist['MA60']  = hist['Close'].rolling(window=60).mean()
        hist['MA120'] = hist['Close'].rolling(window=120).mean()

        latest = hist.iloc[-1]
        ma5   = float(latest['MA5'])   if pd.notna(latest['MA5'])   else current_price
        ma20  = float(latest['MA20'])  if pd.notna(latest['MA20'])  else current_price
        ma60  = float(latest['MA60'])  if pd.notna(latest['MA60'])  else current_price
        ma120 = float(latest['MA120']) if pd.notna(latest['MA120']) else current_price

        # ★ 사카타 5법 패턴 점수 가져오기 ★
        pattern_name, candle_score, _, _ = detect_candle_pattern_advanced(hist)

        # M1 (추세 + 캔들 패턴 통합)
        align_count = sum([ma5>ma20, ma20>ma60, ma60>ma120, ma5>ma60, ma20>ma120])
        ma_score = {5: 92, 4: 78, 3: 62, 2: 46, 1: 32, 0: 15}[align_count]
        
        cross_score = 50
        if len(hist) >= 5:
            prev_ma20 = hist['MA20'].iloc[-2] if pd.notna(hist['MA20'].iloc[-2]) else ma20
            prev_ma60 = hist['MA60'].iloc[-2] if pd.notna(hist['MA60'].iloc[-2]) else ma60
            if ma20 > ma60 and prev_ma20 <= prev_ma60: cross_score = 95
            elif ma20 < ma60 and prev_ma20 >= prev_ma60: cross_score = 8
                
        # 캔들 패턴 스코어를 모듈1에 30% 비중으로 반영
        module1_score = int(ma_score * 0.4 + cross_score * 0.3 + candle_score * 0.3)

        # M2 (거래량)
        hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()
        current_volume = float(hist['Volume'].iloc[-1])
        avg_volume = float(hist['Volume_MA20'].iloc[-1]) if pd.notna(hist['Volume_MA20'].iloc[-1]) else 1
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        raw_score = min(95, max(20, volume_ratio * 38 + 10))
        module2_score = int(min(95, raw_score + 5) if volume_ratio >= 2.0 else raw_score)

        # M3 (매수 조건)
        cond1 = current_price > ma120
        high_20d = float(hist['Close'].tail(20).max())
        cond2 = current_price >= high_20d * 0.95
        pct_chg = ((current_price - float(hist['Close'].iloc[-2])) / float(hist['Close'].iloc[-2])) * 100 if len(hist) >= 2 else 0
        cond3 = -2 <= pct_chg <= 15
        cond4 = volume_ratio >= 1.5
        satisfied = sum([cond1, cond2, cond3, cond4])
        module3_score = {4: 85, 3: 65, 2: 45, 1: 30, 0: 15}[satisfied]

        # M4 (R:R)
        sl_methods = {'open': float(latest['Open']), 'low': float(latest['Low']), '3pct': current_price * 0.97, 'ma20': ma20}
        final_sl = max(sl_methods.values())
        risk_pct   = ((final_sl - current_price) / current_price) * 100 
        target     = current_price + abs(current_price - final_sl) * 2
        reward_pct = ((target - current_price) / current_price) * 100
        risk_reward_ratio = abs(reward_pct / risk_pct) if risk_pct != 0 else 1.0
        module4_score = min(95, max(20, int(risk_reward_ratio * 28 + 18)))

        # M5 (시장 강도)
        module5_score = None
        if vs_kospi is not None:
            if verdict == "⭐ 역주행": module5_score = 100
            elif vs_kospi > 3.0: module5_score = 95
            elif vs_kospi > 1.0: module5_score = 75
            elif vs_kospi > 0: module5_score = 63
            elif vs_kospi > -1.0: module5_score = 50
            else: module5_score = 20

        if module5_score is not None:
            final_score = int(module1_score*0.2 + module2_score*0.2 + module3_score*0.2 + module4_score*0.2 + module5_score*0.2)
        else:
            final_score = int(module1_score*0.25 + module2_score*0.25 + module3_score*0.25 + module4_score*0.25)

        details = {
            'module1': module1_score, 'module2': module2_score, 'module3': module3_score, 
            'module4': module4_score, 'module5': module5_score, 'volume_ratio': round(volume_ratio, 2),
            'conditions': satisfied, 'rr_ratio': round(risk_reward_ratio, 2),
            'vs_kospi': round(vs_kospi, 2) if vs_kospi is not None else None,
            'verdict': verdict, 'pattern_name': pattern_name
        }
        return final_score, details

    except Exception as e:
        return 0, {}

def scan_stocks(stock_list, mode='quick'):
    results = []
    radar_lookup = {}
    radar_df = st.session_state.get('radar_results', None)
    if radar_df is not None and not radar_df.empty and 'ticker' in radar_df.columns:
        for _, rrow in radar_df.iterrows():
            radar_lookup[rrow['ticker']] = {'vs_kospi': rrow.get('vs_kospi'), 'verdict': rrow.get('verdict')}

    stocks_to_scan = stock_list.head(100) if mode == 'quick' else stock_list
    progress_bar = st.progress(0)
    status_text  = st.empty()
    total = len(stocks_to_scan)

    for enum_idx, (df_idx, row) in enumerate(stocks_to_scan.iterrows()):
        code   = str(row['Code'])
        name   = str(row['Name'])
        market = str(row['Market'])
        ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"

        progress_bar.progress((enum_idx + 1) / total)
        status_text.text(f"분석 중: {name} ({code}) - {enum_idx+1}/{total}")

        is_valid, company_name, current_price, hist = load_stock_data(ticker, max_retries=1)
        if not is_valid: continue

        radar_info = radar_lookup.get(ticker, {})
        score, details = calculate_stock_score(hist, current_price, vs_kospi=radar_info.get('vs_kospi'), verdict=radar_info.get('verdict'))

        if score >= 50:
            results.append({
                'ticker': ticker, 'code': code, 'name': name, 'market': market,
                'price': current_price, 'score': score, 
                'module1': details.get('module1',0), 'module2': details.get('module2',0),
                'module3': details.get('module3',0), 'module4': details.get('module4',0),
                'module5': details.get('module5'), 'pattern': details.get('pattern_name', ''),
                'volume_ratio': details.get('volume_ratio',0), 'conditions': details.get('conditions',0),
                'rr_ratio': details.get('rr_ratio',0), 'vs_kospi': details.get('vs_kospi'), 'verdict': details.get('verdict')
            })

    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results).sort_values('score', ascending=False).head(10) if results else pd.DataFrame()

# ========== 시장 레이더 데이터 관련 ==========
TOP50_FALLBACK = [
    ("005930", "삼성전자", "KOSPI"), ("000660", "SK하이닉스", "KOSPI"), ("207940", "삼성바이오로직스", "KOSPI"), 
    ("005380", "현대차", "KOSPI"), ("373220", "LG에너지솔루션", "KOSPI"), ("000270", "기아", "KOSPI"),
    ("068270", "셀트리온", "KOSPI"), ("005490", "POSCO홀딩스", "KOSPI"), ("035420", "NAVER", "KOSPI"),
    ("035720", "카카오", "KOSPI"), ("051910", "LG화학", "KOSPI"), ("006400", "삼성SDI", "KOSPI"),
    ("028260", "삼성물산", "KOSPI"), ("105560", "KB금융", "KOSPI"), ("055550", "신한지주", "KOSPI")
]

@st.cache_data(ttl=300)
def get_kospi_status():
    try:
        k = yf.Ticker("^KS11")
        hist = k.history(period="5d")
        if len(hist) >= 2:
            current = float(hist['Close'].iloc[-1])
            prev = float(hist['Close'].iloc[-2])
            return current, (current - prev) / prev * 100, current - prev, hist
        return None, 0, 0, None
    except: return None, 0, 0, None

@st.cache_data(ttl=300)
def get_stock_today_change(ticker):
    try:
        s = yf.Ticker(ticker)
        hist = s.history(period="5d")
        if len(hist) >= 2:
            cur, prev = float(hist['Close'].iloc[-1]), float(hist['Close'].iloc[-2])
            vol, vol_avg = float(hist['Volume'].iloc[-1]), float(hist['Volume'].mean())
            return cur, (cur - prev) / prev * 100, (vol / vol_avg if vol_avg > 0 else 1.0)
        return None, None, None
    except: return None, None, None

@st.cache_data(ttl=300)
def get_normalized_chart(ticker, kospi_hist):
    try:
        stock_hist = yf.Ticker(ticker).history(period="3mo")
        if stock_hist.empty: return None, None
        return stock_hist.index, stock_hist['Close'] / stock_hist['Close'].iloc[0] * 100
    except: return None, None

@st.cache_data(ttl=3600)
def get_defense_rate(ticker, period="2mo"):
    try:
        k = yf.Ticker("^KS11")
        kospi_hist = k.history(period=period)
        s = yf.Ticker(ticker)
        stock_hist = s.history(period=period)

        if kospi_hist.empty or stock_hist.empty: return None

        try: kospi_hist.index = kospi_hist.index.tz_localize(None)
        except: kospi_hist.index = kospi_hist.index.tz_convert(None)
        try: stock_hist.index = stock_hist.index.tz_localize(None)
        except: stock_hist.index = stock_hist.index.tz_convert(None)

        k_ret = kospi_hist['Close'].pct_change().dropna() * 100
        s_ret = stock_hist['Close'].pct_change().dropna() * 100
        common_idx = k_ret.index.intersection(s_ret.index)
        
        if len(common_idx) < 10: return None
        k_ret, s_ret = k_ret[common_idx], s_ret[common_idx]

        down_mask  = k_ret < 0
        total_down = int(down_mask.sum())

        if total_down == 0:
            return {'total_down_days': 0, 'defense_days': 0, 'defense_rate': 0.0, 'reverse_days': 0, 'reverse_rate': 0.0, 'avg_gap_down': 0.0, 'period': period}

        vs_on_down   = s_ret[down_mask] - k_ret[down_mask]
        defense_days = int((vs_on_down > 0).sum())
        reverse_days = int((s_ret[down_mask] > 0).sum())
        
        return {
            'total_down_days': total_down, 'defense_days': defense_days,
            'defense_rate': round(defense_days / total_down * 100, 1),
            'reverse_days': reverse_days, 'reverse_rate': round(reverse_days / total_down * 100, 1),
            'avg_gap_down': round(float(vs_on_down.mean()), 2), 'period': period,
        }
    except: return None

def run_radar_scan(stock_list):
    kospi_current, kospi_change, kospi_pt, kospi_hist = get_kospi_status()
    results = []
    progress_bar, status_text = st.progress(0), st.empty()
    
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

        results.append({
            "ticker": ticker, "code": code, "name": name, "market": market,
            "price": price, "change_pct": round(chg, 2), "vs_kospi": round(vs_kospi, 2),
            "vol_ratio": round(vol_ratio, 2), "verdict": verdict
        })
    progress_bar.empty(); status_text.empty()
    df = pd.DataFrame(results)
    if not df.empty: df = df.sort_values("vs_kospi", ascending=False).reset_index(drop=True)
    return df, kospi_current, kospi_change, kospi_hist

# =========================================================================
# ★ TAB 4: 배당주 백테스팅 & 배당금 정보 추출 로직 ★
# =========================================================================
DIVIDEND_CANDIDATES = [
    ("105560", "KB금융", "KOSPI"), ("055550", "신한지주", "KOSPI"), ("086790", "하나금융지주", "KOSPI"), 
    ("316140", "우리금융지주", "KOSPI"), ("032640", "LG유플러스", "KOSPI"), ("017670", "SK텔레콤", "KOSPI"), 
    ("030200", "KT", "KOSPI"), ("033780", "KT&G", "KOSPI"), ("024110", "기업은행", "KOSPI"), 
    ("005930", "삼성전자", "KOSPI"), ("005380", "현대차", "KOSPI"), ("090430", "아모레퍼시픽", "KOSPI")
]

def get_dividend_details(stock_obj):
    """최근 배당금(DPS) 및 배당락일 추출 (강건한 예외처리 포함)"""
    try:
        info = stock_obj.info
        dps = info.get('trailingAnnualDividendRate')
        div_date_ts = info.get('exDividendDate')
        
        div_date_str = "조회 불가"
        if div_date_ts:
            div_date_str = datetime.fromtimestamp(div_date_ts).strftime('%Y-%m-%d')
            
        # info에 없으면 배당 내역 직접 뒤지기
        if not dps or not div_date_ts:
            div_history = stock_obj.dividends
            if not div_history.empty:
                # 마지막 배당일
                last_date = div_history.index[-1]
                div_date_str = last_date.strftime('%Y-%m-%d')
                
                # 최근 1년간 배당금 합산 (연배당금)
                one_year_ago = last_date - pd.DateOffset(years=1)
                dps = div_history[div_history.index > one_year_ago].sum()
        
        if not dps: dps = 0
        return dps, div_date_str
    except:
        return 0, "조회 불가"

def scan_all_dividend_stocks(investment_amount):
    """배당주 과거 10년 시뮬레이션 및 배당 정보 추출"""
    results = []
    today = datetime.now()
    target_sell_month, target_sell_day = 12, 26 # 통상적인 배당락일 전 매도 기준
    
    prog, stat = st.progress(0), st.empty()
    for idx, (code, name, market) in enumerate(DIVIDEND_CANDIDATES):
        prog.progress((idx + 1) / len(DIVIDEND_CANDIDATES))
        stat.text(f"배당 정보 및 10년 데이터 연산 중: {name}")
        ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
        
        try:
            stock = yf.Ticker(ticker)
            # 배당 정보 추출
            dps, ex_div_date = get_dividend_details(stock)
            
            # 주가 궤적 백테스팅
            hist = stock.history(period="10y")
            if hist.empty or len(hist) < 252: continue
            
            hist['Month'] = hist.index.month
            hist['Day'] = hist.index.day
            hist['Year'] = hist.index.year
            
            today_month, today_day = today.month, today.day
            buy_prices, sell_prices = [], []
            
            for year in hist['Year'].unique():
                if year == today.year: continue # 올해 데이터 아직 없으므로 스킵
                
                # 해당 연도의 오늘 시점 부근 찾기
                buy_mask = (hist['Year'] == year) & (hist['Month'] == today_month) & (hist['Day'] >= today_day)
                sell_mask = (hist['Year'] == year) & (hist['Month'] == target_sell_month) & (hist['Day'] <= target_sell_day)
                
                if not hist[buy_mask].empty and not hist[sell_mask].empty:
                    b_price = hist[buy_mask].iloc[0]['Close']
                    s_price = hist[sell_mask].iloc[-1]['Close']
                    buy_prices.append(b_price)
                    sell_prices.append(s_price)
            
            if not buy_prices: continue
            
            # 수익률 계산
            returns = [(s - b) / b * 100 for s, b in zip(sell_prices, buy_prices)]
            avg_return = np.mean(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
            expected_profit = investment_amount * (avg_return / 100)
            
            current_price = hist.iloc[-1]['Close']
            dividend_yield = (dps / current_price * 100) if current_price > 0 else 0
            
            # 매수 판정
            if avg_return >= 3.0 and win_rate >= 70: action = "🔥 적극 매수 (Buy Now)"
            elif avg_return > 0: action = "🟡 분할 매수 대기"
            else: action = "🔴 시기 부적합"
            
            results.append({
                'ticker': ticker, 'name': name, 'current_price': current_price,
                'dps': dps, 'div_yield': dividend_yield, 'ex_div_date': ex_div_date,
                'avg_return': avg_return, 'win_rate': win_rate, 
                'expected_profit': expected_profit, 'action': action
            })
        except: continue
        
    prog.empty(); stat.empty()
    if results:
        df = pd.DataFrame(results).sort_values(by='avg_return', ascending=False).reset_index(drop=True)
        return df
    return pd.DataFrame()


# =================================================================================
# 메인 UI
# =================================================================================
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🇰🇷 사카타 5법 정밀 캔들분석 + AI 4대 모듈 + 고배당 백테스팅")

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
    if st.button("레이더 스캔 실행 (시총 50위)", type="primary"):
        with st.spinner("스캔 중..."):
            df, kc, kchg, _ = run_radar_scan(TOP50_FALLBACK)
            st.session_state.radar_results, st.session_state.radar_kospi_change = df, kchg
            st.rerun()
            
    if st.session_state.radar_results is not None and not st.session_state.radar_results.empty:
        df_radar = st.session_state.radar_results.copy()
        df_radar.columns = ['종목코드', '기호', '종목명', '시장', '현재가', '등락률(%)', 'vs코스피(%p)', '거래량배율', '판정']
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
    query = st.text_input("종목명 또는 코드 입력", placeholder="예: 삼성전자, 005930")
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

# ----- TAB 4: 배당주 투자 가이드 (배당금/배당일 포함) -----
with tab4:
    st.subheader("🎁 실전 배당주 투자 가이드 & 시뮬레이터")
    st.markdown("""
    > 💡 국내 주요 고배당주의 **연간 주당 배당금(예상)**과 **과거 10년 치 주가 흐름**을 동시 분석합니다.  
    > 배당락 전 시세차익을 노리는 통계적 전략에 따라 '오늘' 매수 시의 성과를 보여줍니다.
    """)
    
    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        investment_krw = st.number_input("💰 모의 투자금액 (원)", min_value=1000000, max_value=1000000000, value=10000000, step=1000000)
    with col_input2:
        st.write("") 
        st.write("")
        run_div_scan = st.button("🚀 전체 배당주 백테스팅 및 배당정보 업데이트", type="primary", use_container_width=True)
        
    if run_div_scan:
        with st.spinner("⏳ 배당 내역 추출 및 10년 과거 데이터 시뮬레이션 연산 중... (약 1~2분 소요)"):
            div_df = scan_all_dividend_stocks(investment_krw)
            st.session_state.div_scan_results = div_df
            
    if st.session_state.div_scan_results is not None and not st.session_state.div_scan_results.empty:
        div_df = st.session_state.div_scan_results
        
        st.markdown("---")
        st.markdown("### 🏆 배당주 매매 시뮬레이션 리포트")
        st.caption(f"✓ 기준일({datetime.now().strftime('%Y-%m-%d')})에 매수하여 12월 말 배당락 전에 매도했을 경우의 과거 10년 성과 통계입니다.")
        
        display_div_df = div_df.copy()
        
        # 포맷팅 적용
        display_div_df['current_price'] = display_div_df['current_price'].apply(lambda x: f"{x:,.0f}원")
        display_div_df['dps'] = display_div_df['dps'].apply(lambda x: f"{x:,.0f}원" if x > 0 else "미정")
        display_div_df['div_yield'] = display_div_df['div_yield'].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
        display_div_df['avg_return'] = display_div_df['avg_return'].apply(lambda x: f"{x:+.2f}%")
        display_div_df['win_rate'] = display_div_df['win_rate'].apply(lambda x: f"{x:.0f}%")
        display_div_df['expected_profit'] = display_div_df['expected_profit'].apply(lambda x: f"{x:,.0f}원")
        
        # 보기 좋게 컬럼 재배치 및 이름 변경
        display_div_df = display_div_df[['name', 'current_price', 'dps', 'div_yield', 'ex_div_date', 'avg_return', 'win_rate', 'expected_profit', 'action']]
        display_div_df.columns = [
            '종목명', '현재가', '주당배당금(예상)', '시가배당률', '최근배당(락)일', 
            '10년 평균수익률', '승률', f'예상 차익수익({investment_krw//10000}만)', '전략 판정'
        ]
        
        # 색상 스타일 적용
        def color_action(val):
            if 'Buy Now' in str(val): return 'color: #e74c3c; font-weight: bold; background-color: #fdf2e9'
            elif '대기' in str(val): return 'color: #f39c12'
            else: return 'color: #7f8c8d'

        st.dataframe(display_div_df.style.applymap(color_action, subset=['전략 판정']), use_container_width=True, height=500)
        
        st.info("""
        **💡 배당주 투자 전략 팁**
        * **주당배당금(DPS) 및 시가배당률**: 최근 1년간 지급된 배당금의 합산 추정치입니다. 배당 수익 자체를 원할 경우 이 지표가 높은 종목을 배당락일 전까지 보유하세요.
        * **10년 평균수익률 & 승률**: 배당을 받지 않고, 배당락일 직전에 나타나는 '배당 랠리' 고점에서 매도하여 시세차익만 챙길 경우의 통계입니다. **(승률 70% 이상 = 적극 매수 권장)**
        """)
