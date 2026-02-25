import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
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
if 'radar_kospi_change' not in st.session_state:
    st.session_state.radar_kospi_change = None
if 'radar_kospi_current' not in st.session_state:
    st.session_state.radar_kospi_current = None

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
    """
    종목 검색 - 4단계 폴백 검색 + 유사 종목 추천
    Returns: (ticker_or_None, matches_df_or_None, search_type)
      search_type: 'exact' | 'partial' | 'similar' | 'notfound'
    """
    query_raw = str(query).strip()
    query_lower = query_raw.lower()

    # ── 1단계: 종목 코드 직접 입력 (숫자) ──────────────────────────
    if query_raw.isdigit():
        code_padded = query_raw.zfill(6)
        # 정확 코드 일치
        if code_padded in stock_dict:
            return stock_dict[code_padded], None, 'exact'
        # stock_dict는 소문자 이름 & 코드 둘 다 저장 — 원본 코드로도 시도
        if query_raw in stock_dict:
            return stock_dict[query_raw], None, 'exact'
        # all_stocks_df에서 코드 검색
        if all_stocks_df is not None:
            code_matches = all_stocks_df[
                all_stocks_df['Code'].astype(str).str.zfill(6) == code_padded
            ]
            if len(code_matches) == 1:
                code = str(code_matches.iloc[0]['Code'])
                market = str(code_matches.iloc[0]['Market'])
                ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
                return ticker, None, 'exact'

    # ── 2단계: 이름 정확 매칭 (공백·특수문자 제거 후) ───────────────
    q_clean = query_lower.replace(" ", "").replace("(", "").replace(")", "")
    if query_lower in stock_dict:
        return stock_dict[query_lower], None, 'exact'
    if q_clean in stock_dict:
        return stock_dict[q_clean], None, 'exact'

    if all_stocks_df is not None:
        # Name 컬럼 정규화 버전 캐시
        names_lower = all_stocks_df['Name'].str.lower().str.strip()
        names_clean = names_lower.str.replace(r"[\s\(\)\.\-]", "", regex=True)

        # ── 3단계: 부분 포함 검색 (regex=False — 한글/특수문자 안전) ──
        mask_partial = names_lower.str.contains(query_lower, na=False, regex=False)
        partial_matches = all_stocks_df[mask_partial]

        if len(partial_matches) == 1:
            code = str(partial_matches.iloc[0]['Code'])
            market = str(partial_matches.iloc[0]['Market'])
            ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
            return ticker, None, 'exact'
        elif len(partial_matches) > 1:
            return None, partial_matches.head(10).reset_index(drop=True), 'partial'

        # 공백 제거 버전으로 한 번 더
        mask_clean = names_clean.str.contains(q_clean, na=False, regex=False)
        clean_matches = all_stocks_df[mask_clean]
        if len(clean_matches) == 1:
            code = str(clean_matches.iloc[0]['Code'])
            market = str(clean_matches.iloc[0]['Market'])
            ticker = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
            return ticker, None, 'exact'
        elif len(clean_matches) > 1:
            return None, clean_matches.head(10).reset_index(drop=True), 'partial'

        # ── 4단계: difflib 유사도 추천 ────────────────────────────────
        all_names_list = names_lower.tolist()
        # cutoff 0.5부터 시작, 결과 없으면 0.35로 낮춤
        for cutoff in [0.55, 0.40, 0.30]:
            close_names = difflib.get_close_matches(query_lower, all_names_list, n=10, cutoff=cutoff)
            if close_names:
                break

        if close_names:
            sim_mask = names_lower.isin(close_names)
            similar_df = all_stocks_df[sim_mask].copy()
            # 유사도 점수 컬럼 추가 (정렬용)
            similar_df['_sim'] = similar_df['Name'].str.lower().apply(
                lambda n: difflib.SequenceMatcher(None, query_lower, n).ratio()
            )
            similar_df = similar_df.sort_values('_sim', ascending=False).drop(columns=['_sim'])
            return None, similar_df.head(10).reset_index(drop=True), 'similar'

    return None, None, 'notfound' 

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

def calculate_stock_score(hist, current_price, vs_kospi=None, verdict=None):
    """
    종목 점수 계산 (세분화된 5모듈 버전)
    - vs_kospi: 시장 레이더에서 계산된 코스피 대비 초과수익률 (있으면 M5 포함)
    - verdict: 시장 레이더 판정 (역주행/방어 등)
    """
    try:
        if len(hist) < 20:
            return 0, {}

        # ── 이동평균 계산 ──────────────────────────────────────────
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

        # ── 모듈 1: 추세·정배열 (세분화) ──────────────────────────
        # MA 정배열 점수: 몇 개 MA가 정순서인지 세분화
        align_checks = [
            ma5   > ma20,
            ma20  > ma60,
            ma60  > ma120,
            ma5   > ma60,   # 추가 강도
            ma20  > ma120,  # 추가 강도
        ]
        align_count = sum(align_checks)
        if align_count == 5:          # 완전 정배열
            ma_score = 92
        elif align_count == 4:
            ma_score = 78
        elif align_count == 3:
            ma_score = 62
        elif align_count == 2:
            ma_score = 46
        elif align_count == 1:
            ma_score = 32
        else:                         # 완전 역배열
            ma_score = 15

        # MA 기울기 보너스 (+최대 8점)
        slope_bonus = 0
        if len(hist) >= 5:
            ma20_slope = float(hist['MA20'].iloc[-1]) - float(hist['MA20'].iloc[-5]) if pd.notna(hist['MA20'].iloc[-5]) else 0
            ma60_slope = float(hist['MA60'].iloc[-1]) - float(hist['MA60'].iloc[-5]) if pd.notna(hist['MA60'].iloc[-5]) else 0
            if ma20_slope > 0:
                slope_bonus += 4
            if ma60_slope > 0:
                slope_bonus += 4
        ma_score = min(100, ma_score + slope_bonus)

        # 골든/데드크로스 점수 (세분화)
        cross_score = 50
        if len(hist) >= 5:
            prev_ma20 = hist['MA20'].iloc[-2] if pd.notna(hist['MA20'].iloc[-2]) else ma20
            prev_ma60 = hist['MA60'].iloc[-2] if pd.notna(hist['MA60'].iloc[-2]) else ma60
            gap_pct = abs(ma20 - ma60) / ma60 * 100 if ma60 != 0 else 0

            if ma20 > ma60 and prev_ma20 <= prev_ma60:   # 골든크로스 발생
                cross_score = 95
            elif ma20 > ma60:
                if gap_pct >= 3:
                    cross_score = 72   # 골든크로스 이후 안정
                elif gap_pct >= 1:
                    cross_score = 63   # 골든크로스 직후
                else:
                    cross_score = 55   # 막 넘어선 상태
            elif ma20 < ma60 and prev_ma20 >= prev_ma60: # 데드크로스 발생
                cross_score = 8
            else:
                if gap_pct >= 3:
                    cross_score = 22   # 역배열 깊음
                elif gap_pct >= 1:
                    cross_score = 32   # 역배열 진입
                else:
                    cross_score = 42   # 수렴 중

        module1_score = int(ma_score * 0.6 + cross_score * 0.4)

        # ── 모듈 2: 거래량 (연속형 세분화) ───────────────────────
        hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()
        current_volume = float(hist['Volume'].iloc[-1])
        avg_volume = float(hist['Volume_MA20'].iloc[-1]) if pd.notna(hist['Volume_MA20'].iloc[-1]) else 1

        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            # 연속형 점수: 거래량 배율에 비례 (20~95 범위)
            raw_score = min(95, max(20, volume_ratio * 38 + 10))
            # 임계값 보정: 2배 이상이면 추가 보너스
            if volume_ratio >= 3.0:
                volume_score = 95
            elif volume_ratio >= 2.0:
                volume_score = min(95, raw_score + 5)
            else:
                volume_score = raw_score
        else:
            volume_ratio = 0
            volume_score = 45

        module2_score = int(volume_score)

        # ── 모듈 3: 매수 조건 (조건 품질 세분화) ──────────────────
        cond1 = current_price > ma120
        high_20d = float(hist['Close'].tail(20).max())
        low_20d  = float(hist['Close'].tail(20).min())
        cond2 = current_price >= high_20d * 0.95

        if len(hist) >= 2:
            pct_chg = ((current_price - float(hist['Close'].iloc[-2])) / float(hist['Close'].iloc[-2])) * 100
            cond3 = -2 <= pct_chg <= 15
        else:
            pct_chg = 0
            cond3 = False

        cond4 = volume_ratio >= 1.5
        satisfied = sum([cond1, cond2, cond3, cond4])

        # 기본 점수
        base3 = {4: 80, 3: 63, 2: 46, 1: 30, 0: 15}[satisfied]

        # 품질 보너스 (최대 20점)
        quality_bonus = 0
        # cond1 품질: MA120 대비 얼마나 위에 있나
        if cond1 and ma120 > 0:
            margin = (current_price - ma120) / ma120 * 100
            quality_bonus += min(5, margin * 0.5)
        # cond2 품질: 20일 고점에 얼마나 근접
        if high_20d > low_20d:
            pos_ratio = (current_price - low_20d) / (high_20d - low_20d)
            quality_bonus += min(5, pos_ratio * 5)
        # cond3 품질: 최적 변화율 3~10%에 가까울수록
        if cond3 and 3 <= pct_chg <= 10:
            quality_bonus += 5
        elif cond3 and 0 < pct_chg < 3:
            quality_bonus += 3
        # cond4 품질: 거래량 배율이 높을수록
        if cond4:
            quality_bonus += min(5, (volume_ratio - 1.5) * 2.5)

        module3_score = min(100, int(base3 + quality_bonus))

        # ── 모듈 4: 리스크·리워드 (연속형 세분화) ─────────────────
        sl_methods = {
            'open': float(latest['Open']),
            'low':  float(latest['Low']),
            '3pct': current_price * 0.97,
            'ma20': ma20,
        }
        final_sl = max(sl_methods.values())
        risk_pct   = ((final_sl - current_price) / current_price) * 100  # 음수
        target     = current_price + abs(current_price - final_sl) * 2
        reward_pct = ((target - current_price) / current_price) * 100

        risk_reward_ratio = abs(reward_pct / risk_pct) if risk_pct != 0 else 1.0
        # 연속형 점수: R:R에 비례 (20~95 범위)
        module4_score = min(95, max(20, int(risk_reward_ratio * 28 + 18)))

        # ── 모듈 5: 시장 상대 강도 (레이더 데이터 있을 때) ────────
        module5_score = None
        if vs_kospi is not None:
            if verdict == "⭐ 역주행":           # 지수 하락 중 상승 → 최고
                module5_score = 100
            elif vs_kospi > 3.0:
                module5_score = 95
            elif vs_kospi > 2.0:
                module5_score = 85
            elif vs_kospi > 1.0:
                module5_score = 75
            elif vs_kospi > 0:
                module5_score = 63
            elif vs_kospi > -1.0:
                module5_score = 50
            elif vs_kospi > -2.0:
                module5_score = 35
            else:
                module5_score = 20

        # ── 최종 점수 (모듈 가중 합산) ────────────────────────────
        if module5_score is not None:
            # M5 포함: 각 20%
            final_score = int(
                module1_score * 0.20 +
                module2_score * 0.20 +
                module3_score * 0.20 +
                module4_score * 0.20 +
                module5_score * 0.20
            )
        else:
            final_score = int(
                module1_score * 0.25 +
                module2_score * 0.25 +
                module3_score * 0.25 +
                module4_score * 0.25
            )

        details = {
            'module1':      module1_score,
            'module2':      module2_score,
            'module3':      module3_score,
            'module4':      module4_score,
            'module5':      module5_score,
            'volume_ratio': round(volume_ratio, 2),
            'conditions':   satisfied,
            'rr_ratio':     round(risk_reward_ratio, 2),
            'pct_chg':      round(pct_chg, 2) if len(hist) >= 2 else 0,
            'vs_kospi':     round(vs_kospi, 2) if vs_kospi is not None else None,
            'verdict':      verdict,
        }

        return final_score, details

    except Exception as e:
        return 0, {}

def scan_stocks(stock_list, mode='quick'):
    """
    종목 스캔 (시장 레이더 데이터 연동 + M5 반영)
    """
    results = []

    # ─ 시장 레이더 데이터 미리 수집 ─────────────────────────────────
    radar_lookup = {}   # {ticker: {'vs_kospi': float, 'verdict': str}}
    radar_df = st.session_state.get('radar_results', None)
    if radar_df is not None and not radar_df.empty and 'ticker' in radar_df.columns:
        for _, rrow in radar_df.iterrows():
            radar_lookup[rrow['ticker']] = {
                'vs_kospi': rrow.get('vs_kospi', None),
                'verdict':  rrow.get('verdict',  None),
            }
        st.info(f"📡 시장 레이더 데이터 {len(radar_lookup)}개 종목 연동 완료 → M5 모듈 적용")
    else:
        st.info("⚠️ 시장 레이더 데이터 없음 → M1~M4 기준 점수 산정 (M5 제외)")

    if mode == 'quick':
        stocks_to_scan = stock_list.head(100)
        st.info("🔍 주요 대형주 100개 종목 스캔 중...")
    else:
        stocks_to_scan = stock_list
        st.warning(f"🔍 전체 {len(stock_list)}개 종목 스캔 중... 약 10-20분 소요됩니다.")

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
        if not is_valid:
            continue

        # 레이더 데이터 연동
        radar_info = radar_lookup.get(ticker, {})
        vs_kospi   = radar_info.get('vs_kospi', None)
        verdict    = radar_info.get('verdict',  None)

        score, details = calculate_stock_score(
            hist, current_price, vs_kospi=vs_kospi, verdict=verdict
        )

        if score >= 50:   # 50점 이상만 저장 (더 많은 후보 확보 후 상위 10개로 필터)
            results.append({
                'ticker':       ticker,
                'code':         code,
                'name':         name,
                'market':       market,
                'price':        current_price,
                'score':        score,
                'module1':      details.get('module1', 0),
                'module2':      details.get('module2', 0),
                'module3':      details.get('module3', 0),
                'module4':      details.get('module4', 0),
                'module5':      details.get('module5'),         # None 가능
                'volume_ratio': details.get('volume_ratio', 0),
                'conditions':   details.get('conditions', 0),
                'rr_ratio':     details.get('rr_ratio', 0),
                'vs_kospi':     details.get('vs_kospi'),        # None 가능
                'verdict':      details.get('verdict'),         # None 가능
                'pct_chg':      details.get('pct_chg', 0),
            })

        time.sleep(0.1)

    progress_bar.empty()
    status_text.empty()

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

# ========== 시장 레이더 함수들 ==========

# 시총 상위 50 종목 (폴백용 하드코딩)
TOP50_FALLBACK = [
    ("005930", "삼성전자",      "KOSPI"),
    ("000660", "SK하이닉스",    "KOSPI"),
    ("207940", "삼성바이오로직스","KOSPI"),
    ("005380", "현대차",        "KOSPI"),
    ("373220", "LG에너지솔루션",  "KOSPI"),
    ("000270", "기아",          "KOSPI"),
    ("068270", "셀트리온",       "KOSPI"),
    ("005490", "POSCO홀딩스",   "KOSPI"),
    ("035420", "NAVER",         "KOSPI"),
    ("035720", "카카오",         "KOSPI"),
    ("051910", "LG화학",        "KOSPI"),
    ("006400", "삼성SDI",       "KOSPI"),
    ("028260", "삼성물산",       "KOSPI"),
    ("105560", "KB금융",        "KOSPI"),
    ("055550", "신한지주",       "KOSPI"),
    ("066570", "LG전자",        "KOSPI"),
    ("003550", "LG",            "KOSPI"),
    ("034730", "SK",            "KOSPI"),
    ("032830", "삼성생명",       "KOSPI"),
    ("086790", "하나금융지주",    "KOSPI"),
    ("096770", "SK이노베이션",    "KOSPI"),
    ("017670", "SK텔레콤",       "KOSPI"),
    ("030200", "KT",            "KOSPI"),
    ("316140", "우리금융지주",    "KOSPI"),
    ("012330", "현대모비스",      "KOSPI"),
    ("009150", "삼성전기",       "KOSPI"),
    ("011200", "HMM",           "KOSPI"),
    ("010950", "S-Oil",         "KOSPI"),
    ("000810", "삼성화재",       "KOSPI"),
    ("024110", "기업은행",       "KOSPI"),
    ("033780", "KT&G",          "KOSPI"),
    ("003490", "대한항공",       "KOSPI"),
    ("090430", "아모레퍼시픽",    "KOSPI"),
    ("011170", "롯데케미칼",      "KOSPI"),
    ("018260", "삼성에스디에스",  "KOSPI"),
    ("010130", "고려아연",       "KOSPI"),
    ("000100", "유한양행",       "KOSPI"),
    ("352820", "하이브",         "KOSPI"),
    ("251270", "넷마블",         "KOSPI"),
    ("259960", "크래프톤",       "KOSPI"),
    ("247540", "에코프로비엠",    "KOSDAQ"),
    ("086520", "에코프로",        "KOSDAQ"),
    ("091990", "셀트리온헬스케어","KOSDAQ"),
    ("196170", "알테오젠",       "KOSDAQ"),
    ("357780", "솔브레인",       "KOSDAQ"),
    ("041510", "에스엠",         "KOSDAQ"),
    ("035900", "JYP Ent.",      "KOSDAQ"),
    ("122870", "와이지엔터테인먼트","KOSDAQ"),
    ("112040", "위메이드",       "KOSDAQ"),
    ("263750", "펄어비스",       "KOSDAQ"),
]

@st.cache_data(ttl=300)  # 5분 캐시
def get_kospi_status():
    """코스피 현재 지수 및 등락률"""
    try:
        k = yf.Ticker("^KS11")
        hist = k.history(period="5d")
        if len(hist) >= 2:
            current = float(hist['Close'].iloc[-1])
            prev    = float(hist['Close'].iloc[-2])
            change_pct = (current - prev) / prev * 100
            change_pt  = current - prev
            return current, change_pct, change_pt, hist
        return None, 0, 0, None
    except:
        return None, 0, 0, None

@st.cache_data(ttl=300)
def get_stock_today_change(ticker):
    """개별 종목 오늘 등락률"""
    try:
        s = yf.Ticker(ticker)
        hist = s.history(period="5d")
        if len(hist) >= 2:
            cur  = float(hist['Close'].iloc[-1])
            prev = float(hist['Close'].iloc[-2])
            chg  = (cur - prev) / prev * 100
            vol  = float(hist['Volume'].iloc[-1])
            vol_avg = float(hist['Volume'].mean())
            vol_ratio = vol / vol_avg if vol_avg > 0 else 1.0
            return cur, chg, vol_ratio
        return None, None, None
    except:
        return None, None, None

@st.cache_data(ttl=300)
def get_normalized_chart(ticker, kospi_hist):
    """코스피 vs 종목 정규화 비교 차트 데이터 (3개월)"""
    try:
        s = yf.Ticker(ticker)
        stock_hist = s.history(period="3mo")
        if stock_hist.empty:
            return None, None
        # 시작점 100으로 정규화
        stock_norm = stock_hist['Close'] / stock_hist['Close'].iloc[0] * 100
        return stock_hist.index, stock_norm
    except:
        return None, None


@st.cache_data(ttl=3600)
def get_defense_rate(ticker, period="2mo"):
    """
    2개월 동안 코스피 하락일에 해당 종목이 방어(상승+덜 하락)한 빈도수 계산
    Returns dict: total_down_days, defense_days, defense_rate, reverse_days, reverse_rate, avg_gap_down
    """
    try:
        k = yf.Ticker("^KS11")
        kospi_hist = k.history(period=period)
        s = yf.Ticker(ticker)
        stock_hist = s.history(period=period)

        if kospi_hist.empty or stock_hist.empty:
            return None

        # tz 제거 후 공통 날짜 정렬
        try:
            kospi_hist.index = kospi_hist.index.tz_localize(None)
        except Exception:
            kospi_hist.index = kospi_hist.index.tz_convert(None)
        try:
            stock_hist.index = stock_hist.index.tz_localize(None)
        except Exception:
            stock_hist.index = stock_hist.index.tz_convert(None)

        k_ret = kospi_hist['Close'].pct_change().dropna() * 100
        s_ret = stock_hist['Close'].pct_change().dropna() * 100

        common_idx = k_ret.index.intersection(s_ret.index)
        if len(common_idx) < 10:
            return None

        k_ret = k_ret[common_idx]
        s_ret = s_ret[common_idx]

        down_mask  = k_ret < 0
        total_down = int(down_mask.sum())

        if total_down == 0:
            return {'total_down_days': 0, 'defense_days': 0, 'defense_rate': 0.0,
                    'reverse_days': 0, 'reverse_rate': 0.0, 'avg_gap_down': 0.0, 'period': period}

        vs_on_down   = s_ret[down_mask] - k_ret[down_mask]
        defense_days = int((vs_on_down > 0).sum())
        reverse_days = int((s_ret[down_mask] > 0).sum())
        defense_rate = round(defense_days / total_down * 100, 1)
        reverse_rate = round(reverse_days / total_down * 100, 1)
        avg_gap      = round(float(vs_on_down.mean()), 2)

        return {
            'total_down_days': total_down,
            'defense_days':    defense_days,
            'defense_rate':    defense_rate,
            'reverse_days':    reverse_days,
            'reverse_rate':    reverse_rate,
            'avg_gap_down':    avg_gap,
            'period':          period,
        }
    except Exception:
        return None

def run_radar_scan(top50_list):
    """시장 레이더 스캔: 시총 상위 50종목 vs 코스피"""
    kospi_current, kospi_change, kospi_pt, kospi_hist = get_kospi_status()

    results = []
    progress_bar = st.progress(0)
    status_text  = st.empty()
    total = len(top50_list)

    for i, (code, name, market) in enumerate(top50_list):
        suffix = ".KS" if market == "KOSPI" else ".KQ"
        ticker = f"{code}{suffix}"

        progress_bar.progress((i + 1) / total)
        status_text.text(f"📡 스캔 중: {name} ({i+1}/{total})")

        price, chg, vol_ratio = get_stock_today_change(ticker)
        if price is None:
            continue

        vs_kospi = chg - kospi_change  # 코스피 대비 초과 수익률

        # 판정
        if chg > 0 and kospi_change < 0:
            verdict = "⭐ 역주행"
            verdict_color = "gold"
        elif vs_kospi > 1.0:
            verdict = "✅ 강한 방어"
            verdict_color = "limegreen"
        elif vs_kospi > 0:
            verdict = "🛡️ 방어"
            verdict_color = "lightgreen"
        elif vs_kospi > -1.0:
            verdict = "➖ 동행"
            verdict_color = "lightgray"
        else:
            verdict = "🔴 이탈"
            verdict_color = "salmon"

        results.append({
            "ticker":       ticker,
            "code":         code,
            "name":         name,
            "market":       market,
            "price":        price,
            "change_pct":   round(chg, 2),
            "vs_kospi":     round(vs_kospi, 2),
            "vol_ratio":    round(vol_ratio, 2),
            "verdict":      verdict,
            "verdict_color":verdict_color,
        })
        time.sleep(0.05)

    progress_bar.empty()
    status_text.empty()

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("vs_kospi", ascending=False).reset_index(drop=True)
        df.index += 1  # 순위 1부터
    return df, kospi_current, kospi_change, kospi_hist

# ========== UI ==========
st.title("📊 전문가급 주식 분석 시스템")
st.markdown("### 🇰🇷 한국 주식 전체 검색 + AI 기반 4대 모듈 분석")
st.markdown("**✅ 정교한 패턴 분석 (RSI+거래량) | ✅ 리스크 관리 시각화 | ✅ 투자 적합 종목 자동 추천 | ✅ 시장 레이더**")
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

# ========== 탭 1 | 시장 레이더, 탭 2 | 투자 적합, 탭 3 | 개별 분석 ==========
tab1, tab2, tab3 = st.tabs([
    "📡 시장 레이더",
    "🎯 투자 적합 종목 추천",
    "🔍 개별 종목 분석"
])

# ================================================
# TAB 1: 시장 레이더
# ================================================
with tab1:
    st.subheader("📡 오늘의 시장 레이더")
    st.markdown("""
    > 📌 **어떻게 활용할까요?**  
    > 코스피가 하락하는 날, 오히려 **상승** 하거나 **덜 떨어지는 종목**이 진짜 알짜배기 주식입니다.  
    > 시장 1위~50위 종목의 오늘 등락률을 코스피와 비교하여 **매일 관찰 훈련**을 시작하세요.
    """)

    # 코스피 실시간 지수
    kospi_val, kospi_chg, kospi_pt, kospi_hist_data = get_kospi_status()

    if kospi_val:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("📊 코스피 지수", f"{kospi_val:,.2f}",
                      delta=f"{kospi_pt:+.2f}pt ({kospi_chg:+.2f}%)",
                      delta_color="normal")
        with c2:
            if kospi_chg < -1.5:
                st.error("🔴 하락장 (역주행 종목을 주목)")
            elif kospi_chg < 0:
                st.warning("🟡 약하락 (\ubc29어주 주목)")
            elif kospi_chg > 1.5:
                st.success("🟢 강한 상승장")
            else:
                st.info("🟢 소폭 상승")
        with c3:
            st.metric("평가 기준", "75점 이상", "투자 적합")
        with c4:
            st.metric("스캔 대상", "시총상위 50종목", "스스로 학습")
    else:
        st.warning("⚠️ 코스피 지수 로드 실패. 잠시 훈 다시 시도하세요.")

    st.markdown("---")

    # 레이더 실행 버튼
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        radar_btn = st.button("📡 시장 레이더 스캔 실행 (시총 50위, 약 1-2분)",
                              type="primary", use_container_width=True)
    with col_r2:
        if st.button("🔄 레이더 새로고침", use_container_width=True):
            st.session_state.radar_results = None
            st.session_state.radar_kospi_change = None
            get_kospi_status.clear()
            get_stock_today_change.clear()
            st.rerun()

    if radar_btn:
        st.session_state.radar_results = None

    # 킱 해상도 지사용 여부 포소트 (레이더 ON 시)
    if st.session_state.radar_results is None and radar_btn:
        # 스캔 실행
        st.info("🔍 시총 상위 50종목 스캔 시작...")
        # FinanceDataReader로 실제 시총 상위 50 가져오기 (가능하면)
        top50_list = TOP50_FALLBACK  # 폴백 사용
        if all_stocks_df is not None and not all_stocks_df.empty:
            try:
                # FinanceDataReader는 시총 순으로 정렬된 경우가 많음
                top_df = all_stocks_df.head(50)
                top50_list = [
                    (str(r['Code']), str(r['Name']), str(r['Market']))
                    for _, r in top_df.iterrows()
                ]
            except:
                top50_list = TOP50_FALLBACK

        with st.spinner("📡 시장 레이더 스캔 중..."):
            radar_df, k_val, k_chg, k_hist = run_radar_scan(top50_list)

        st.session_state.radar_results = radar_df
        st.session_state.radar_kospi_change = k_chg
        st.session_state.radar_kospi_current = k_val
        st.rerun()

    # 레이더 결과 표시
    if st.session_state.radar_results is not None and not st.session_state.radar_results.empty:
        radar_df = st.session_state.radar_results
        k_chg    = st.session_state.radar_kospi_change or 0
        k_val    = st.session_state.radar_kospi_current or 0

        # 필터 탭
        filter_opt = st.radio(
            "포트폴리오 필터",
            ["전체 보기", "⭐ 역주행 + ✅ 방어만", "⭐ 역주행만"],
            horizontal=True
        )
        if filter_opt == "⭐ 역주행 + ✅ 방어만":
            view_df = radar_df[radar_df['verdict'].str.contains("역주행|방어")]
        elif filter_opt == "⭐ 역주행만":
            view_df = radar_df[radar_df['verdict'].str.contains("역주행")]
        else:
            view_df = radar_df

        st.markdown(f"""
        #### 포스피 ({k_val:,.2f}) 오늘 {k_chg:+.2f}% | 레이더 결과
        """)

        # 주요 테이블
        table_df = view_df[['name','market','price','change_pct','vs_kospi','vol_ratio','verdict']].copy()
        table_df.columns = ['종목명', '시장', '현재가', '등락률(%)', 'vs코스피(%p)', '거래량배율', '판정']
        table_df['현재가'] = table_df['현재가'].apply(lambda x: f"{x:,.0f}원")
        table_df['등락률(%)'] = table_df['등락률(%)'].apply(lambda x: f"{x:+.2f}%")
        table_df['vs코스피(%p)'] = table_df['vs코스피(%p)'].apply(lambda x: f"{x:+.2f}%p")
        table_df['거래량배율'] = table_df['거래량배율'].apply(lambda x: f"{x:.2f}배")
        st.dataframe(table_df, use_container_width=True, height=450)

        st.markdown("""**등급 설명**: ⭐ 역주행 = 코스피 하락인데 상승 | ✅ 강한 방어 = 코스피 +1%p 초과 | 🛡️ 방어 = 코스피보다 덜 하락 | ➖ 동행 | 🔴 이탈 = 코스피보다 -1%p 초과 하락""")

        st.markdown("---")

        # ====== 시각화 1: 코스피 vs 종목 등락률 스코터 차트 ======
        st.markdown("### 📊 코스피 vs 종목 등락률 비교")

        fig_scatter = go.Figure()

        colors = {
            "⭐ 역주행": "gold",
            "✅ 강한 방어": "limegreen",
            "🛡️ 방어": "lightgreen",
            "➖ 동행": "lightgray",
            "🔴 이탈": "salmon"
        }

        for verdict_type, color in colors.items():
            sub = view_df[view_df['verdict'] == verdict_type]
            if sub.empty:
                continue
            fig_scatter.add_trace(go.Scatter(
                x=[k_chg] * len(sub),
                y=sub['change_pct'],
                mode='markers+text',
                name=verdict_type,
                text=sub['name'],
                textposition='top center',
                marker=dict(size=14, color=color,
                            line=dict(color='white', width=1)),
                hovertemplate="%{text}<br>등락률: %{y:+.2f}%<extra></extra>"
            ))

        # 코스피 등락률 기준선
        fig_scatter.add_hline(y=k_chg, line_dash="dash", line_color="blue",
                              annotation_text=f"코스피 {k_chg:+.2f}%",
                              annotation_position="right")
        fig_scatter.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

        fig_scatter.update_layout(
            title="종목별 등락률 vs 코스피",
            xaxis_title="코스피 등락률 (%)",
            yaxis_title="종목 등락률 (%)",
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")

        # ====== 시각화 2: vs코스피 항목 막대 차트 (정렬된) ======
        st.markdown("### 🏆 코스피 대비 초과 수익률 순위")

        bar_df = view_df.sort_values('vs_kospi', ascending=True)
        bar_colors = [colors.get(v, 'lightgray') for v in bar_df['verdict']]

        fig_bar = go.Figure(go.Bar(
            x=bar_df['vs_kospi'],
            y=bar_df['name'],
            orientation='h',
            marker_color=bar_colors,
            text=bar_df['vs_kospi'].apply(lambda x: f"{x:+.2f}%p"),
            textposition='outside'
        ))
        fig_bar.add_vline(x=0, line_dash="solid", line_color="gray", line_width=2)
        fig_bar.update_layout(
            title="코스피 대비 초과/미달 수익률 (%p)",
            xaxis_title="vs 코스피 (%p)",
            height=max(400, len(bar_df) * 22),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # ====== 시각화 3: 코스피 vs 관심 종목 정규화 라인샵 ======
        st.markdown("### 📈 3개월 정규화 흐름 비교 (코스피 vs 선택 종목)")
        st.caption("코스피 시작점을 100으로 정규화하여, 시장보다 얼마나 강하게 움직였는지 확인합니다.")

        # 코스피 3개월 정규화
        if kospi_hist_data is not None and len(kospi_hist_data) >= 2:
            k3 = kospi_hist_data.tail(60) if len(kospi_hist_data) >= 60 else kospi_hist_data
            k_norm = k3['Close'] / k3['Close'].iloc[0] * 100

            # 선택에 노말할 종목 (vs코스피 상위 5개)
            top5_tickers = view_df.head(5)

            fig_norm = go.Figure()
            fig_norm.add_trace(go.Scatter(
                x=k3.index, y=k_norm,
                name='코스피',
                line=dict(color='red', width=3, dash='dash'),
                hovertemplate='코스피: %{y:.1f}<extra></extra>'
            ))

            palette = ['royalblue','orange','green','purple','brown']
            for ci, (_, row2) in enumerate(top5_tickers.iterrows()):
                idx_dates, s_norm = get_normalized_chart(row2['ticker'], kospi_hist_data)
                if idx_dates is None:
                    continue
                fig_norm.add_trace(go.Scatter(
                    x=idx_dates, y=s_norm,
                    name=row2['name'],
                    line=dict(color=palette[ci % 5], width=2),
                    hovertemplate=f"{row2['name']}: %{{y:.1f}}<extra></extra>"
                ))

            fig_norm.add_hline(y=100, line_dash='dot', line_color='gray',
                               annotation_text="시작점 100")
            fig_norm.update_layout(
                title="코스피 vs 상위 5종목 3개월 정규화 흐름",
                yaxis_title="정규화 지수 (100 = 시작점)",
                height=500
            )
            st.plotly_chart(fig_norm, use_container_width=True)

        st.markdown("---")

        # ====== 시각화 4: 2개월 방어율 분석 ======
        st.markdown("### 🛡️ 2개월 하락장 방어율 분석")
        st.caption(
            "코스피가 하락한 날 중, 이 종목이 코스피보다 선방(덜 하락 또는 오히려 상승)한 날의 비율입니다. "
            "높을수록 하락장에 강한 '알짜배기 주식'입니다."
        )

        with st.spinner("📊 2개월 방어율 계산 중... (종목당 약 1초)"):
            defense_rows = []
            tickers_to_analyze = view_df.head(20)['ticker'].tolist()
            names_to_analyze   = view_df.head(20)['name'].tolist()

            dprog = st.progress(0)
            for di, (dticker, dname) in enumerate(zip(tickers_to_analyze, names_to_analyze)):
                dprog.progress((di + 1) / len(tickers_to_analyze))
                dr = get_defense_rate(dticker, period="2mo")
                if dr:
                    # 방어력 등급
                    if dr['defense_rate'] >= 70:
                        d_grade = "⭐⭐⭐ 최강 방어"
                        d_color = "#28a745"
                    elif dr['defense_rate'] >= 55:
                        d_grade = "⭐⭐ 강한 방어"
                        d_color = "#85c720"
                    elif dr['defense_rate'] >= 40:
                        d_grade = "⭐ 보통 방어"
                        d_color = "#ffc107"
                    else:
                        d_grade = "🔴 취약"
                        d_color = "#dc3545"

                    defense_rows.append({
                        "종목명":        dname,
                        "코스피하락일":  dr['total_down_days'],
                        "방어성공일":    dr['defense_days'],
                        "방어율(%)":     dr['defense_rate'],
                        "역주행일":      dr['reverse_days'],
                        "역주행율(%)":   dr['reverse_rate'],
                        "평균초과(%p)":  dr['avg_gap_down'],
                        "방어등급":      d_grade,
                    })
            dprog.empty()

        if defense_rows:
            def_df = pd.DataFrame(defense_rows).sort_values("방어율(%)", ascending=False)

            # 방어율 컬러 스타일
            def color_defense(val):
                try:
                    v = float(val)
                    if v >= 70: return "background-color:#d4edda; color:#155724; font-weight:bold"
                    elif v >= 55: return "background-color:#fff3cd; color:#856404; font-weight:bold"
                    elif v >= 40: return "background-color:#fff8e1; color:#6d4c00"
                    else: return "background-color:#f8d7da; color:#721c24"
                except:
                    return ""

            def_styled = def_df.style.applymap(color_defense, subset=["방어율(%)"])
            st.dataframe(def_styled, use_container_width=True, height=400)

            # 방어율 수평 막대차트
            fig_def = go.Figure()
            fig_def.add_trace(go.Bar(
                y=def_df["종목명"],
                x=def_df["방어율(%)"],
                orientation='h',
                name="방어율",
                marker_color=[
                    "#28a745" if v >= 70 else "#85c720" if v >= 55 else "#ffc107" if v >= 40 else "#dc3545"
                    for v in def_df["방어율(%)"]
                ],
                text=[f"{v:.1f}%" for v in def_df["방어율(%)"]],
                textposition='outside',
            ))
            fig_def.add_trace(go.Bar(
                y=def_df["종목명"],
                x=def_df["역주행율(%)"],
                orientation='h',
                name="역주행율(코스피↓ + 종목↑)",
                marker_color="gold",
                text=[f"{v:.1f}%" for v in def_df["역주행율(%)"]],
                textposition='outside',
                visible='legendonly'
            ))
            fig_def.add_vline(x=50, line_dash="dash", line_color="gray",
                              annotation_text="50% 기준선", annotation_position="top")
            fig_def.update_layout(
                title="2개월 코스피 하락일 방어율 비교 (높을수록 하락장에 강한 종목)",
                xaxis_title="방어율 (%)",
                xaxis=dict(range=[0, 115]),
                height=max(350, len(def_df) * 28),
                barmode='overlay',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
            )
            st.plotly_chart(fig_def, use_container_width=True)

            st.info("""
**📌 방어율 해석 가이드**
- **방어율**: 코스피 하락일 중 해당 종목이 코스피보다 선방(덜 떨어지거나 오히려 상승)한 비율  
- **역주행율**: 코스피 하락일 중 해당 종목이 아예 상승한 비율 (더 희귀하고 더 가치 있음)  
- **평균초과(%p)**: 코스피 하락일 평균적으로 코스피보다 몇 %p 앞섰는지  
- ⭐⭐⭐ **70% 이상**: 최강 방어주 — 하락장에서도 굳건한 진짜 알짜배기  
- ⭐⭐ **55~69%**: 강한 방어주 — 시장보다 확실히 강함  
- ⭐ **40~54%**: 보통 — 시장 평균 수준  
- 🔴 **40% 미만**: 취약 — 하락장에서 시장보다 더 떨어지는 경향  
            """)
        else:
            st.warning("⚠️ 방어율 데이터를 가져올 수 없습니다.")

        st.markdown("---")

        # ====== 종목을 개별 분석으로 보내기 ======
        st.markdown("### 🔍 관심 종목 상세 분석")
        st.caption("아래 버튼을 클릭하면 '개별 종목 분석' 탭에서 상세 4대 모듈 분석으로 이동합니다.")

        btn_cols = st.columns(5)
        for ci, (_, row2) in enumerate(view_df.head(10).iterrows()):
            with btn_cols[ci % 5]:
                if st.button(
                    f"{row2['verdict']} {row2['name']}",
                    key=f"radar_detail_{row2['ticker']}",
                    use_container_width=True
                ):
                    st.session_state.current_ticker = row2['ticker']
                    st.rerun()

    elif st.session_state.radar_results is not None and st.session_state.radar_results.empty:
        st.warning("⚠️ 스캔 결과가 없습니다.")
    else:
        st.info("👆 '시장 레이더 스캔 실행' 버튼을 눌러주세요.")
        # 사용 안내
        st.markdown("""
        #### 📚 시장 레이더 활용 가이드
        | 판정 | 의미 | 전력 액션 |
        |------|------|----------|
        | ⭐ **역주행** | 코스피 하락인데 상승 | 최우선 관심 종목으로 등록 |
        | ✅ **강한 방어** | 코스피+1%p 이상 초과 | 관심 종목으로 등록 |
        | 🛡️ **방어** | 코스피보다 덜 하락 | 추세 지속 확인 |
        | ➖ **동행** | 코스피와 유사한 움직임 | 별도 판단 기준 필요 |
        | 🔴 **이탈** | 코스피보다 훨씩 더 하락 | 매수 시점 아님 |
        """)

# ================================================
# TAB 2: 투자 적합 종목 추천
# ================================================
with tab2:
    st.subheader("🎯 투자 적합 종목 추천")
    st.markdown("""
    > 📌 **M1** 추세·정배열 · **M2** 거래량 이상 · **M3** 4대 매수조건 · **M4** 리스크:리워드 · **M5** 시장 상대강도  
    > 시장 레이더(Tab 1)를 먼저 실행하면 **M5 시장 상대강도**가 자동 반영됩니다.
    """)

    # 점수 기준 인포박스
    st.info("""
    📊 **종합 점수 기준**  
    🟢 **75점 이상** : 강력 매수 추천 → 즉시 상세 분석  
    🟡 **55~74점** : 신중 매수 → 리스크 관리 필수  
    🔴 **55점 미만** : 매수 부적합 → 관망  
    """)

    # 레이더 연동 상태 알림
    radar_df_tab2 = st.session_state.get('radar_results', None)
    if radar_df_tab2 is not None and not radar_df_tab2.empty:
        st.success(f"✅ 시장 레이더 데이터 연동 완료 ({len(radar_df_tab2)}개 종목 · M5 모듈 활성화)")
    else:
        st.warning("⚠️ 시장 레이더 데이터 없음 → Tab 1에서 '시장 레이더 스캔'을 먼저 실행하면 M5 점수가 추가됩니다.")

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
        res_df = st.session_state.scan_results
        has_m5 = 'module5' in res_df.columns and res_df['module5'].notna().any()

        st.success(f"✅ 상위 {len(res_df)}개 종목 발견!  {'(M5 시장 상대강도 포함)' if has_m5 else '(M1~M4 기준)'}"
        )

        # ── 메인 결과 테이블 ────────────────────────────────────────────
        base_cols   = ['name', 'code', 'market', 'price', 'score', 'module1', 'module2', 'module3', 'module4']
        base_heads  = ['종목명', '코드', '시장', '현재가', '최종점수', 'M1(추세)', 'M2(거래량)', 'M3(매수조건)', 'M4(R:R)']

        if has_m5:
            extra_cols  = base_cols  + ['module5', 'vs_kospi', 'verdict']
            extra_heads = base_heads + ['M5(시장강도)', '코스피대비(%p)', '레이더판정']
        else:
            extra_cols  = base_cols
            extra_heads = base_heads

        display_df = res_df[extra_cols].copy()
        display_df.columns = extra_heads
        display_df['현재가'] = display_df['현재가'].apply(lambda x: f"{x:,.0f}원")

        # 점수 색상 하이라이트용 스타일
        def color_score(val):
            try:
                v = int(val)
                if v >= 75: return 'background-color:#d4edda; color:#155724; font-weight:bold'
                elif v >= 55: return 'background-color:#fff3cd; color:#856404; font-weight:bold'
                else: return 'background-color:#f8d7da; color:#721c24'
            except:
                return ''

        styled = display_df.style.applymap(color_score, subset=['최종점수'])
        st.dataframe(styled, use_container_width=True, height=400)

        # ── 상세 점수 카드 ────────────────────────────────────────────
        with st.expander("📊 종목별 상세 점수 분석 카드"):
            for i, (_, row) in enumerate(res_df.iterrows()):
                # NaN 안전 처리 (pandas Series는 None 대신 NaN 반환)
                verdict_val = row.get('verdict') if 'verdict' in row.index else None
                verdict_str = str(verdict_val) if verdict_val is not None and not (isinstance(verdict_val, float) and pd.isna(verdict_val)) else ''
                
                vs_val = row.get('vs_kospi') if 'vs_kospi' in row.index else None
                vs_is_valid = vs_val is not None and not (isinstance(vs_val, float) and pd.isna(vs_val))
                vs_str = f"{float(vs_val):+.2f}%p" if vs_is_valid else 'N/A'
                
                m5_val = row.get('module5') if 'module5' in row.index else None
                m5_is_valid = m5_val is not None and not (isinstance(m5_val, float) and pd.isna(m5_val))
                m5_str = f"{int(float(m5_val))}점" if m5_is_valid else 'N/A (레이더 미실행)'
                
                score = int(row['score'])
                badge = "🟢 강력 매수" if score >= 75 else "🟡 신중 매수" if score >= 55 else "🔴 관망"

                st.markdown(f"""
**{i+1}. {row['name']}** `{row['code']}` · {row['market']} &nbsp; {badge}
""")
                cols_card = st.columns([1.2, 1, 1, 1, 1, 1])
                cols_card[0].metric("최종점수",  f"{score}점")
                cols_card[1].metric("M1 추세",   f"{row['module1']}점")
                cols_card[2].metric("M2 거래량", f"{row['module2']}점")
                cols_card[3].metric("M3 조건",   f"{row['module3']}점")
                cols_card[4].metric("M4 R:R",    f"{row['module4']}점")
                cols_card[5].metric("M5 시장강도", m5_str)

                detail_cols = st.columns(4)
                detail_cols[0].write(f"📈 거래량: {row['volume_ratio']:.2f}배")
                detail_cols[1].write(f"✅ 조건: {row['conditions']}/4")
                detail_cols[2].write(f"⚖️ R:R = {row['rr_ratio']:.2f}:1")
                detail_cols[3].write(f"📡 코스피대비 {vs_str}  {verdict_str}")
                st.markdown("---")

        # ── 점수 비교 차트 (모듈별 누적 바) ──────────────────────────────
        st.markdown("### 📊 종목별 투자 적합도 점수 비교")

        chart_names = res_df['name'].tolist()
        m1_vals = res_df['module1'].tolist()
        m2_vals = res_df['module2'].tolist()
        m3_vals = res_df['module3'].tolist()
        m4_vals = res_df['module4'].tolist()

        if has_m5:
            m5_vals = [v if v is not None else 0 for v in res_df['module5'].tolist()]
        else:
            m5_vals = None

        fig_compare = go.Figure()
        # 총 점수 오버레이 (마커)
        fig_compare.add_trace(go.Scatter(
            x=chart_names,
            y=res_df['score'].tolist(),
            mode='markers+text',
            text=[f"<b>{s}</b>" for s in res_df['score'].tolist()],
            textposition='top center',
            marker=dict(
                size=14,
                color=['#28a745' if s >= 75 else '#ffc107' if s >= 55 else '#dc3545'
                       for s in res_df['score'].tolist()],
                symbol='diamond'
            ),
            name='최종점수',
            yaxis='y2'
        ))
        # 모듈별 누적 바
        weight_label = '(각 20%)' if has_m5 else '(각 25%)'
        fig_compare.add_trace(go.Bar(x=chart_names, y=m1_vals, name=f'M1 추세 {weight_label}',    marker_color='#4A90D9'))
        fig_compare.add_trace(go.Bar(x=chart_names, y=m2_vals, name=f'M2 거래량 {weight_label}',  marker_color='#50C878'))
        fig_compare.add_trace(go.Bar(x=chart_names, y=m3_vals, name=f'M3 조건 {weight_label}',    marker_color='#FF8C42'))
        fig_compare.add_trace(go.Bar(x=chart_names, y=m4_vals, name=f'M4 R:R {weight_label}',     marker_color='#9B59B6'))
        if m5_vals:
            fig_compare.add_trace(go.Bar(x=chart_names, y=m5_vals, name=f'M5 시장강도 (20%)', marker_color='#E74C3C'))

        fig_compare.add_hline(y=75, line_dash="dash", line_color="green",
                              annotation_text="🟢 강력 매수 (75점)", annotation_position="top right")
        fig_compare.add_hline(y=55, line_dash="dash", line_color="orange",
                              annotation_text="🟡 신중 매수 (55점)", annotation_position="top right")

        fig_compare.update_layout(
            title=f"투자 적합도 모듈별 점수 분해 {'(M5 시장강도 포함)' if has_m5 else '(M1~M4)'}",
            barmode='group',
            xaxis_title="종목명",
            yaxis_title="모듈 점수 (0-100)",
            yaxis2=dict(title='최종점수', overlaying='y', side='right', range=[0, 110]),
            height=520,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # ── 2개월 하락장 방어율 분석 (Tab2 추천 종목) ──────────────────
        st.markdown("---")
        st.markdown("### 🛡️ 2개월 하락장 방어율 분석")
        st.caption(
            "코스피가 하락한 날 중, 추천 종목들이 코스피보다 선방(덜 하락 또는 오히려 상승)한 날의 비율입니다. "
            "높을수록 하락장에 강한 '알짜배기 주식'입니다."
        )

        with st.spinner("📊 2개월 방어율 계산 중... (종목당 약 1초)"):
            tab2_defense_rows = []
            for _, drow in res_df.iterrows():
                dticker2 = drow['ticker']
                dname2   = drow['name']
                dr2 = get_defense_rate(dticker2, period="2mo")
                if dr2['total_down_days'] > 0:
                    if dr2['defense_rate'] >= 70:
                        dgrade2 = "⭐⭐⭐ 최강 방어"
                    elif dr2['defense_rate'] >= 55:
                        dgrade2 = "⭐⭐ 강한 방어"
                    elif dr2['defense_rate'] >= 40:
                        dgrade2 = "⭐ 보통"
                    else:
                        dgrade2 = "🔴 취약"
                    tab2_defense_rows.append({
                        "종목명":       dname2,
                        "코스피하락일": dr2['total_down_days'],
                        "방어성공일":   dr2['defense_days'],
                        "방어율(%)":    dr2['defense_rate'],
                        "역주행일":     dr2['reverse_days'],
                        "역주행율(%)":  dr2['reverse_rate'],
                        "평균초과(%p)": dr2['avg_gap_down'],
                        "방어등급":     dgrade2,
                    })

        if tab2_defense_rows:
            tab2_def_df = pd.DataFrame(tab2_defense_rows).sort_values("방어율(%)", ascending=False)

            def color_defense_tab2(val):
                try:
                    v = float(val)
                    if v >= 70:   return 'background-color:#d4edda; color:#155724; font-weight:bold'
                    elif v >= 55: return 'background-color:#d1ecf1; color:#0c5460; font-weight:bold'
                    elif v >= 40: return 'background-color:#fff3cd; color:#856404'
                    else:         return 'background-color:#f8d7da; color:#721c24'
                except:
                    return ''

            tab2_def_styled = tab2_def_df.style.applymap(color_defense_tab2, subset=["방어율(%)"])
            st.dataframe(tab2_def_styled, use_container_width=True)

            fig_def2 = go.Figure()
            fig_def2.add_trace(go.Bar(
                y=tab2_def_df["종목명"],
                x=tab2_def_df["방어율(%)"],
                orientation='h',
                name="방어율(코스피 선방)",
                marker_color=[
                    "#28a745" if v >= 70 else "#17a2b8" if v >= 55 else "#ffc107" if v >= 40 else "#dc3545"
                    for v in tab2_def_df["방어율(%)"]
                ],
                text=[f"{v:.1f}%" for v in tab2_def_df["방어율(%)"]],
                textposition='outside',
            ))
            fig_def2.add_trace(go.Bar(
                y=tab2_def_df["종목명"],
                x=tab2_def_df["역주행율(%)"],
                orientation='h',
                name="역주행율(코스피↓ + 종목↑)",
                marker_color="gold",
                text=[f"{v:.1f}%" for v in tab2_def_df["역주행율(%)"]],
                textposition='outside',
                visible='legendonly'
            ))
            fig_def2.add_vline(x=50, line_dash="dash", line_color="gray",
                               annotation_text="50% 기준선", annotation_position="top")
            fig_def2.update_layout(
                title="추천 종목 2개월 코스피 하락일 방어율 비교",
                xaxis_title="방어율 (%)",
                xaxis=dict(range=[0, 115]),
                height=max(350, len(tab2_def_df) * 30),
                barmode='overlay',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
            )
            st.plotly_chart(fig_def2, use_container_width=True)

            st.info("""
**📌 방어율 해석 가이드**
- **방어율**: 코스피 하락일 중 해당 종목이 코스피보다 선방(덜 떨어지거나 오히려 상승)한 비율  
- **역주행율**: 코스피 하락일 중 해당 종목이 아예 상승한 비율 (더 희귀하고 더 가치 있음)  
- ⭐⭐⭐ **70% 이상**: 최강 방어주 — 하락장에서도 굳건한 진짜 알짜배기  
- ⭐⭐ **55~69%**: 강한 방어주 — 시장보다 확실히 강함  
- ⭐ **40~54%**: 보통 — 시장 평균 수준  
- 🔴 **40% 미만**: 취약 — 하락장에서 시장보다 더 떨어지는 경향  
            """)
        else:
            st.warning("⚠️ 방어율 데이터를 가져올 수 없습니다.")

        # ── 초기화 버튼 ──────────────────────────────────────────────
        if st.button("🔄 다시 검색", key="scan_reset_btn"):
            st.session_state.scan_mode = None
            st.session_state.scan_results = None
            st.rerun()

    elif st.session_state.scan_mode and st.session_state.scan_results is not None and st.session_state.scan_results.empty:
        st.warning("⚠️ 조건에 맞는 종목이 없습니다. 기준을 낮춰보세요.")

# ================================================
# TAB 3: 개별 종목 분석
# ================================================
with tab3:
    st.subheader("🔍 개별 종목 분석")

    # current_ticker가 설정되어 있으면 바로 분석
    if st.session_state.current_ticker:
        final_ticker = st.session_state.current_ticker
        # 세션 초기화 (다음 검색을 위해)
        st.session_state.current_ticker = None
    else:
        # 검색
        st.markdown("""
        <div style='background:#EBF5FB; border-left:4px solid #2E86C1;
             padding:10px 16px; border-radius:6px; margin-bottom:12px;'>
        🔎 <b>종목명</b> 또는 <b>종목코드</b>로 검색하세요<br>
        <span style='font-size:12px; color:#555;'>
        • 이름 검색: <code>파미셀</code> / <code>삼성</code> / <code>카카오</code><br>
        • 코드 검색: <code>005930</code> / <code>35720</code><br>
        • 오타여도 유사 종목을 자동으로 추천해 드립니다 ✨
        </span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "종목 검색",
                placeholder="예) 파미셀, 삼성전자, 005930, 카카오뱅크...",
                key="stock_search_input",
                label_visibility="collapsed"
            )
        with col2:
            search_btn = st.button("🔎 검색", type="primary", use_container_width=True)

        if not query:
            st.info("👆 종목명이나 코드를 입력 후 검색 버튼을 눌러주세요")
            with st.expander("💡 자주 찾는 종목 예시"):
                quick_stocks = [
                    ("005930","삼성전자","KOSPI"), ("000660","SK하이닉스","KOSPI"),
                    ("035420","NAVER","KOSPI"),   ("035720","카카오","KOSPI"),
                    ("207940","삼성바이오로직스","KOSPI"), ("051910","LG화학","KOSPI"),
                    ("068270","셀트리온","KOSPI"), ("028260","삼성물산","KOSPI"),
                    ("066570","LG전자","KOSPI"),  ("015760","한국전력","KOSPI"),
                    ("091990","셀트리온헬스케어","KOSDAQ"), ("196170","알테오젠","KOSDAQ"),
                    ("293490","카카오게임즈","KOSDAQ"), ("112040","위메이드","KOSDAQ"),
                    ("264900","파미셀","KOSDAQ"),
                ]
                qcols = st.columns(3)
                for qi, (code, name, mkt) in enumerate(quick_stocks):
                    with qcols[qi % 3]:
                        qticker = f"{code}.KS" if mkt=='KOSPI' else f"{code}.KQ"
                        if st.button(f"{name}", key=f"quick_{code}", use_container_width=True):
                            st.session_state.current_ticker = qticker
                            st.rerun()
            st.stop()

        if not search_btn:
            st.stop()

        # 검색 실행
        with st.spinner("🔍 검색 중..."):
            ticker, matches, search_type = search_stock(query, stock_dict, all_stocks_df)

        # ── 검색 결과 처리 ─────────────────────────────────────────────
        if ticker:
            # 정확히 1개 찾음 → 바로 분석 시작
            final_ticker = ticker

        elif matches is not None and len(matches) > 0:
            if search_type == 'partial':
                st.info(f"🔍 **'{query}'** 포함 종목 {len(matches)}개를 찾았습니다. 분석할 종목을 선택해 주세요.")
            elif search_type == 'similar':
                st.warning(f"💡 **'{query}'** 와(과) 정확히 일치하는 종목이 없습니다.\n혹시 이 종목을 찾으셨나요?")

            # 종목 카드 그리드 (한 줄에 2개)
            n_cols = 2
            rows_iter = [matches.iloc[i:i+n_cols] for i in range(0, len(matches), n_cols)]
            for row_group in rows_iter:
                card_cols = st.columns(n_cols)
                for ci, (_, row) in enumerate(row_group.iterrows()):
                    code   = str(row['Code'])
                    name   = str(row['Name'])
                    market = str(row['Market'])
                    ticker_code = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
                    with card_cols[ci]:
                        sim_label = ""
                        if search_type == 'similar':
                            sim_score = difflib.SequenceMatcher(
                                None, query.strip().lower(), name.lower()
                            ).ratio()
                            sim_label = f" &nbsp; `유사도 {sim_score*100:.0f}%`"
                        st.markdown(
                            f"""<div style='border:1px solid #ddd; border-radius:8px;
                                padding:10px 14px; margin-bottom:6px;
                                background:#f8f9fa;'>
                            <b style='font-size:15px'>{name}</b>{sim_label}<br>
                            <span style='color:#666; font-size:12px'>코드: {code} &nbsp;|&nbsp; {market}</span>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        if st.button(f"📊 {name} 분석",
                                     key=f"sel_{code}_{ci}",
                                     use_container_width=True,
                                     type="primary"):
                            st.session_state.current_ticker = ticker_code
                            st.rerun()

            st.stop()

        else:
            # 완전 검색 실패 → 마지막 시도: difflib cutoff 더 낮게
            last_try = []
            if all_stocks_df is not None:
                names_lower = all_stocks_df['Name'].str.lower().str.strip()
                last_try = difflib.get_close_matches(
                    query.strip().lower(), names_lower.tolist(), n=6, cutoff=0.25
                )

            if last_try:
                st.error(f"❌ **'{query}'** 종목을 찾지 못했습니다.")
                st.info("💡 혹시 이런 종목을 찾으셨나요?")
                last_mask = names_lower.isin(last_try)
                last_df = all_stocks_df[last_mask].head(6).reset_index(drop=True)
                lc = st.columns(min(3, len(last_df)))
                for ci, (_, row) in enumerate(last_df.iterrows()):
                    code   = str(row['Code'])
                    name   = str(row['Name'])
                    market = str(row['Market'])
                    ticker_code = f"{code}.KS" if market == 'KOSPI' else f"{code}.KQ"
                    with lc[ci % 3]:
                        if st.button(f"🔍 {name}\n({code})",
                                     key=f"lastres_{code}",
                                     use_container_width=True):
                            st.session_state.current_ticker = ticker_code
                            st.rerun()
            else:
                st.error(f"❌ **'{query}'** 종목을 찾지 못했습니다.")
                st.info("💡 **검색 팁:** 종목코드(6자리 숫자)로 검색하면 더 정확합니다.\n예: 삼성전자 → 005930, 카카오 → 035720")

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
        st.session_state.current_ticker = None
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
        # 연속형 점수: 거래량 배율에 비례 (20~95)
        raw_vscore = min(95, max(20, volume_ratio * 38 + 10))
        if volume_ratio >= 3.0:
            breakout = "폭발적 폭등 🟢🟢"
            volume_score = 95
        elif volume_ratio >= 2.0:
            breakout = "높은 폭등 🟢"
            volume_score = min(95, raw_vscore + 5)
        elif volume_ratio >= 1.5:
            breakout = "중상 폭등 🟢"
            volume_score = raw_vscore
        elif volume_ratio >= 1.0:
            breakout = "보통 🟡"
            volume_score = raw_vscore
        else:
            breakout = "낮음 🔴"
            volume_score = raw_vscore
    else:
        volume_ratio = 0
        breakout = "부족"
        volume_score = 45

    module2_score = int(volume_score)

    # 모듈 3 (세분화: 조건 품질 보너스 포함)
    cond1 = current_price > ma120
    high_20d = float(hist['Close'].tail(20).max())
    low_20d  = float(hist['Close'].tail(20).min())
    cond2 = current_price >= high_20d * 0.95   # 20일 고점 근처

    if len(hist) >= 2:
        pct_chg = ((current_price - float(hist['Close'].iloc[-2])) / float(hist['Close'].iloc[-2])) * 100
        cond3 = -2 <= pct_chg <= 15
    else:
        pct_chg = 0
        cond3 = False

    cond4 = volume_ratio >= 1.5
    satisfied = sum([cond1, cond2, cond3, cond4])

    # 기본 점수
    base3 = {4: 80, 3: 63, 2: 46, 1: 30, 0: 15}[satisfied]
    # 품질 보너스 (최대 20점)
    qbonus = 0
    if cond1 and ma120 > 0:
        qbonus += min(5, (current_price - ma120) / ma120 * 100 * 0.5)
    if high_20d > low_20d:
        qbonus += min(5, (current_price - low_20d) / (high_20d - low_20d) * 5)
    if cond3 and 3 <= pct_chg <= 10:
        qbonus += 5
    elif cond3 and 0 < pct_chg < 3:
        qbonus += 3
    if cond4:
        qbonus += min(5, (volume_ratio - 1.5) * 2.5)
    module3_score = min(100, int(base3 + qbonus))

    # 모듈 4 (연속형 R:R 점수)
    sl_methods = {
        'open': float(latest['Open']),
        'low':  float(latest['Low']),
        '3pct': current_price * 0.97,
        'ma20': ma20
    }
    final_sl = max(sl_methods.values())
    risk_pct   = ((final_sl - current_price) / current_price) * 100
    target     = current_price + abs(current_price - final_sl) * 2
    reward_pct = ((target - current_price) / current_price) * 100

    risk_reward_ratio = abs(reward_pct / risk_pct) if risk_pct != 0 else 1.0
    # 연속형 점수: R:R에 비례 (20~95)
    module4_score = min(95, max(20, int(risk_reward_ratio * 28 + 18)))

    # ── 모듈 5: 시장 레이더 연동 ────────────────────────────────────────
    module5_score  = None
    m5_vs_kospi    = None
    m5_verdict     = None
    m5_kospi_chg   = None
    m5_stock_chg   = None

    radar_df_tab3 = st.session_state.get('radar_results', None)
    if radar_df_tab3 is not None and not radar_df_tab3.empty and 'ticker' in radar_df_tab3.columns:
        matched = radar_df_tab3[radar_df_tab3['ticker'] == final_ticker]
        if not matched.empty:
            r = matched.iloc[0]
            m5_vs_kospi  = float(r['vs_kospi'])   if pd.notna(r.get('vs_kospi'))  else None
            m5_verdict   = str(r['verdict'])       if pd.notna(r.get('verdict'))   else None
            m5_stock_chg = float(r['change_pct'])  if pd.notna(r.get('change_pct'))else None
            m5_kospi_chg = float(st.session_state.get('radar_kospi_change', 0) or 0)

            if m5_vs_kospi is not None:
                if m5_verdict == '⭐ 역주행':
                    module5_score = 100
                elif m5_vs_kospi > 3.0:
                    module5_score = 95
                elif m5_vs_kospi > 2.0:
                    module5_score = 85
                elif m5_vs_kospi > 1.0:
                    module5_score = 75
                elif m5_vs_kospi > 0:
                    module5_score = 63
                elif m5_vs_kospi > -1.0:
                    module5_score = 50
                elif m5_vs_kospi > -2.0:
                    module5_score = 35
                else:
                    module5_score = 20

    # **최종 점수**
    if module5_score is not None:
        final_score = int(
            module1_score * 0.20 +
            module2_score * 0.20 +
            module3_score * 0.20 +
            module4_score * 0.20 +
            module5_score * 0.20
        )
        score_formula = (f"({module1_score}×0.20) + ({module2_score}×0.20) + ({module3_score}×0.20) + "
                         f"({module4_score}×0.20) + ({module5_score}×0.20) = **{final_score}점** (M5 포함)")
    else:
        final_score = int(module1_score * 0.25 + module2_score * 0.25 + module3_score * 0.25 + module4_score * 0.25)
        score_formula = (f"({module1_score}×0.25) + ({module2_score}×0.25) + ({module3_score}×0.25) + "
                         f"({module4_score}×0.25) = **{final_score}점**")

    # ========== 상세 분석 UI ==========
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

    - **거래량 급증**: 평균 대비 2배 이상 = 강한 매수/매도 세력 유입
    - **거래량 감소**: 조정 시 거래량 감소 = 건전한 조정 (긍정)
    - **유동성 검증**: 일평균 거래대금이 충분한지 확인

    **점수**: 2.0배↑=90점 | 1.5~2.0배=75점 | 1.2~1.5배=60점 | 0.8~1.2배=45점 | 0.8배↓=25점
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
    4가지 핵심 조건을 충족하면 강력한 매수 신호로 판단합니다.

    1. **120일선 상회**: 중장기 상승 추세 확인
    2. **20일 신고가**: 단기 모멘텀 확보
    3. **최적 변동폭 (-2~15%)**: 과열되지 않은 건전한 상승
    4. **거래량 급증 (1.5배↑)**: 강한 매수 세력 유입

    **점수**: 4개=100점 | 3개=80점 | 2개=60점 | 1개=40점 | 0개=20점
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


    st.markdown("---")

    # 모듈 게이지 (M5 유무에 따라 동적 표시)
    if module5_score is not None:
        st.subheader("📊 5대 모듈 종합 점수 (M5 시장 상대강도 포함)")
        fig_modules = make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=("모듈1: 추세&패턴", "모듈2: 거래량", "모듈3: 매수신호",
                            "모듈4: 리스크", "모듈5: 시장강도", "")
        )
        gauge_data = [
            (module1_score, 1, 1, "darkblue"),
            (module2_score, 1, 2, "darkorange"),
            (module3_score, 1, 3, "darkgreen"),
            (module4_score, 2, 1, "darkviolet"),
            (module5_score, 2, 2, "#E74C3C"),
        ]
        h_mod = 500
    else:
        st.subheader("📊 4대 모듈 종합 점수")
        fig_modules = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=("모듈1: 추세&패턴", "모듈2: 거래량", "모듈3: 매수신호", "모듈4: 리스크")
        )
        gauge_data = [
            (module1_score, 1, 1, "darkblue"),
            (module2_score, 1, 2, "darkorange"),
            (module3_score, 2, 1, "darkgreen"),
            (module4_score, 2, 2, "darkviolet"),
        ]
        h_mod = 600

    for (sv, r, c, color) in gauge_data:
        fig_modules.add_trace(go.Indicator(
            mode="gauge+number",
            value=sv,
            title={'text': f"{sv}점"},
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
        ), row=r, col=c)

    fig_modules.update_layout(height=h_mod, showlegend=False)
    st.plotly_chart(fig_modules, use_container_width=True)

    # ── M5 시장 레이더 분석 박스 ────────────────────────────────────────
    st.subheader("📡 모듈 5: 시장 상대강도 (코스피 대비)")
    st.info("""
    **📚 모듈 5란?**  
    코스피가 하락하는 날, 오히려 상승하거나 덜 떨어지는 종목이 진짜 알짜배기 주식입니다.  
    Tab 1(시장 레이더)에서 스캔을 먼저 실행하면 이 종목의 시장 상대강도 데이터가 자동 연동됩니다.
    """)

    if module5_score is not None:
        m5c1, m5c2, m5c3, m5c4 = st.columns(4)
        m5c1.metric("📡 M5 점수", f"{module5_score}점")
        m5c2.metric("코스피 대비", f"{m5_vs_kospi:+.2f}%p" if m5_vs_kospi is not None else 'N/A')
        m5c3.metric("오늘 이 종목", f"{m5_stock_chg:+.2f}%" if m5_stock_chg is not None else 'N/A')
        m5c4.metric("코스피 변동", f"{m5_kospi_chg:+.2f}%" if m5_kospi_chg is not None else 'N/A')

        verdict_info = {
            '⭐ 역주행': ('🟢', '시장 하락 중 상승 → 최우선 관심 종목'),
            '✅ 강한 방어': ('🟢', '코스피 대비 +1%p 초과 → 강한 주식'),
            '🛡️ 방어': ('🟡', '코스피보다 덜 하락 → 방어력 있음'),
            '➖ 동행': ('⚪', '시장과 비슷한 수준'),
            '🔴 이탈': ('🔴', '코스피보다 크게 하락 → 주의')
        }.get(m5_verdict, ('⚪', '데이터 없음'))

        st.markdown(f"""
| 항목 | 내용 |
|------|------|
| 레이더 판정 | {verdict_info[0]} **{m5_verdict}** |
| 진단 | {verdict_info[1]} |
| M5 점수 | **{module5_score}점** (Tab 1 레이더 데이터 연동) |
| 코스피 대비 실적 | {f'{m5_vs_kospi:+.2f}%p' if m5_vs_kospi is not None else 'N/A'} |
        """)
    else:
        st.warning("⚠️ Tab 1(시장 레이더)에서 스캔을 먼저 실행하시면 M5 점수가 추가됩니다.")

    st.markdown("---")

    # ── 2개월 하락장 방어율 분석 (Tab3 개별 종목) ──────────────────
    st.markdown("### 🛡️ 2개월 하락장 방어율 분석")
    st.caption(
        "코스피가 하락한 날 중, 이 종목이 코스피보다 선방(덜 하락 또는 오히려 상승)한 날의 비율입니다. "
        "높을수록 하락장에 강한 '알짜배기 주식'입니다."
    )

    with st.spinner("📊 2개월 방어율 계산 중..."):
        dr_tab3 = get_defense_rate(final_ticker, period="2mo")

    if dr_tab3['total_down_days'] > 0:
        dr3_defense_rate  = dr_tab3['defense_rate']
        dr3_reverse_rate  = dr_tab3['reverse_rate']
        dr3_total_down    = dr_tab3['total_down_days']
        dr3_defense_days  = dr_tab3['defense_days']
        dr3_reverse_days  = dr_tab3['reverse_days']
        dr3_avg_gap       = dr_tab3['avg_gap_down']

        if dr3_defense_rate >= 70:
            dr3_grade = "⭐⭐⭐ 최강 방어주"
            dr3_color = "success"
        elif dr3_defense_rate >= 55:
            dr3_grade = "⭐⭐ 강한 방어주"
            dr3_color = "info"
        elif dr3_defense_rate >= 40:
            dr3_grade = "⭐ 보통"
            dr3_color = "warning"
        else:
            dr3_grade = "🔴 취약"
            dr3_color = "error"

        # 지표 카드 (4컬럼)
        dr3_c1, dr3_c2, dr3_c3, dr3_c4 = st.columns(4)
        dr3_c1.metric("🛡️ 방어율",    f"{dr3_defense_rate:.1f}%",
                      help="코스피 하락일 중 이 종목이 코스피보다 선방한 비율")
        dr3_c2.metric("⭐ 역주행율",   f"{dr3_reverse_rate:.1f}%",
                      help="코스피 하락일 중 이 종목이 아예 상승한 비율")
        dr3_c3.metric("📅 코스피 하락일", f"{dr3_total_down}일 중 {dr3_defense_days}일 방어")
        dr3_c4.metric("📊 평균초과(%p)", f"{dr3_avg_gap:+.2f}%p",
                      help="코스피 하락일에 평균적으로 코스피보다 몇 %p 앞섰는지")

        # 등급 표시
        getattr(st, dr3_color)(f"### {dr3_grade}")

        # 방어율 게이지 바 차트
        fig_dr3 = go.Figure()
        fig_dr3.add_trace(go.Bar(
            x=["방어율(%)", "역주행율(%)"],
            y=[dr3_defense_rate, dr3_reverse_rate],
            marker_color=["#28a745" if dr3_defense_rate >= 70 else "#17a2b8" if dr3_defense_rate >= 55 else "#ffc107" if dr3_defense_rate >= 40 else "#dc3545",
                          "gold"],
            text=[f"{dr3_defense_rate:.1f}%", f"{dr3_reverse_rate:.1f}%"],
            textposition='outside',
        ))
        fig_dr3.add_hline(y=50, line_dash="dash", line_color="gray",
                          annotation_text="50% 기준선", annotation_position="right")
        fig_dr3.add_hline(y=70, line_dash="dot", line_color="green",
                          annotation_text="⭐⭐⭐ 최강 70%", annotation_position="right")
        fig_dr3.update_layout(
            title=f"{company_name} 2개월 하락장 방어율",
            yaxis_title="비율 (%)",
            yaxis=dict(range=[0, 110]),
            height=320,
        )
        st.plotly_chart(fig_dr3, use_container_width=True)

        st.markdown(f"""
| 항목 | 값 |
|------|------|
| 🛡️ 방어율 | **{dr3_defense_rate:.1f}%** ({dr3_defense_days}일 / {dr3_total_down}일) |
| ⭐ 역주행율 | **{dr3_reverse_rate:.1f}%** ({dr3_reverse_days}일) |
| 📊 평균 초과수익(%p) | **{dr3_avg_gap:+.2f}%p** |
| 🏆 방어 등급 | **{dr3_grade}** |
        """)

        st.info("""
**📌 방어율 해석 가이드**
- **방어율**: 코스피 하락일 중 해당 종목이 코스피보다 선방(덜 떨어지거나 오히려 상승)한 비율  
- **역주행율**: 코스피 하락일 중 해당 종목이 아예 상승한 비율 (더 희귀하고 더 가치 있음)  
- ⭐⭐⭐ **70% 이상**: 최강 방어주 — 하락장에서도 굳건한 진짜 알짜배기  
- ⭐⭐ **55~69%**: 강한 방어주 — 시장보다 확실히 강함  
- ⭐ **40~54%**: 보통 — 시장 평균 수준  
- 🔴 **40% 미만**: 취약 — 하락장에서 시장보다 더 떨어지는 경향  
        """)
    else:
        st.warning("⚠️ 2개월 방어율 데이터를 불러올 수 없습니다. (데이터 부족 또는 네트워크 오류)")

    st.markdown("---")

    # 최종 평가
    st.header("🏆 최종 종합 평가")

    st.info("""
    **📊 평가 기준**

    - 🟢 **75점 이상**: 강력 매수 추천 → 모든 지표가 긍정적, 적극적 매수 고려
    - 🟡 **55~74점**: 신중 매수 → 일부 지표가 긍정적, 리스크 관리 필수
    - 🔴 **55점 미만**: 매수 부적합 → 현재 매수 시점이 아님, 관망 권장
    """)

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
        st.markdown("**모든 지표가 긍정적입니다. 적극적인 매수를 고려하세요.**")
    elif final_score >= 55:
        st.warning("### 🟡 신중 매수")
        st.markdown("**일부 지표가 긍정적입니다. 리스크 관리를 철저히 하세요.**")
    else:
        st.error("### 🔴 매수 부적합")
        st.markdown("**현재 매수 시점이 아닙니다. 관망을 권장합니다.**")

    # 기여도 테이블
    st.markdown("### 📊 모듈별 기여도")
    if module5_score is not None:
        w = 0.20
        mods   = ['M1 추세&패턴', 'M2 거래량', 'M3 매수조건', 'M4 R:R리스크', 'M5 시장강도']
        scores = [module1_score, module2_score, module3_score, module4_score, module5_score]
    else:
        w = 0.25
        mods   = ['M1 추세&패턴', 'M2 거래량', 'M3 매수조건', 'M4 R:R리스크']
        scores = [module1_score, module2_score, module3_score, module4_score]

    contrib_df = pd.DataFrame({
        '모듈':  mods,
        '점수':  scores,
        '가중치': [f"{int(w*100)}%"] * len(mods),
        '기여도': [f"{s * w:.1f}점" for s in scores]
    })
    st.dataframe(contrib_df, use_container_width=True)

    st.markdown(f"""
    **최종 점수**: {score_formula}
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
