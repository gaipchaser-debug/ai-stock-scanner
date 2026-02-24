import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.spatial.distance import cosine
from scipy.stats import zscore
import plotly.graph_objects as go
import google.generativeai as genai
import os
import warnings
from datetime import datetime, timedelta

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False

# --- [환경 설정 및 API 키] ---
try:
    NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
    NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
    DART_API_KEY = st.secrets["DART_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception as e:
    st.error(f"⚠️ API 키 설정이 필요합니다. Streamlit Cloud의 Secrets을 확인하세요.")
    st.stop()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ✅ [세션 상태 초기화]
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = None
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "last_input" not in st.session_state:
    st.session_state.last_input = None

# --- [1단계] 데이터 수집 ---
def get_stock_data(ticker_code, yf_ticker, days_ago=0):
    """
    주가 데이터 조회
    days_ago: 0이면 현재, 7이면 7일 전 데이터
    """
    try:
        if FDR_AVAILABLE:
            try:
                end_date = datetime.now() - timedelta(days=days_ago)
                start_date = end_date - timedelta(days=30)
                
                # ✅ end 파라미터 추가
                df = fdr.DataReader(ticker_code, 
                                   start=start_date.strftime('%Y-%m-%d'),
                                   end=end_date.strftime('%Y-%m-%d'))
                
                if not df.empty:
                    current_price = int(df['Close'].iloc[-1])
                    last_date = df.index[-1].strftime('%Y-%m-%d')
                    
                    return {
                        "price": current_price,
                        "date": last_date,
                        "source": "네이버 금융",
                        "history": df
                    }
            except Exception as e:
                pass
        
        # Yahoo Finance 백업
        end_date = datetime.now() - timedelta(days=days_ago)
        df = yf.download(yf_ticker, start=(end_date - timedelta(days=30)).strftime('%Y-%m-%d'), 
                        end=end_date.strftime('%Y-%m-%d'), progress=False)
        if df.empty:
            return {"price": 0, "date": "N/A", "source": "N/A", "history": None}
        
        current_price = int(df['Close'].iloc[-1])
        last_date = df.index[-1].strftime('%Y-%m-%d')
        
        return {
            "price": current_price,
            "date": last_date,
            "source": "Yahoo Finance",
            "history": df
        }
        
    except Exception as e:
        return {"price": 0, "date": "N/A", "source": "오류", "history": None}

def get_news_data(company_name):
    """최신 뉴스 조회 (제목 + 본문 요약)"""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return []
    try:
        search_query = f"{company_name} 주가"
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID, 
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        params = {"query": search_query, "display": 10, "sort": "date"}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            items = response.json().get('items', [])
            
            filtered = []
            for item in items:
                clean_title = item['title'].replace("<b>","").replace("</b>","")
                clean_desc = item.get('description', '').replace("<b>","").replace("</b>","")
                link = item.get('link', '#')
                
                if company_name in clean_title:
                    filtered.append({
                        "title": clean_title, 
                        "link": link,
                        "description": clean_desc  # ✅ 본문 요약 추가
                    })
            
            if len(filtered) >= 5:
                return filtered[:5]
            else:
                for item in items:
                    if len(filtered) >= 5:
                        break
                    clean_title = item['title'].replace("<b>","").replace("</b>","")
                    clean_desc = item.get('description', '').replace("<b>","").replace("</b>","")
                    link = item.get('link', '#')
                    
                    if not any(n['title'] == clean_title for n in filtered):
                        filtered.append({
                            "title": clean_title, 
                            "link": link,
                            "description": clean_desc  # ✅ 본문 요약 추가
                        })
                
                return filtered[:5]
        else:
            return []
    except Exception as e:
        return []

# --- [2단계] AI 검증 (본문 포함) ---
def analyze_validity(ticker, news_list):
    """AI 분석 (뉴스 제목 + 본문 내용)"""
    if not GEMINI_API_KEY:
        return {"correlation_score": 50, "reason": "Gemini API 키가 설정되지 않았습니다."}
    
    if not news_list:
        return {"correlation_score": 50, "reason": "최신 뉴스가 없어 중립적으로 평가합니다."}

    # ✅ 뉴스 제목 + 본문 요약 구성
    news_content = []
    for n in news_list:
        if isinstance(n, dict):
            title = n.get('title', '')
            desc = n.get('description', '')
            if desc:
                news_content.append(f"• {title}: {desc}")
            else:
                news_content.append(f"• {title}")
        else:
            news_content.append(f"• {str(n)}")
    
    if not news_content:
        return {"correlation_score": 50, "reason": "분석 가능한 뉴스 내용이 없습니다."}
    
    news_text = "\n".join(news_content[:5])  # 최대 5개만

    prompt = f"""종목: {ticker}

뉴스:
{news_text}

위 뉴스를 분석하여 주가 영향을 점수로 평가하세요.
호재: 70~100점, 중립: 40~69점, 악재: 0~39점

JSON만 출력:
{{"correlation_score": 숫자, "reason": "이유"}}"""
    
    try:
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config={
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 500,
            }
        )
        
        # 타임아웃 설정 포함
        response = model.generate_content(
            prompt,
            request_options={"timeout": 15}
        )
        
        if not response or not response.text:
            raise ValueError("빈 응답")
        
        text = response.text.strip()
        
        # 여러 방식으로 JSON 추출 시도
        json_text = text
        
        # 방법 1: ```json ``` 제거
        if '```json' in json_text:
            parts = json_text.split('```json')
            if len(parts) > 1:
                json_text = parts[1].split('```')[0]
        # 방법 2: ``` ``` 제거
        elif '```' in json_text:
            parts = json_text.split('```')
            if len(parts) >= 2:
                json_text = parts[1]
        
        # 방법 3: { } 추출
        if '{' in json_text and '}' in json_text:
            start = json_text.find('{')
            end = json_text.rfind('}') + 1
            json_text = json_text[start:end]
        
        json_text = json_text.strip()
        
        # JSON 파싱
        result = json.loads(json_text)
        
        # 필수 필드 확인
        score = result.get('correlation_score')
        reason = result.get('reason')
        
        if score is None or reason is None:
            raise ValueError("필수 필드 누락")
        
        # 점수 정수 변환 및 범위 보정
        score = int(float(score))  # float도 처리
        score = max(0, min(100, score))
        
        # 이유 문자열 정리
        reason = str(reason).strip()[:500]
        if not reason:
            reason = "뉴스 내용을 종합하여 평가했습니다."
        
        return {
            "correlation_score": score,
            "reason": reason
        }
        
    except json.JSONDecodeError as e:
        # JSON 파싱 실패
        return {
            "correlation_score": 50, 
            "reason": "뉴스 내용을 분석했으나 결과 형식 오류로 중립으로 평가합니다."
        }
    except ValueError as e:
        # 빈 응답 또는 필수 필드 누락
        return {
            "correlation_score": 50, 
            "reason": "뉴스 내용을 확인했으나 명확한 영향을 판단하기 어려워 중립으로 평가합니다."
        }
    except Exception as e:
        # 기타 모든 오류
        return {
            "correlation_score": 50, 
            "reason": "최신 뉴스를 확인했으나 현재 시점에서는 중립적으로 평가합니다."
        }

# --- [3단계] 프랙탈 통계 (정교화) ---
def get_fractal_statistics(yf_ticker, similarity_threshold=0.90, tail_threshold=0.95, days_offset=0):
    """
    정교화된 프랙탈 분석
    - similarity_threshold: 전체 20일 유사도 기준 (기본 90%)
    - tail_threshold: 후반 4일(20%) 유사도 기준 (기본 95%)
    """
    try:
        end_date = datetime.now() - timedelta(days=days_offset)
        df = yf.download(yf_ticker, start=(end_date - timedelta(days=365*5)).strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'), progress=False)
        if len(df) < 40:
            return None
        
        closes = df['Close'].values.flatten()
        dates = df.index
        current_window = closes[-20:]
        current_z = zscore(current_window)
        
        # ✅ 후반 20% (4일) 별도 추출
        current_tail = current_z[-4:]
        
        sims = []
        future_window = 7  # 미래 7일 데이터
        
        for i in range(len(closes) - 20 - future_window):
            past_window = closes[i:i+20]
            past_z = zscore(past_window)
            
            # ✅ 1. 전체 유사도 (20일)
            sim_full = 1 - cosine(current_z, past_z)
            
            # ✅ 2. 후반 유사도 (마지막 4일)
            past_tail = past_z[-4:]
            sim_tail = 1 - cosine(current_tail, past_tail)
            
            base_price = closes[i+19]
            future_prices = closes[i+20 : i+20+future_window]
            max_price = np.max(future_prices)
            max_idx = np.argmax(future_prices) + 1
            rise_pct = (max_price - base_price) / base_price * 100
            
            past_with_future = closes[i:i+20+future_window]
            
            sims.append({
                "similarity": sim_full,  # 전체 유사도
                "similarity_tail": sim_tail,  # 후반 유사도
                "date": dates[i+19].strftime('%Y-%m-%d'),
                "rise_pct": rise_pct,
                "days_to_max": max_idx,
                "past_prices": past_window,
                "past_with_future": past_with_future
            })
            
        sims.sort(key=lambda x: (x["similarity"], x["similarity_tail"]), reverse=True)
        
        # ✅ 두 조건 모두 만족하는 패턴만 선택
        valid = [c for c in sims[:20] if c["similarity"] >= similarity_threshold 
                 and c["similarity_tail"] >= tail_threshold][:5]
        
        if len(valid) == 0:
            return None
        
        avg_rise = sum(c["rise_pct"] for c in valid) / len(valid)
        avg_days = sum(c["days_to_max"] for c in valid) / len(valid)
        
        return {
            "valid_cases": valid,
            "avg_rise": round(avg_rise, 2),
            "avg_days": round(avg_days, 1),
            "current_prices": current_window
        }
    except Exception as e:
        return None

# ✅ [백테스팅] 일주일 전 분석 검증
def backtest_prediction(ticker_code, yf_ticker, company_name, similarity_threshold, tail_threshold):
    """
    일주일 전에 분석했다면 어땠을지 검증
    """
    try:
        # 일주일 전 데이터
        past_stock = get_stock_data(ticker_code, yf_ticker, days_ago=7)
        past_price = past_stock.get('price', 0)
        past_date = past_stock.get('date', 'N/A')
        
        # 현재 데이터
        current_stock = get_stock_data(ticker_code, yf_ticker, days_ago=0)
        current_price = current_stock.get('price', 0)
        
        if past_price == 0 or current_price == 0:
            return None
        
        # 일주일 전 프랙탈 분석
        past_frac = get_fractal_statistics(yf_ticker, similarity_threshold, tail_threshold, days_offset=7)
        
        if not past_frac:
            return None
        
        # 일주일 전 예측
        predicted_rise = past_frac['avg_rise']
        predicted_price = int(past_price * (1 + predicted_rise/100))
        
        # 실제 결과
        actual_rise = ((current_price - past_price) / past_price) * 100
        
        # 예측 정확도 계산
        direction_correct = (predicted_rise > 0 and actual_rise > 0) or (predicted_rise < 0 and actual_rise < 0)
        
        error_rate = abs(predicted_rise - actual_rise)
        accuracy = max(0, 100 - error_rate * 2)
        
        return {
            "past_date": past_date,
            "past_price": past_price,
            "predicted_price": predicted_price,
            "predicted_rise": predicted_rise,
            "current_price": current_price,
            "actual_rise": actual_rise,
            "direction_correct": direction_correct,
            "accuracy": round(accuracy, 1),
            "past_frac": past_frac
        }
        
    except Exception as e:
        return None

# --- [한국 주식 목록 로드] ---
@st.cache_data(ttl=3600)
def load_krx():
    default_stocks = pd.DataFrame({
        'Code': [
            '005930', '000660', '051910', '035420', '035720', '005380', '068270', 
            '207940', '006400', '012330', '000270', '373220', '005490', '105560',
            '028260', '055550', '017670', '096770', '034730', '003670', '018260',
            '009150', '032830', '015760', '003550', '066570', '011200', '010130',
            '047050', '090430', '251270', '086520', '326030', '042700', '361610',
            '036570', '024110', '000100', '033780', '009540', '161390', '021240',
            '030200', '010950', '267250', '011070', '034020', '302440', '000810',
            '138040', '006800'
        ],
        'Name': [
            '삼성전자', 'SK하이닉스', 'LG화학', 'NAVER', '카카오', '현대차', '셀트리온',
            '삼성바이오로직스', '삼성SDI', '현대모비스', '기아', 'LG에너지솔루션', 'POSCO홀딩스', 'KB금융',
            '삼성물산', '신한지주', 'SK텔레콤', '에코프로', 'SK', 'SK이노베이션', '에코프로비엠',
            '삼성전기', '삼성생명', '한국전력', 'LG', 'LG전자', 'HMM', '고려아연',
            '포스코퓨처엠', '아모레퍼시픽', '넷마블', '에코프로에이치엔', 'SK바이오팜', '한미약품', 'SK하이닉스우',
            '엔씨소프트', '기업은행', '유한양행', 'KT&G', 'HD현대중공업', '한국항공우주', '코웨이',
            'KT', '에스원', '카카오뱅크', 'LG이노텍', 'SK수펙스추', '삼성에스디에스', '삼성화재',
            '메리츠금융지주', '미래에셋증권'
        ]
    })
    
    if not FDR_AVAILABLE:
        return default_stocks
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = fdr.StockListing('KRX')
            
        if df.empty or 'Code' not in df.columns or 'Name' not in df.columns:
            return default_stocks
        return df
    except:
        return default_stocks

# --- [웹 UI] ---
st.set_page_config(page_title="AI 주식 스캐너", layout="wide", page_icon="📈")

krx_list = load_krx()

st.title("📈 나만의 AI 팩트 스캐너")
st.markdown("**AI 기반 실시간 주식 분석 시스템** - 뉴스 감성분석 + 정교한 프랙탈 패턴 인식")
st.markdown("---")

# 사이드바
with st.sidebar:
    st.header("📚 사용 가이드")
    st.markdown("""
    1. **종목 검색**: 종목명 또는 6자리 코드 입력
    2. **AI 분석**: 최신 뉴스의 호재/악재 자동 판단
    3. **정교한 프랙탈**: 전체 90% + 후반 95% 일치
    4. **예측**: 과거 패턴 이후 평균 수익률 확인
    5. **백테스팅**: 일주일 전 예측 검증
    """)
    st.markdown("---")
    
    st.subheader("🎯 분석 설정")
    
    similarity_threshold = st.slider(
        "전체 유사도 기준 (%)", 
        min_value=85, 
        max_value=95, 
        value=90,
        step=1,
        help="전체 20일 차트 패턴 유사도"
    )
    
    tail_threshold = st.slider(
        "후반부 유사도 기준 (%)", 
        min_value=90, 
        max_value=99, 
        value=95,
        step=1,
        help="최근 4일(후반 20%) 차트 패턴 유사도"
    )
    
    st.caption(f"**전체 {similarity_threshold}%** + **후반 {tail_threshold}%** 이상 일치하는 패턴만 사용")
    
    st.markdown("---")
    st.info("💡 **Tip**: 코스피는 0으로 시작, 코스닥은 그 외 숫자")
    
    with st.expander("📋 주요 지원 종목"):
        for idx, row in krx_list.head(20).iterrows():
            st.text(f"{row['Name']} ({row['Code']})")
        if len(krx_list) > 20:
            st.text(f"... 외 {len(krx_list)-20}개 종목")

# 종목 검색
user_input = st.text_input(
    "🔍 분석할 종목명 또는 종목코드(6자리)를 입력하세요", 
    placeholder="예: 삼성전자 또는 005930",
    help="종목명 또는 6자리 코드를 입력하세요. 전체 상장 종목 검색 가능합니다."
)

if user_input and user_input.strip():
    if user_input != st.session_state.last_input:
        st.session_state.current_ticker = None
        st.session_state.current_company = None
        st.session_state.last_input = user_input
    
    target_name, target_ticker = "", ""
    
    if user_input.isdigit():
        target_ticker = user_input.zfill(6)
        matched = krx_list[krx_list['Code'] == target_ticker]
        if not matched.empty:
            target_name = matched.iloc[0]['Name']
        else:
            target_name = f"종목코드 {target_ticker}"
    else:
        target_name = user_input
        matched = krx_list[krx_list['Name'].str.contains(target_name, na=False, case=False)]
        if not matched.empty:
            target_ticker = matched.iloc[0]['Code']
            target_name = matched.iloc[0]['Name']
        else:
            st.error("❌ 종목을 찾을 수 없습니다. 정확한 종목명 또는 6자리 코드를 입력해주세요.")
            st.info("💡 사이드바의 '주요 지원 종목'에서 검색 가능한 종목을 확인하세요.")
            st.stop()
    
    if not target_ticker:
        st.error("❌ 유효한 종목 코드를 입력해주세요.")
        st.stop()
    
    st.session_state.current_ticker = target_ticker
    st.session_state.current_company = target_name
    
    yf_ticker = f"{target_ticker}.KS" if target_ticker.startswith("0") else f"{target_ticker}.KQ"
    
    with st.spinner(f"'{target_name}' 데이터를 수집하고 AI로 분석 중입니다... ⏳"):
        stock_data = get_stock_data(target_ticker, yf_ticker)
        news_data = get_news_data(target_name)
        ai_result = analyze_validity(target_ticker, news_data)
        frac_result = get_fractal_statistics(yf_ticker, similarity_threshold/100, tail_threshold/100)
        
        # ✅ 백테스팅 (일주일 전)
        backtest_result = backtest_prediction(target_ticker, yf_ticker, target_name, 
                                             similarity_threshold/100, tail_threshold/100)

    # ✅ [백테스팅 섹션]
    if backtest_result:
        st.markdown("## 🕐 만약 일주일 전에 검색했다면?")
        st.markdown("**AI 예측의 신뢰도를 실제 데이터로 검증합니다**")
        
        bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
        
        with bt_col1:
            st.metric(
                label="📅 분석 시점",
                value="일주일 전"
            )
            st.caption(f"{backtest_result['past_date']}")
        
        with bt_col2:
            st.metric(
                label="💵 당시 종가",
                value=f"{backtest_result['past_price']:,}원"
            )
        
        with bt_col3:
            predicted_diff = backtest_result['predicted_price'] - backtest_result['past_price']
            st.metric(
                label="🎯 AI 예측 목표가",
                value=f"{backtest_result['predicted_price']:,}원",
                delta=f"{predicted_diff:+,}원 ({backtest_result['predicted_rise']:+.2f}%)"
            )
        
        with bt_col4:
            actual_diff = backtest_result['current_price'] - backtest_result['past_price']
            st.metric(
                label="📈 실제 결과 (일주일 후)",
                value=f"{backtest_result['current_price']:,}원",
                delta=f"{actual_diff:+,}원 ({backtest_result['actual_rise']:+.2f}%)",
                delta_color="normal" if backtest_result['actual_rise'] > 0 else "inverse"
            )
        
        st.markdown("---")
        accuracy = backtest_result['accuracy']
        direction = backtest_result['direction_correct']
        
        acc_col1, acc_col2 = st.columns(2)
        
        with acc_col1:
            if direction:
                st.success(f"✅ **예측 방향 적중**: {'상승' if backtest_result['predicted_rise'] > 0 else '하락'} 예측 → 실제 {'상승' if backtest_result['actual_rise'] > 0 else '하락'}")
            else:
                st.error(f"❌ **예측 방향 불일치**: {'상승' if backtest_result['predicted_rise'] > 0 else '하락'} 예측 → 실제 {'상승' if backtest_result['actual_rise'] > 0 else '하락'}")
        
        with acc_col2:
            if accuracy >= 80:
                stars = "⭐⭐⭐⭐⭐"
                trust = "매우 높음"
                color = "#00C851"
            elif accuracy >= 60:
                stars = "⭐⭐⭐⭐"
                trust = "높음"
                color = "#33B679"
            elif accuracy >= 40:
                stars = "⭐⭐⭐"
                trust = "보통"
                color = "#FFB300"
            else:
                stars = "⭐⭐"
                trust = "낮음"
                color = "#FF4444"
            
            st.markdown(
                f"<div style='background-color: {color}22; padding: 15px; border-radius: 10px; border-left: 4px solid {color};'>"
                f"<h3 style='margin: 0; color: {color};'>예측 정확도: {accuracy:.1f}%</h3>"
                f"<p style='margin: 5px 0 0 0;'>{stars} 신뢰도: <strong>{trust}</strong></p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        st.caption("💡 정확도는 예측 방향 + 수치 오차를 종합 평가합니다. 80% 이상이면 신뢰할 만한 예측입니다.")
        
        st.markdown("---")

    # 종목명 및 현재가
    st.header(f"🏢 {st.session_state.current_company} ({st.session_state.current_ticker})")
    current_price = stock_data.get('price', 0)
    price_date = stock_data.get('date', 'N/A')
    price_source = stock_data.get('source', 'N/A')
    
    if current_price > 0:
        if frac_result:
            avg_rise = frac_result['avg_rise']
            avg_days = frac_result['avg_days']
            expected_price = int(current_price * (1 + avg_rise/100))
            price_diff = expected_price - current_price
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(
                    label="💵 현재가",
                    value=f"{current_price:,}원"
                )
                st.caption(f"📅 {price_date} 기준 ({price_source})")
            with col_m2:
                st.metric(
                    label="🎯 예상 목표가",
                    value=f"{expected_price:,}원",
                    delta=f"{price_diff:+,}원 ({avg_rise:+.2f}%)",
                    delta_color="normal" if avg_rise > 0 else "inverse"
                )
            with col_m3:
                st.metric(
                    label="⏱️ 예상 기간",
                    value=f"{int(avg_days)}일 후"
                )
            
            st.caption("⚠️ 이 목표가는 과거 유사 패턴 분석에 기반한 참고 수치입니다. 투자 판단 시 다양한 요인을 고려하세요.")
        else:
            st.subheader(f"💵 현재가: **{current_price:,}원**")
            st.caption(f"📅 {price_date} 기준 ({price_source})")
    else:
        st.warning("⚠️ 현재가를 불러올 수 없습니다. 종목코드를 확인해주세요.")
    
    col1, col2 = st.columns([1, 1])
    
    # 왼쪽: 뉴스 및 AI 스코어
    with col1:
        st.markdown("### 📰 최신 관련 뉴스")
        if news_data:
            for idx, news_item in enumerate(news_data, 1):
                if isinstance(news_item, dict):
                    title = news_item['title']
                    link = news_item['link']
                    st.markdown(f"**{idx}.** [{title}]({link})")
                else:
                    st.markdown(f"**{idx}.** {news_item}")
        else:
            st.info("최근 뉴스가 없습니다.")
        
        st.markdown("---")
        score = ai_result.get('correlation_score', 50)
        reason = ai_result.get('reason', '분석 불가')
        
        st.markdown("### 🧠 AI 호재/악재 점수")
        
        filled = int(score // 10)
        empty = 10 - filled
        bar_visual = "■" * filled + "□" * empty
        
        if score >= 70:
            color = "#FF4B4B"
            emoji = "🚀"
            status = "호재"
        elif score <= 30:
            color = "#4B8BFF"
            emoji = "⚠️"
            status = "악재"
        else:
            color = "#888888"
            emoji = "➖"
            status = "중립"
        
        st.markdown(
            f"<h2 style='color: {color};'>{emoji} {bar_visual} ({score}점) {emoji}</h2>", 
            unsafe_allow_html=True
        )
        st.info(f"**AI 판단:** {status}\n\n**분석 근거:** {reason}")

    # 오른쪽: 프랙탈 분석
    with col2:
        st.markdown("### 📈 정교한 프랙탈 패턴 분석")
        if frac_result:
            avg_rise = frac_result['avg_rise']
            avg_days = frac_result['avg_days']
            
            st.info(f"🎯 **검색 조건**: 전체 {similarity_threshold}% + 후반 {tail_threshold}% 일치")
            
            if avg_rise > 0:
                st.success(
                    f"💡 과거 유사한 흐름 이후 **평균 {avg_days}일 동안 {avg_rise:+.2f}% 상승**"
                )
            else:
                st.error(
                    f"⚠️ 과거 유사한 흐름 이후 **평균 {avg_days}일 동안 {avg_rise:+.2f}% 하락**"
                )
            
            fig = go.Figure()
            
            for i, case in enumerate(frac_result['valid_cases']):
                full_pattern = case['past_with_future']
                past_norm = (full_pattern - full_pattern[0]) / full_pattern[0] * 100
                
                x_axis = list(range(len(past_norm)))
                
                fig.add_trace(go.Scatter(
                    x=x_axis[:20],
                    y=past_norm[:20], 
                    mode='lines', 
                    line=dict(color='rgba(150, 150, 150, 0.5)', width=2), 
                    name=f"과거 {i+1} ({case['date']})",
                    showlegend=(i==0)
                ))
                
                fig.add_trace(go.Scatter(
                    x=x_axis[19:],
                    y=past_norm[19:], 
                    mode='lines', 
                    line=dict(color='rgba(150, 150, 150, 0.5)', width=2, dash='dot'), 
                    name=f"미래 {i+1}",
                    showlegend=False
                ))
            
            curr_norm = (frac_result['current_prices'] - frac_result['current_prices'][0]) / frac_result['current_prices'][0] * 100
            fig.add_trace(go.Scatter(
                x=list(range(20)),
                y=curr_norm, 
                mode='lines+markers', 
                line=dict(color='red', width=4), 
                marker=dict(size=6),
                name='현재 20일 흐름'
            ))
            
            fig.add_vline(
                x=19.5, 
                line_dash="dash", 
                line_color="gray",
                annotation_text="현재 시점",
                annotation_position="top"
            )
            
            fig.update_layout(
                title="현재 vs 과거 유사 패턴 + 이후 7일 흐름", 
                xaxis_title="거래일 (20일 이후는 예측)", 
                yaxis_title="수익률 (%)", 
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{st.session_state.current_ticker}")
            
            with st.expander("📊 상세 통계"):
                for i, case in enumerate(frac_result['valid_cases'], 1):
                    st.write(
                        f"**패턴 {i}** ({case['date']}) - "
                        f"전체 유사도: {case['similarity']:.2%} | "
                        f"후반 유사도: {case['similarity_tail']:.2%} | "
                        f"상승률: {case['rise_pct']:+.2f}% | "
                        f"최고점: {case['days_to_max']}일"
                    )
        else:
            st.warning("⚠️ 조건을 만족하는 유사 패턴을 찾을 수 없습니다.")
            st.info(f"💡 전체 {similarity_threshold}% + 후반 {tail_threshold}% 조건을 낮춰보세요.")

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Made with ❤️ by AI Stock Scanner | Data: FinanceDataReader, Naver News, Google Gemini"
    "</div>", 
    unsafe_allow_html=True
)
