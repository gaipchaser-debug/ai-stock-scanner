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
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time

try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False
    st.warning("⚠️ FinanceDataReader 미설치 - Yahoo Finance로 대체됩니다.")

# --- [환경 설정 및 API 키] ---
try:
    NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
    NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
    DART_API_KEY = st.secrets["DART_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "")
    NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")
    DART_API_KEY = os.getenv("DART_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error("🔴 GEMINI_API_KEY가 설정되지 않았습니다.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- [세션 상태 초기화] ---
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'current_company' not in st.session_state:
    st.session_state.current_company = None

# --- [1단계] 데이터 수집 ---
def get_stock_data(ticker_code, yf_ticker, days_ago=0):
    """
    days_ago=0: 오늘 현재가
    days_ago=10: 10일 전 종가
    """
    if FDR_AVAILABLE:
        try:
            end_date = datetime.now() - timedelta(days=days_ago)
            start_date = end_date - timedelta(days=30)
            
            df = fdr.DataReader(
                ticker_code, 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            if df.empty:
                return {"price": 0, "date": "", "source": "오류", "dataframe": None}
            
            current_price = int(df['Close'].iloc[-1])
            last_date = df.index[-1].strftime('%Y-%m-%d %H:%M')
            
            return {
                "price": current_price,
                "date": last_date,
                "source": "네이버 금융",
                "dataframe": df
            }
        except Exception as e:
            pass
    
    # Fallback to Yahoo Finance
    try:
        df = yf.download(yf_ticker, period="1d", progress=False)
        if df.empty:
            return {"price": 0, "date": "", "source": "오류", "dataframe": None}
        
        current_price = int(df['Close'].iloc[-1])
        last_date = df.index[-1].strftime('%Y-%m-%d')
        
        return {
            "price": current_price,
            "date": last_date,
            "source": "Yahoo Finance",
            "dataframe": df
        }
    except Exception as e:
        return {"price": 0, "date": "", "source": "오류", "dataframe": None}


def crawl_news_content(url):
    """뉴스 기사 본문 크롤링"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return ""
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 네이버 뉴스 본문 추출
        article = soup.find('article') or soup.find('div', {'id': 'articleBodyContents'}) or soup.find('div', {'class': 'article_body'})
        if article:
            # 스크립트/광고 제거
            for script in article(['script', 'style', 'aside', 'footer']):
                script.decompose()
            text = article.get_text(strip=True, separator=' ')
            return text[:1000]  # 최대 1000자
        
        return ""
    except Exception as e:
        return ""


def get_news_data(company_name):
    """뉴스 제목 + 본문 크롤링"""
    try:
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID, 
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        params = {
            "query": f"{company_name} 주가", 
            "display": 10,  # 더 많이 수집 후 필터링
            "sort": "date"
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code != 200:
            return []
        
        items = response.json().get('items', [])
        news_data = []
        
        for item in items:
            title = item['title'].replace("<b>","").replace("</b>","")
            link = item['link']
            description = item.get('description', '').replace("<b>","").replace("</b>","")
            
            # 회사명 필터링
            if company_name not in title and company_name not in description:
                continue
            
            # 본문 크롤링 (최대 5개)
            content = ""
            if len(news_data) < 5:
                content = crawl_news_content(link)
                time.sleep(0.3)  # 크롤링 간격
            
            news_data.append({
                "title": title,
                "link": link,
                "description": description,
                "content": content
            })
            
            if len(news_data) >= 5:
                break
        
        return news_data
        
    except Exception as e:
        return []


# --- [2-1단계] 개별 뉴스 호재/악재 분석 (강화된 프롬프트) ---
def analyze_single_news(ticker, news_item):
    """개별 뉴스 하나에 대한 호재/악재 명확한 판단"""
    text = f"[제목] {news_item['title']}\n[요약] {news_item['description']}"
    if news_item.get('content'):
        text += f"\n[본문] {news_item['content'][:800]}"
    
    prompt = f"""당신은 20년 경력의 전문 증권 애널리스트입니다. 
다음 뉴스가 주가에 미치는 영향을 **명확하게** 판단하세요.

**종목코드**: {ticker}

**뉴스 내용**:
{text}

**판단 기준 (엄격하게 적용)**:

🟢 **호재 (70~100점)** - 주가 상승 가능성이 높은 뉴스
  ✓ 매출/영업이익 증가, 실적 개선
  ✓ 신규 계약/수주 (특히 대규모)
  ✓ 기술 혁신, 신제품 출시
  ✓ 투자 유치, 자금 조달 성공
  ✓ 시장 점유율 상승
  ✓ 긍정적 전망, 목표가 상향
  ✓ 배당 확대, 자사주 매입

🔴 **악재 (0~30점)** - 주가 하락 가능성이 높은 뉴스
  ✗ 매출/영업이익 감소, 적자
  ✗ 계약 취소, 수주 실패
  ✗ 소송, 규제, 제재
  ✗ 경영진 사퇴/구속
  ✗ 시장 점유율 하락
  ✗ 부정적 전망, 목표가 하향
  ✗ 구조조정, 감원

🟡 **중립 (40~60점)** - 주가 영향이 불분명하거나 단순 사실
  • 정기 행사/공지 (주총, IR)
  • 인사 발령 (중립적)
  • 일반적인 업계 동향
  • 확정되지 않은 추측성 보도

**중요**: 
- 뉴스가 "실적 증가", "수주", "매출 상승" 등 긍정적 키워드를 포함하면 **반드시 호재**로 판단
- 뉴스가 "실적 감소", "적자", "소송", "규제" 등 부정적 키워드를 포함하면 **반드시 악재**로 판단
- **중립은 최후의 선택**: 명확한 긍정/부정 요소가 없을 때만 사용

**출력 형식** (JSON만):
{{"sentiment": "호재" or "악재" or "중립", "score": 0~100, "reason": "구체적 이유 50자"}}

예시:
- "4분기 영업이익 30% 증가" → {{"sentiment": "호재", "score": 88, "reason": "실적 대폭 개선으로 긍정적"}}
- "소송 패소, 100억 배상" → {{"sentiment": "악재", "score": 25, "reason": "대규모 배상금으로 부정적"}}
- "정기 주주총회 개최" → {{"sentiment": "중립", "score": 50, "reason": "단순 정기 행사"}}
"""
    
    try:
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config={
                "temperature": 0.1,  # 약간의 창의성 허용
                "top_p": 0.9,
                "max_output_tokens": 300
            }
        )
        
        response = model.generate_content(prompt, request_options={"timeout": 15})
        text = response.text.strip()
        
        # JSON 추출
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            text = text[start:end]
        
        result = json.loads(text.strip())
        
        sentiment = result.get("sentiment", "중립")
        score = int(float(result.get("score", 50)))
        
        # 점수 기반 감성 재검증
        if score >= 70:
            sentiment = "호재"
        elif score <= 30:
            sentiment = "악재"
        else:
            sentiment = "중립"
        
        return {
            "sentiment": sentiment,
            "score": score,
            "reason": result.get("reason", "")[:100]
        }
        
    except Exception as e:
        return {"sentiment": "중립", "score": 50, "reason": f"분석 오류: {str(e)[:30]}"}


# --- [2-2단계] 전체 뉴스 종합 분석 ---
def analyze_validity(ticker, news_data_with_sentiment):
    """개별 뉴스 감성 종합"""
    if not news_data_with_sentiment:
        return {
            "correlation_score": 50, 
            "reason": "최신 뉴스가 없어 중립적으로 평가합니다."
        }
    
    # 점수 평균
    scores = [n["sentiment_data"]["score"] for n in news_data_with_sentiment]
    avg_score = int(np.mean(scores))
    
    # 호재/악재 카운트
    sentiments = [n["sentiment_data"]["sentiment"] for n in news_data_with_sentiment]
    positive_count = sentiments.count("호재")
    negative_count = sentiments.count("악재")
    neutral_count = sentiments.count("중립")
    
    # 종합 판단
    if positive_count > negative_count:
        if positive_count >= 3:
            reason = f"강한 호재! 호재 뉴스 {positive_count}개, 악재 {negative_count}개로 긍정적 전망"
        else:
            reason = f"호재 뉴스 {positive_count}개, 악재 {negative_count}개로 긍정적 전망"
    elif negative_count > positive_count:
        if negative_count >= 3:
            reason = f"강한 악재! 악재 뉴스 {negative_count}개, 호재 {positive_count}개로 부정적 전망"
        else:
            reason = f"악재 뉴스 {negative_count}개, 호재 {positive_count}개로 부정적 전망"
    else:
        reason = f"혼재 상황: 호재 {positive_count}개, 악재 {negative_count}개, 중립 {neutral_count}개"
    
    return {
        "correlation_score": avg_score,
        "reason": reason
    }


# --- [3단계] 프랙탈 통계 (14일 = 2주 기준, 완화된 조건) ---
@st.cache_data(ttl=3600)
def get_fractal_statistics(yf_ticker, similarity_threshold=0.85, tail_threshold=0.90, days_offset=0):
    """
    과거 14일(2주) 차트 패턴 분석
    - 전체 14일 유사도 ≥ similarity_threshold (기본 85%)
    - 후반 3일(약 21%) 유사도 ≥ tail_threshold (기본 90%)
    - 미래 14일 예측
    """
    try:
        # 과거 3년 데이터 (5년 → 3년으로 완화)
        end_date = datetime.now() - timedelta(days=days_offset)
        start_date = end_date - timedelta(days=365*3)
        
        df = yf.download(
            yf_ticker, 
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if len(df) < 50:  # 최소 50일로 완화 (60 → 50)
            return None
        
        closes = df['Close'].values
        
        # 현재 14일 패턴
        if len(closes) < 14:
            return None
        
        current_window = closes[-14:]
        
        # zscore 안전 계산
        if np.std(current_window) < 0.01:  # 거의 변화 없는 경우
            return None
        
        current_z = zscore(current_window)
        
        # 과거 패턴 검색
        similarities = []
        
        for i in range(len(closes) - 28):  # 14일 + 14일 미래
            past_window = closes[i:i+14]
            
            if np.std(past_window) < 0.01:
                continue
            
            try:
                past_z = zscore(past_window)
            except:
                continue
            
            # 전체 유사도
            try:
                sim_full = 1 - cosine(current_z, past_z)
                if np.isnan(sim_full):
                    continue
            except:
                continue
            
            # 후반 3일 유사도
            try:
                sim_tail = 1 - cosine(current_z[-3:], past_z[-3:])
                if np.isnan(sim_tail):
                    sim_tail = 0
            except:
                sim_tail = 0
            
            # 필터링 (기본 85%/90%로 완화)
            if sim_full < similarity_threshold or sim_tail < tail_threshold:
                continue
            
            # 미래 14일 데이터
            future_14d = closes[i+14:i+28]
            if len(future_14d) < 14:
                continue
            
            base_price = closes[i+13]  # 14일째 종가
            max_price = future_14d.max()
            max_idx = future_14d.argmax()
            rise_pct = ((max_price - base_price) / base_price) * 100
            
            past_with_future = closes[i:i+28]
            
            similarities.append({
                "similarity": sim_full,
                "tail_similarity": sim_tail,
                "date": df.index[i].strftime('%Y-%m-%d'),
                "rise_pct": round(rise_pct, 2),
                "days_to_max": max_idx + 1,
                "past_prices": past_window.tolist(),
                "past_with_future": past_with_future.tolist()
            })
        
        if not similarities:
            return None
        
        # 유사도 순 정렬 후 상위 5개
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        valid_cases = similarities[:5]
        
        avg_rise = round(np.mean([c["rise_pct"] for c in valid_cases]), 2)
        avg_days = round(np.mean([c["days_to_max"] for c in valid_cases]), 1)
        
        return {
            "avg_rise": avg_rise,
            "avg_days": avg_days,
            "valid_cases": valid_cases,
            "current_prices": current_window.tolist()
        }
        
    except Exception as e:
        st.error(f"⚠️ 프랙탈 분석 오류: {str(e)[:100]}")
        return None


# --- [백테스팅] (10일로 완화) ---
def backtest_prediction(ticker_code, yf_ticker, days_ago=10):
    """10일 전 검색했다면?"""
    try:
        # 10일 전 데이터
        past_data = get_stock_data(ticker_code, yf_ticker, days_ago=days_ago)
        past_price = past_data["price"]
        
        if past_price == 0:
            return None
        
        # 10일 전 프랙탈 분석 (완화된 조건)
        past_fractal = get_fractal_statistics(
            yf_ticker, 
            similarity_threshold=0.80,  # 80%로 완화
            tail_threshold=0.85,  # 85%로 완화
            days_offset=days_ago
        )
        
        if not past_fractal:
            return None
        
        predicted_rise = past_fractal["avg_rise"]
        predicted_price = int(past_price * (1 + predicted_rise / 100))
        
        # 오늘 실제 가격
        current_data = get_stock_data(ticker_code, yf_ticker, days_ago=0)
        current_price = current_data["price"]
        
        actual_rise = ((current_price - past_price) / past_price) * 100
        
        # 정확도 계산
        if predicted_rise > 0 and actual_rise > 0:
            accuracy = min(predicted_rise, actual_rise) / max(predicted_rise, actual_rise) * 100
        elif predicted_rise < 0 and actual_rise < 0:
            accuracy = min(abs(predicted_rise), abs(actual_rise)) / max(abs(predicted_rise), abs(actual_rise)) * 100
        else:
            accuracy = 0
        
        return {
            "past_price": past_price,
            "predicted_price": predicted_price,
            "predicted_rise": predicted_rise,
            "actual_price": current_price,
            "actual_rise": actual_rise,
            "accuracy": round(accuracy, 1),
            "direction_match": (predicted_rise > 0 and actual_rise > 0) or (predicted_rise < 0 and actual_rise < 0)
        }
        
    except Exception as e:
        return None


# --- [한국 주식 목록] ---
@st.cache_data(ttl=3600)
def load_krx():
    """한국거래소 상장 종목"""
    if FDR_AVAILABLE:
        try:
            df = fdr.StockListing('KRX')
            return df[['Code', 'Name']]
        except:
            pass
    
    # 기본 목록
    return pd.DataFrame({
        'Code': ['005930', '000660', '035420', '051910', '035720', '207940', '005490', '068270', 
                 '006400', '105560', '055550', '096770', '028260', '012330', '017670', '032830',
                 '066570', '003550', '015760', '034730', '018260', '011200', '030200', '051900',
                 '010950', '036570', '024110', '086790', '009150', '029780', '064350', '003490',
                 '161390', '000270', '010130', '047050', '042660', '003670', '011070', '034220',
                 '139480', '090430', '004020', '251270', '000100', '259960', '095720', '000810',
                 '033780', '006260', '039490'],
        'Name': ['삼성전자', 'SK하이닉스', 'NAVER', 'LG화학', '카카오', '삼성바이오로직스', 'POSCO홀딩스', '셀트리온',
                 '삼성SDI', 'KB금융', '신한지주', 'SK이노베이션', '삼성물산', '현대모비스', 'SK텔레콤', '삼성생명',
                 'LG전자', 'LG', '한국전력', 'SK', '삼성에스디에스', 'HMM', 'KT&G', 'LG생활건강',
                 'S-Oil', '엔씨소프트', '기업은행', '하나금융지주', '삼성전기', 'SKC', '현대로템', 'CJ제일제당',
                 '한국타이어앤테크놀로지', '기아', '고려아연', '포스코인터내셔널', '한진칼', '포스코퓨처엠', 'LG이노텍', 'LG디스플레이',
                 '이마트', '아모레퍼시픽', '현대건설', '넷마블', '유한양행', '크래프톤', '웹젠', '삼성화재',
                 'KT', 'LS', '키움증권']
    })


# ==================================================================
# ========================= UI 레이아웃 ============================
# ==================================================================

st.set_page_config(
    page_title="AI 주식 팩트 스캐너",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI 주식 팩트 스캐너")
st.caption("명확한 호재/악재 분석 + 2주(14일) 차트 패턴 기반 백테스팅")

# --- 사이드바 ---
with st.sidebar:
    st.header("📌 사용 방법")
    st.markdown("""
    1️⃣ **종목명 또는 코드 입력** (예: 삼성전자, 005930)
    2️⃣ **각 뉴스별 호재/악재 명확 분석**
    3️⃣ **과거 14일(2주) 유사 차트 패턴 검색**
    4️⃣ **10일 전 백테스팅 결과** 확인
    """)
    
    st.divider()
    
    # 유사도 슬라이더 (기본값 낮춤)
    st.subheader("🎚️ 프랙탈 분석 설정")
    similarity_threshold = st.slider(
        "전체 유사도",
        min_value=75,
        max_value=95,
        value=85,  # 90 → 85로 낮춤
        step=1,
        help="과거 14일 차트와 현재 14일 차트의 전체 유사도"
    )
    
    tail_threshold = st.slider(
        "후반 유사도",
        min_value=80,
        max_value=98,
        value=90,  # 95 → 90으로 낮춤
        step=1,
        help="최근 3일 차트 패턴의 유사도 (더 정밀)"
    )
    
    st.divider()
    
    st.subheader("📋 인기 종목")
    popular_stocks = ["삼성전자", "SK하이닉스", "NAVER", "카카오", "LG화학", "셀트리온"]
    for stock in popular_stocks:
        st.markdown(f"• {stock}")

# --- 메인 영역 ---
user_input = st.text_input(
    "🔍 종목명 또는 코드 입력",
    placeholder="예: 삼성전자, 005930, LG화학",
    key="stock_input"
)

if user_input:
    # 종목 검색
    krx_df = load_krx()
    
    matched = krx_df[
        (krx_df['Name'].str.contains(user_input, na=False)) | 
        (krx_df['Code'] == user_input)
    ]
    
    if matched.empty:
        st.error("❌ 종목을 찾을 수 없습니다.")
        st.stop()
    
    ticker_code = matched.iloc[0]['Code']
    company_name = matched.iloc[0]['Name']
    yf_ticker = ticker_code + ".KS" if ticker_code[0] == '0' else ticker_code + ".KQ"
    
    # 세션 상태 업데이트
    st.session_state.current_ticker = ticker_code
    st.session_state.current_company = company_name
    
    st.header(f"🏢 {company_name} ({ticker_code})")
    
    # --- 현재가 ---
    stock_data = get_stock_data(ticker_code, yf_ticker)
    current_price = stock_data["price"]
    
    if current_price > 0:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("💵 현재가", f"{current_price:,}원")
            st.caption(f"📅 {stock_data['date']} 기준 ({stock_data['source']})")
    else:
        st.warning("⚠️ 현재가 정보를 가져올 수 없습니다.")
    
    st.divider()
    
    # --- 백테스팅 ---
    st.subheader("⏮️ 10일 전 검색했다면?")
    
    with st.spinner("백테스팅 분석 중..."):
        backtest = backtest_prediction(ticker_code, yf_ticker, days_ago=10)
    
    if backtest:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("10일 전 가격", f"{backtest['past_price']:,}원")
        
        with col2:
            st.metric("AI 예측 가격", f"{backtest['predicted_price']:,}원", 
                     f"{backtest['predicted_rise']:+.2f}%")
        
        with col3:
            st.metric("실제 가격(오늘)", f"{backtest['actual_price']:,}원",
                     f"{backtest['actual_rise']:+.2f}%")
        
        with col4:
            accuracy = backtest['accuracy']
            direction = "✅" if backtest['direction_match'] else "❌"
            
            if accuracy >= 80:
                star = "⭐⭐⭐⭐⭐"
            elif accuracy >= 60:
                star = "⭐⭐⭐⭐"
            elif accuracy >= 40:
                star = "⭐⭐⭐"
            else:
                star = "⭐⭐"
            
            st.metric("예측 정확도", f"{accuracy:.1f}% {star}")
            st.caption(f"{direction} 방향 {'적중' if backtest['direction_match'] else '불일치'}")
    else:
        st.info("💡 백테스팅 데이터가 부족합니다. 최근 상장되었거나 거래가 적은 종목일 수 있습니다.")
    
    st.divider()
    
    # --- 뉴스 + AI 분석 ---
    st.subheader("📰 최신 뉴스 호재/악재 명확 분석")
    
    with st.spinner("뉴스 수집 및 각 뉴스별 AI 분석 중..."):
        news_data = get_news_data(company_name)
        
        # 각 뉴스별 감성 분석
        news_with_sentiment = []
        for news in news_data:
            sentiment_result = analyze_single_news(ticker_code, news)
            news_with_sentiment.append({
                **news,
                "sentiment_data": sentiment_result
            })
        
        # 종합 분석
        ai_result = analyze_validity(ticker_code, news_with_sentiment)
    
    score = ai_result["correlation_score"]
    reason = ai_result["reason"]
    
    # 점수 시각화
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if score >= 70:
            status = "🟢 호재"
        elif score >= 40:
            status = "🟡 중립"
        else:
            status = "🔴 악재"
        
        st.markdown(f"### {status}")
        st.markdown(f"**종합 점수**: {score}/100")
    
    with col2:
        st.markdown("**종합 분석**")
        st.info(reason)
    
    # 점수 바
    st.progress(score / 100)
    
    # 뉴스 목록 (호재/악재 표시)
    if news_with_sentiment:
        st.markdown("**📄 개별 뉴스 분석 결과**")
        
        for idx, news in enumerate(news_with_sentiment, 1):
            sentiment = news["sentiment_data"]["sentiment"]
            s_score = news["sentiment_data"]["score"]
            s_reason = news["sentiment_data"]["reason"]
            
            # 감성에 따른 이모지
            if sentiment == "호재":
                emoji = "🟢"
            elif sentiment == "악재":
                emoji = "🔴"
            else:
                emoji = "🟡"
            
            with st.expander(f"{emoji} **{sentiment}** ({s_score}점) | {news['title'][:60]}..."):
                st.markdown(f"**분석 근거**: {s_reason}")
                st.markdown(f"**링크**: [{news['link']}]({news['link']})")
                st.markdown(f"**요약**: {news['description']}")
                if news.get('content'):
                    st.markdown(f"**본문 일부**: {news['content'][:300]}...")
    else:
        st.warning("최신 뉴스를 찾을 수 없습니다.")
    
    st.divider()
    
    # --- 프랙탈 분석 ---
    st.subheader("📊 과거 2주(14일) 유사 차트 패턴 분석")
    
    with st.spinner("과거 3년 데이터 분석 중..."):
        fractal = get_fractal_statistics(
            yf_ticker, 
            similarity_threshold=similarity_threshold/100, 
            tail_threshold=tail_threshold/100
        )
    
    if fractal:
        avg_rise = fractal["avg_rise"]
        avg_days = fractal["avg_days"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📈 평균 예상 상승률", f"{avg_rise:+.2f}%")
        with col2:
            st.metric("📅 평균 최고가 도달일", f"{avg_days:.1f}일")
        
        # 목표가 계산
        target_price = int(current_price * (1 + avg_rise / 100))
        st.success(f"🎯 **AI 예상 목표가**: {target_price:,}원 (현재가 대비 {avg_rise:+.2f}%)")
        
        # 차트
        fig = go.Figure()
        
        # 현재 패턴 (빨간색)
        fig.add_trace(go.Scatter(
            x=list(range(14)),
            y=fractal["current_prices"],
            mode='lines+markers',
            name='현재 14일 패턴',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        # 과거 유사 패턴
        colors = ['gray', 'lightgray', 'darkgray', 'silver', 'dimgray']
        for idx, case in enumerate(fractal["valid_cases"]):
            past_with_future = case["past_with_future"]
            
            # 과거 14일 (실선)
            fig.add_trace(go.Scatter(
                x=list(range(14)),
                y=past_with_future[:14],
                mode='lines',
                name=f'{case["date"]} (전체 {case["similarity"]:.1%}, 후반 {case["tail_similarity"]:.1%})',
                line=dict(color=colors[idx % len(colors)], width=2, dash='solid'),
                showlegend=True
            ))
            
            # 미래 14일 (점선)
            fig.add_trace(go.Scatter(
                x=list(range(13, 28)),
                y=past_with_future[13:28],
                mode='lines',
                name=f'{case["date"]} 미래 (+{case["rise_pct"]:.1f}%)',
                line=dict(color=colors[idx % len(colors)], width=2, dash='dot'),
                showlegend=False
            ))
        
        # 현재 시점 구분선
        fig.add_vline(
            x=13.5, 
            line=dict(color='blue', width=2, dash='dash'),
            annotation_text="현재 시점",
            annotation_position="top"
        )
        
        fig.update_layout(
            title="과거 14일(2주) + 미래 14일 차트 패턴 비교",
            xaxis_title="거래일 (14일 이후는 과거 사례의 미래 흐름)",
            yaxis_title="주가 (원)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker_code}_{similarity_threshold}")
        
        # 상세 통계
        with st.expander("📋 유사 패턴 상세 통계"):
            for idx, case in enumerate(fractal["valid_cases"], 1):
                st.markdown(f"""
                **패턴 {idx}** ({case['date']})
                - 전체 유사도: {case['similarity']:.1%}
                - 후반 3일 유사도: {case['tail_similarity']:.1%}
                - 14일 후 상승률: {case['rise_pct']:+.2f}%
                - 최고가 도달: {case['days_to_max']}일째
                """)
                st.divider()
    else:
        st.warning(f"⚠️ 유사도 {similarity_threshold}%(전체), {tail_threshold}%(후반) 기준으로 유사한 과거 패턴을 찾지 못했습니다.")
        st.info("""
        💡 **해결 방법**:
        - 사이드바에서 유사도 기준을 낮춰보세요 (75~85%)
        - 최근 상장되었거나 거래량이 적은 종목일 수 있습니다
        - 백테스팅 기간은 10일 전으로 완화되어 더 많은 데이터를 찾습니다
        """)

# --- 푸터 ---
st.divider()
st.caption("© 2026 AI 주식 팩트 스캐너 | 명확한 호재/악재 분석 + 2주(14일) 차트 패턴 기반")
