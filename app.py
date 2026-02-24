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
            "display": 10,
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
                time.sleep(0.3)
            
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


# --- [2-1단계] 개별 뉴스 호재/악재 분석 ---
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
                "temperature": 0.1,
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


# --- [3단계] 크로스 종목 프랙탈 통계 (전체 KRX에서 유사 패턴 검색) ---
@st.cache_data(ttl=3600)
def get_cross_stock_fractal(current_ticker, current_prices, similarity_threshold=0.95):
    """
    혁신적 기능: 현재 종목의 14일 패턴과 유사한 패턴을 **다른 모든 종목**에서도 검색
    - 단일 종목이 아닌 전체 시장에서 유사 패턴 찾기
    - "이런 차트 모양은 보통 이렇게 진행된다" 일반 법칙 발견
    """
    try:
        # KRX 전체 종목 로드
        if not FDR_AVAILABLE:
            return None
        
        krx_df = fdr.StockListing('KRX')
        
        # 현재 종목 제외, 시가총액 상위 100개만 (속도 최적화)
        krx_df = krx_df[krx_df['Code'] != current_ticker].head(100)
        
        current_z = zscore(current_prices)
        similarities = []
        
        st.info(f"🔍 전체 KRX 종목 중 상위 100개에서 유사 패턴 검색 중... (유사도 {similarity_threshold*100:.0f}% 이상)")
        
        progress_bar = st.progress(0)
        
        for idx, row in krx_df.iterrows():
            progress_bar.progress((idx + 1) / len(krx_df))
            
            ticker = row['Code']
            name = row['Name']
            yf_ticker = ticker + ".KS" if ticker[0] == '0' else ticker + ".KQ"
            
            try:
                # 최근 1년 데이터
                df = yf.download(yf_ticker, period="1y", progress=False)
                if len(df) < 50:
                    continue
                
                closes = df['Close'].values
                
                # 14일 윈도우로 스캔
                for i in range(len(closes) - 28):
                    past_window = closes[i:i+14]
                    
                    if np.std(past_window) < 0.01:
                        continue
                    
                    try:
                        past_z = zscore(past_window)
                        sim = 1 - cosine(current_z, past_z)
                        
                        if np.isnan(sim) or sim < similarity_threshold:
                            continue
                        
                        # 미래 14일
                        future_14d = closes[i+14:i+28]
                        if len(future_14d) < 14:
                            continue
                        
                        base_price = closes[i+13]
                        max_price = future_14d.max()
                        max_idx = future_14d.argmax()
                        rise_pct = ((max_price - base_price) / base_price) * 100
                        
                        similarities.append({
                            "ticker": ticker,
                            "name": name,
                            "similarity": sim,
                            "date": df.index[i].strftime('%Y-%m-%d'),
                            "rise_pct": round(rise_pct, 2),
                            "days_to_max": max_idx + 1,
                            "past_prices": past_window.tolist(),
                            "past_with_future": closes[i:i+28].tolist()
                        })
                    except:
                        continue
                        
            except:
                continue
        
        progress_bar.empty()
        
        if not similarities:
            return None
        
        # 유사도 순 정렬 후 상위 10개
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        valid_cases = similarities[:10]
        
        avg_rise = round(np.mean([c["rise_pct"] for c in valid_cases]), 2)
        avg_days = round(np.mean([c["days_to_max"] for c in valid_cases]), 1)
        
        return {
            "avg_rise": avg_rise,
            "avg_days": avg_days,
            "valid_cases": valid_cases,
            "current_prices": current_prices
        }
        
    except Exception as e:
        st.error(f"크로스 종목 분석 오류: {str(e)[:100]}")
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

# 페이지 새로고침 시 세션 완전 초기화
if 'initialized' not in st.session_state:
    st.session_state.clear()
    st.session_state.initialized = True

st.title("📈 AI 주식 팩트 스캐너")
st.caption("크로스 종목 유사 패턴 분석 + 명확한 호재/악재 분석")

# --- 사이드바 ---
with st.sidebar:
    st.header("📌 사용 방법")
    st.markdown("""
    1️⃣ **종목명 또는 코드 입력**
    2️⃣ **각 뉴스별 호재/악재 명확 분석**
    3️⃣ **전체 KRX 종목에서 유사 패턴 검색**
    4️⃣ **"이런 차트는 보통 이렇게 진행된다" 법칙 발견**
    """)
    
    st.divider()
    
    st.subheader("🎚️ 유사도 설정")
    similarity_threshold = st.slider(
        "크로스 종목 유사도",
        min_value=90,
        max_value=99,
        value=95,
        step=1,
        help="전체 종목에서 현재 차트와 유사한 패턴을 찾는 기준 (95% 이상 권장)"
    )
    
    st.divider()
    
    st.subheader("📋 인기 종목")
    popular_stocks = ["삼성전자", "SK하이닉스", "NAVER", "카카오", "LG화학", "셀트리온"]
    for stock in popular_stocks:
        st.markdown(f"• {stock}")

# --- 메인 영역 ---
# 입력란에 고유 key 부여 + on_change 콜백
def clear_cache():
    """입력 변경 시 캐시 완전 초기화"""
    st.cache_data.clear()

user_input = st.text_input(
    "🔍 종목명 또는 코드 입력",
    placeholder="예: 삼성전자, 005930, LG화학",
    key="stock_input_unique",
    on_change=clear_cache
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
    
    # 강제 헤더 갱신 (캐시 우회)
    st.header(f"🏢 {company_name} ({ticker_code})")
    st.caption(f"입력 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # --- 현재가 ---
    stock_data = get_stock_data(ticker_code, yf_ticker)
    current_price = stock_data["price"]
    
    if current_price > 0:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("💵 현재가", f"{current_price:,}원")
            st.caption(f"📅 {stock_data['date']} ({stock_data['source']})")
    else:
        st.warning("⚠️ 현재가 정보를 가져올 수 없습니다.")
    
    st.divider()
    
    # --- 뉴스 + AI 분석 ---
    st.subheader("📰 최신 뉴스 호재/악재 명확 분석")
    
    with st.spinner("뉴스 수집 및 AI 분석 중..."):
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
    
    # 뉴스 목록
    if news_with_sentiment:
        st.markdown("**📄 개별 뉴스 분석 결과**")
        
        for idx, news in enumerate(news_with_sentiment, 1):
            sentiment = news["sentiment_data"]["sentiment"]
            s_score = news["sentiment_data"]["score"]
            s_reason = news["sentiment_data"]["reason"]
            
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
                    st.markdown(f"**본문**: {news['content'][:300]}...")
    else:
        st.warning("최신 뉴스를 찾을 수 없습니다.")
    
    st.divider()
    
    # --- 크로스 종목 프랙탈 분석 ---
    st.subheader("🌐 전체 시장 유사 패턴 분석 (크로스 종목)")
    st.info("""
    💡 **혁신적 기능**: 현재 종목의 2주 차트 패턴과 유사한 패턴을 **다른 모든 종목**에서도 검색합니다.
    
    "이런 차트 모양은 보통 이렇게 진행된다"는 일반 법칙을 발견하여 미래 예측 정확도를 높입니다.
    """)
    
    # 현재 종목의 최근 14일 데이터
    try:
        df_current = yf.download(yf_ticker, period="3mo", progress=False)
        if len(df_current) >= 14:
            current_prices = df_current['Close'].values[-14:]
            
            with st.spinner(f"🔍 KRX 전체 종목에서 유사도 {similarity_threshold}% 이상 패턴 검색 중... (최대 2분 소요)"):
                cross_fractal = get_cross_stock_fractal(
                    ticker_code, 
                    current_prices,
                    similarity_threshold=similarity_threshold/100
                )
            
            if cross_fractal:
                avg_rise = cross_fractal["avg_rise"]
                avg_days = cross_fractal["avg_days"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📈 평균 예상 상승률", f"{avg_rise:+.2f}%")
                with col2:
                    st.metric("📅 평균 최고가 도달일", f"{avg_days:.1f}일")
                with col3:
                    st.metric("🔍 발견된 유사 패턴", f"{len(cross_fractal['valid_cases'])}개")
                
                # 목표가
                target_price = int(current_price * (1 + avg_rise / 100))
                st.success(f"🎯 **AI 예상 목표가**: {target_price:,}원 (현재가 대비 {avg_rise:+.2f}%)")
                
                # 차트
                fig = go.Figure()
                
                # 현재 패턴 (빨간색)
                fig.add_trace(go.Scatter(
                    x=list(range(14)),
                    y=current_prices.tolist(),
                    mode='lines+markers',
                    name=f'{company_name} 현재 패턴',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                ))
                
                # 유사 패턴 (다른 종목들)
                colors = ['gray', 'lightgray', 'darkgray', 'silver', 'dimgray', 
                         'slategray', 'lightslategray', 'darkslategray', 'gainsboro', 'whitesmoke']
                
                for idx, case in enumerate(cross_fractal["valid_cases"][:5]):  # 상위 5개만 표시
                    past_with_future = case["past_with_future"]
                    
                    # 과거 14일
                    fig.add_trace(go.Scatter(
                        x=list(range(14)),
                        y=past_with_future[:14],
                        mode='lines',
                        name=f'{case["name"]} ({case["date"]}, {case["similarity"]:.1%})',
                        line=dict(color=colors[idx], width=2, dash='solid')
                    ))
                    
                    # 미래 14일
                    fig.add_trace(go.Scatter(
                        x=list(range(13, 28)),
                        y=past_with_future[13:28],
                        mode='lines',
                        name=f'{case["name"]} 미래 (+{case["rise_pct"]:.1f}%)',
                        line=dict(color=colors[idx], width=2, dash='dot'),
                        showlegend=False
                    ))
                
                # 현재 시점
                fig.add_vline(
                    x=13.5,
                    line=dict(color='blue', width=2, dash='dash'),
                    annotation_text="현재 시점",
                    annotation_position="top"
                )
                
                fig.update_layout(
                    title=f"{company_name} vs 전체 시장 유사 패턴 비교 (유사도 {similarity_threshold}% 이상)",
                    xaxis_title="거래일 (14일 이후는 유사 사례의 미래 흐름)",
                    yaxis_title="정규화된 주가",
                    hovermode='x unified',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"cross_chart_{ticker_code}_{similarity_threshold}")
                
                # 상세 통계
                with st.expander("📋 유사 패턴 상세 통계 (전체 종목)"):
                    for idx, case in enumerate(cross_fractal["valid_cases"], 1):
                        st.markdown(f"""
                        **패턴 {idx}** - **{case['name']}** ({case['ticker']})
                        - 발생 시점: {case['date']}
                        - 유사도: {case['similarity']:.2%}
                        - 14일 후 상승률: {case['rise_pct']:+.2f}%
                        - 최고가 도달: {case['days_to_max']}일째
                        """)
                        st.divider()
                
                st.success(f"""
                ✅ **결론**: 현재 {company_name}의 차트 패턴과 유사한 패턴이 과거 다른 종목에서 {len(cross_fractal['valid_cases'])}번 발생했으며,
                평균적으로 **{avg_days:.1f}일 후 {avg_rise:+.2f}% 변동**했습니다.
                """)
                
            else:
                st.warning(f"""
                ⚠️ 유사도 {similarity_threshold}% 이상의 패턴을 다른 종목에서 찾지 못했습니다.
                
                💡 **해결 방법**:
                - 사이드바에서 유사도 기준을 낮춰보세요 (90~93%)
                - 현재 차트가 매우 독특한 패턴일 수 있습니다
                - 시장 전체가 비슷한 흐름일 때는 유사 패턴이 적을 수 있습니다
                """)
        else:
            st.error("현재 종목의 데이터가 부족합니다 (최소 14일 필요).")
            
    except Exception as e:
        st.error(f"데이터 로딩 오류: {str(e)}")

# --- 푸터 ---
st.divider()
st.caption("© 2026 AI 주식 팩트 스캐너 | 크로스 종목 유사 패턴 분석 + 명확한 호재/악재 분석")
