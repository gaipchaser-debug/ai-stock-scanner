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

# --- [1단계] 데이터 수집 (캐시 제거 - 실시간 조회) ---
def get_stock_data(ticker_code, yf_ticker, days_ago=0):
    """
    days_ago=0: 오늘 현재가
    days_ago=7: 7일 전 종가
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
            "display": 5, 
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
            
            # 본문 크롤링 (최대 3개)
            content = ""
            if len(news_data) < 3:
                content = crawl_news_content(link)
                time.sleep(0.3)  # 크롤링 간격
            
            news_data.append({
                "title": title,
                "link": link,
                "description": description,
                "content": content
            })
        
        return news_data[:5]
        
    except Exception as e:
        return []


# --- [2단계] AI 검증 (강화된 분석) ---
def analyze_validity(ticker, news_data):
    """뉴스 본문 기반 세밀한 AI 분석"""
    if not news_data:
        return {
            "correlation_score": 50, 
            "reason": "최신 뉴스가 없어 중립적으로 평가합니다."
        }
    
    # 뉴스 본문 구조화
    news_texts = []
    for idx, news in enumerate(news_data, 1):
        text = f"[뉴스 {idx}]\n제목: {news['title']}\n요약: {news['description']}"
        if news.get('content'):
            text += f"\n본문 일부: {news['content'][:500]}"
        news_texts.append(text)
    
    news_combined = "\n\n".join(news_texts)
    
    prompt = f"""당신은 전문 증권 애널리스트입니다. 다음 뉴스들을 종합 분석하여 주가 영향도를 0~100점으로 평가하세요.

**종목코드**: {ticker}

**최신 뉴스들**:
{news_combined}

**평가 기준**:
- **호재(긍정적, 70~100점)**: 실적 개선, 신규 계약, 기술 혁신, 시장 점유율 상승, 배당 확대, 긍정적 전망
- **중립(40~69점)**: 일반적 보도, 단순 사실 전달, 영향 불분명
- **악재(부정적, 0~39점)**: 실적 악화, 소송/규제, 경영진 문제, 시장 점유율 하락, 부정적 전망

**분석 방법**:
1. 각 뉴스의 핵심 내용을 파악
2. 주가에 미치는 직접적 영향 평가
3. 긍정/부정 요소의 강도와 지속성 고려
4. 종합 점수와 구체적 근거 제시

**응답 형식** (반드시 JSON만 출력):
{{"correlation_score": 점수(0~100 정수), "reason": "구체적 분석 근거(100자 이내)"}}
"""
    
    try:
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config={
                "temperature": 0.05,  # 더 일관된 분석
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 500
            }
        )
        
        response = model.generate_content(
            prompt,
            request_options={"timeout": 20}
        )
        
        text = response.text.strip()
        
        # JSON 추출 (3단계)
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        
        # { } 영역만 추출
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            text = text[start:end]
        
        result = json.loads(text.strip())
        
        # 필드 검증
        if "correlation_score" not in result or "reason" not in result:
            return {
                "correlation_score": 50,
                "reason": "AI 응답 형식 오류 - 중립으로 평가합니다."
            }
        
        # 점수 변환 및 클램프
        score = int(float(result["correlation_score"]))
        score = max(0, min(100, score))
        
        return {
            "correlation_score": score,
            "reason": result["reason"][:200]  # 최대 200자
        }
        
    except json.JSONDecodeError:
        return {
            "correlation_score": 50,
            "reason": "AI 응답 파싱 실패 - 중립으로 평가합니다."
        }
    except Exception as e:
        return {
            "correlation_score": 50,
            "reason": f"AI 분석 오류 ({str(e)[:30]}) - 중립으로 평가합니다."
        }


# --- [3단계] 프랙탈 통계 (7일 기준) ---
@st.cache_data(ttl=3600)
def get_fractal_statistics(yf_ticker, similarity_threshold=0.90, tail_threshold=0.95, days_offset=0):
    """
    과거 7일 차트 패턴 분석
    - 전체 7일 유사도 ≥ similarity_threshold
    - 후반 2일(약 28%) 유사도 ≥ tail_threshold
    - 미래 7일 예측
    """
    try:
        # 과거 5년 데이터
        end_date = datetime.now() - timedelta(days=days_offset)
        start_date = end_date - timedelta(days=365*5)
        
        df = yf.download(
            yf_ticker, 
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if len(df) < 40:  # 최소 40일 필요
            return None
        
        closes = df['Close'].values
        
        # 현재 7일 패턴
        current_window = closes[-7:]
        current_z = zscore(current_window)
        
        # 과거 패턴 검색
        similarities = []
        
        for i in range(len(closes) - 14):  # 7일 + 7일 미래
            past_window = closes[i:i+7]
            past_z = zscore(past_window)
            
            # 전체 유사도
            sim_full = 1 - cosine(current_z, past_z)
            
            # 후반 2일 유사도
            sim_tail = 1 - cosine(current_z[-2:], past_z[-2:])
            
            # 필터링
            if sim_full < similarity_threshold or sim_tail < tail_threshold:
                continue
            
            # 미래 7일 데이터
            future_7d = closes[i+7:i+14]
            if len(future_7d) < 7:
                continue
            
            base_price = closes[i+6]  # 7일째 종가
            max_price = future_7d.max()
            max_idx = future_7d.argmax()
            rise_pct = ((max_price - base_price) / base_price) * 100
            
            past_with_future = closes[i:i+14]
            
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
        st.error(f"프랙탈 분석 오류: {e}")
        return None


# --- [백테스팅] ---
def backtest_prediction(ticker_code, yf_ticker, days_ago=7):
    """7일 전 검색했다면?"""
    try:
        # 7일 전 데이터
        past_data = get_stock_data(ticker_code, yf_ticker, days_ago=days_ago)
        past_price = past_data["price"]
        
        if past_price == 0:
            return None
        
        # 7일 전 프랙탈 분석
        past_fractal = get_fractal_statistics(yf_ticker, days_offset=days_ago)
        
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
st.caption("뉴스 본문 분석 + 7일 차트 패턴 기반 백테스팅")

# --- 사이드바 ---
with st.sidebar:
    st.header("📌 사용 방법")
    st.markdown("""
    1️⃣ **종목명 또는 코드 입력** (예: 삼성전자, 005930)
    2️⃣ **실시간 뉴스 본문 AI 분석**
    3️⃣ **과거 7일 유사 차트 패턴 검색**
    4️⃣ **7일 전 백테스팅 결과** 확인
    """)
    
    st.divider()
    
    # 유사도 슬라이더
    st.subheader("🎚️ 프랙탈 분석 설정")
    similarity_threshold = st.slider(
        "전체 유사도",
        min_value=85,
        max_value=95,
        value=90,
        step=1,
        help="과거 7일 차트와 현재 7일 차트의 전체 유사도"
    )
    
    tail_threshold = st.slider(
        "후반 유사도",
        min_value=85,
        max_value=98,
        value=95,
        step=1,
        help="최근 2일 차트 패턴의 유사도 (더 정밀)"
    )
    
    st.divider()
    
    st.subheader("📋 인기 종목")
    popular_stocks = ["삼성전자", "SK하이닉스", "NAVER", "카카오", "LG화학", "셀트리온"]
    for stock in popular_stocks:
        st.markdown(f"• {stock}")

# --- 메인 영역 ---
user_input = st.text_input(
    "🔍 종목명 또는 코드 입력",
    placeholder="예: 삼성전자, 005930, 키움증권"
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
    st.subheader("⏮️ 일주일 전(7일 전) 검색했다면?")
    
    with st.spinner("백테스팅 분석 중..."):
        backtest = backtest_prediction(ticker_code, yf_ticker, days_ago=7)
    
    if backtest:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("7일 전 가격", f"{backtest['past_price']:,}원")
        
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
                color = "green"
            elif accuracy >= 60:
                star = "⭐⭐⭐⭐"
                color = "blue"
            elif accuracy >= 40:
                star = "⭐⭐⭐"
                color = "orange"
            else:
                star = "⭐⭐"
                color = "red"
            
            st.metric("예측 정확도", f"{accuracy:.1f}% {star}")
            st.caption(f"{direction} 방향 {'적중' if backtest['direction_match'] else '불일치'}")
    else:
        st.info("백테스팅 데이터가 부족합니다.")
    
    st.divider()
    
    # --- 뉴스 + AI 분석 ---
    st.subheader("📰 최신 뉴스 AI 분석")
    
    with st.spinner("뉴스 본문 수집 및 AI 분석 중..."):
        news_data = get_news_data(company_name)
        ai_result = analyze_validity(ticker_code, news_data)
    
    score = ai_result["correlation_score"]
    reason = ai_result["reason"]
    
    # 점수 시각화
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if score >= 70:
            status = "🟢 호재"
            color = "green"
        elif score >= 40:
            status = "🟡 중립"
            color = "orange"
        else:
            status = "🔴 악재"
            color = "red"
        
        st.markdown(f"### {status}")
        st.markdown(f"**점수**: {score}/100")
    
    with col2:
        st.markdown("**분석 근거**")
        st.info(reason)
    
    # 점수 바
    st.progress(score / 100)
    
    # 뉴스 목록
    if news_data:
        st.markdown("**📄 분석 뉴스 목록**")
        for idx, news in enumerate(news_data, 1):
            with st.expander(f"{idx}. {news['title'][:50]}..."):
                st.markdown(f"**링크**: [{news['link']}]({news['link']})")
                st.markdown(f"**요약**: {news['description']}")
                if news.get('content'):
                    st.markdown(f"**본문 일부**: {news['content'][:300]}...")
    else:
        st.warning("최신 뉴스를 찾을 수 없습니다.")
    
    st.divider()
    
    # --- 프랙탈 분석 ---
    st.subheader("📊 과거 7일 유사 차트 패턴 분석")
    
    with st.spinner("과거 5년 데이터 분석 중..."):
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
            x=list(range(7)),
            y=fractal["current_prices"],
            mode='lines+markers',
            name='현재 7일 패턴',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        # 과거 유사 패턴
        colors = ['gray', 'lightgray', 'darkgray', 'silver', 'dimgray']
        for idx, case in enumerate(fractal["valid_cases"]):
            past_with_future = case["past_with_future"]
            
            # 과거 7일 (실선)
            fig.add_trace(go.Scatter(
                x=list(range(7)),
                y=past_with_future[:7],
                mode='lines',
                name=f'{case["date"]} (전체 {case["similarity"]:.1%}, 후반 {case["tail_similarity"]:.1%})',
                line=dict(color=colors[idx % len(colors)], width=2, dash='solid'),
                showlegend=True
            ))
            
            # 미래 7일 (점선)
            fig.add_trace(go.Scatter(
                x=list(range(6, 14)),
                y=past_with_future[6:14],
                mode='lines',
                name=f'{case["date"]} 미래 (+{case["rise_pct"]:.1f}%)',
                line=dict(color=colors[idx % len(colors)], width=2, dash='dot'),
                showlegend=False
            ))
        
        # 현재 시점 구분선
        fig.add_vline(
            x=6.5, 
            line=dict(color='blue', width=2, dash='dash'),
            annotation_text="현재 시점",
            annotation_position="top"
        )
        
        fig.update_layout(
            title="과거 7일 + 미래 7일 차트 패턴 비교",
            xaxis_title="거래일 (7일 이후는 과거 사례의 미래 흐름)",
            yaxis_title="주가 (원)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker_code}")
        
        # 상세 통계
        with st.expander("📋 유사 패턴 상세 통계"):
            for idx, case in enumerate(fractal["valid_cases"], 1):
                st.markdown(f"""
                **패턴 {idx}** ({case['date']})
                - 전체 유사도: {case['similarity']:.1%}
                - 후반 2일 유사도: {case['tail_similarity']:.1%}
                - 7일 후 상승률: {case['rise_pct']:+.2f}%
                - 최고가 도달: {case['days_to_max']}일째
                """)
                st.divider()
    else:
        st.warning(f"⚠️ 유사도 {similarity_threshold}%(전체), {tail_threshold}%(후반) 기준으로 유사한 과거 패턴을 찾지 못했습니다.")
        st.info("💡 사이드바에서 유사도 기준을 낮춰보세요 (85~90%).")

# --- 푸터 ---
st.divider()
st.caption("© 2026 AI 주식 팩트 스캐너 | 본문 크롤링 + 7일 차트 패턴 기반 분석")
