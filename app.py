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
import traceback

try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False

# --- [환경 설정] ---
try:
    NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
    NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "")
    NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- [데이터 수집] ---
def get_stock_data(ticker_code, yf_ticker, days_ago=0):
    """주가 데이터 조회"""
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
                return {"price": 0, "date": "", "source": "오류"}
            
            return {
                "price": int(df['Close'].iloc[-1]),
                "date": df.index[-1].strftime('%Y-%m-%d %H:%M'),
                "source": "네이버 금융"
            }
        except:
            pass
    
    try:
        df = yf.download(yf_ticker, period="1d", progress=False)
        if df.empty:
            return {"price": 0, "date": "", "source": "오류"}
        
        return {
            "price": int(df['Close'].iloc[-1]),
            "date": df.index[-1].strftime('%Y-%m-%d'),
            "source": "Yahoo Finance"
        }
    except:
        return {"price": 0, "date": "", "source": "오류"}


def crawl_news_content(url):
    """뉴스 본문 크롤링"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return ""
        
        soup = BeautifulSoup(response.text, 'html.parser')
        article = soup.find('article') or soup.find('div', {'id': 'articleBodyContents'})
        if article:
            for script in article(['script', 'style']):
                script.decompose()
            return article.get_text(strip=True)[:1000]
        return ""
    except:
        return ""


def get_news_data(company_name):
    """뉴스 수집"""
    try:
        if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
            return []
        
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID, 
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        params = {"query": f"{company_name} 주가", "display": 10, "sort": "date"}
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code != 200:
            return []
        
        items = response.json().get('items', [])
        news_data = []
        
        for item in items:
            title = item['title'].replace("<b>","").replace("</b>","")
            link = item['link']
            description = item.get('description', '').replace("<b>","").replace("</b>","")
            
            if company_name not in title and company_name not in description:
                continue
            
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
        st.warning(f"뉴스 수집 오류: {str(e)}")
        return []


def analyze_single_news(ticker, news_item):
    """개별 뉴스 분석"""
    if not GEMINI_API_KEY:
        return {"sentiment": "중립", "score": 50, "reason": "API 키 없음"}
    
    text = f"[제목] {news_item['title']}\n[요약] {news_item['description']}"
    if news_item.get('content'):
        text += f"\n[본문] {news_item['content'][:800]}"
    
    prompt = f"""전문 증권 애널리스트로서 다음 뉴스를 분석하세요.

종목: {ticker}
뉴스: {text}

판단 기준:
🟢 호재 (70~100점): 실적↑, 수주, 신제품, 투자 유치
🔴 악재 (0~30점): 실적↓, 소송, 규제, 감원
🟡 중립 (40~60점): 단순 공지, 영향 불분명

JSON 출력:
{{"sentiment": "호재" or "악재" or "중립", "score": 0~100, "reason": "이유 50자"}}
"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt, request_options={"timeout": 15})
        text = response.text.strip()
        
        # JSON 추출
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            text = text[start:end]
        
        result = json.loads(text.strip())
        sentiment = result.get("sentiment", "중립")
        score = int(float(result.get("score", 50)))
        
        # 재검증
        if score >= 70:
            sentiment = "호재"
        elif score <= 30:
            sentiment = "악재"
        
        return {
            "sentiment": sentiment,
            "score": max(0, min(100, score)),
            "reason": result.get("reason", "")[:100]
        }
    except Exception as e:
        return {"sentiment": "중립", "score": 50, "reason": f"분석 오류"}


def analyze_validity(ticker, news_with_sentiment):
    """종합 분석"""
    if not news_with_sentiment:
        return {"correlation_score": 50, "reason": "뉴스 없음"}
    
    scores = [n["sentiment_data"]["score"] for n in news_with_sentiment]
    sentiments = [n["sentiment_data"]["sentiment"] for n in news_with_sentiment]
    
    avg_score = int(np.mean(scores))
    positive = sentiments.count("호재")
    negative = sentiments.count("악재")
    
    if positive > negative:
        reason = f"호재 {positive}개, 악재 {negative}개 - 긍정적"
    elif negative > positive:
        reason = f"악재 {negative}개, 호재 {positive}개 - 부정적"
    else:
        reason = f"호재 {positive}개, 악재 {negative}개 - 혼재"
    
    return {"correlation_score": avg_score, "reason": reason}


# --- [2주 전 백테스팅] ---
def backtest_2weeks_ago(ticker_code, yf_ticker):
    """2주 전 검색했다면?"""
    try:
        # 2주 전 데이터
        past_data = get_stock_data(ticker_code, yf_ticker, days_ago=14)
        past_price = past_data["price"]
        
        if past_price == 0:
            return None
        
        # 2주 전 프랙탈 분석
        df_past = yf.download(yf_ticker, period="3mo", progress=False)
        if len(df_past) < 28:
            return None
        
        # 2주 전 시점의 최근 14일
        two_weeks_ago_idx = len(df_past) - 14
        if two_weeks_ago_idx < 14:
            return None
        
        past_14d = df_past['Close'].values[two_weeks_ago_idx-14:two_weeks_ago_idx]
        
        # 간단한 예측: 이전 14일 평균 상승률
        if len(past_14d) >= 14:
            avg_change = ((past_14d[-1] - past_14d[0]) / past_14d[0]) * 100
            predicted_price = int(past_price * (1 + avg_change / 100))
        else:
            return None
        
        # 실제 현재 가격
        current_data = get_stock_data(ticker_code, yf_ticker, days_ago=0)
        current_price = current_data["price"]
        
        actual_rise = ((current_price - past_price) / past_price) * 100
        predicted_rise = avg_change
        
        # 정확도
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


# --- [크로스 종목 프랙탈] ---
def get_cross_stock_fractal(current_ticker, current_prices, similarity_threshold=0.95):
    """전체 KRX에서 유사 패턴 검색"""
    try:
        if not FDR_AVAILABLE:
            st.warning("FinanceDataReader가 필요합니다.")
            return None
        
        # 현재 패턴 정규화
        if np.std(current_prices) < 0.01:
            return None
        
        current_z = zscore(current_prices)
        
        # KRX 종목 로드
        krx_df = fdr.StockListing('KRX')
        krx_df = krx_df[krx_df['Code'] != current_ticker].head(50)  # 100→50개로 축소 (속도)
        
        similarities = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (_, row) in enumerate(krx_df.iterrows()):
            progress_bar.progress((idx + 1) / len(krx_df))
            status_text.text(f"검색 중... {idx+1}/{len(krx_df)} ({row['Name']})")
            
            ticker = row['Code']
            name = row['Name']
            yf_ticker = ticker + ".KS" if ticker[0] == '0' else ticker + ".KQ"
            
            try:
                df = yf.download(yf_ticker, period="1y", progress=False)
                if len(df) < 50:
                    continue
                
                closes = df['Close'].values
                
                # 14일 윈도우 스캔
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
        status_text.empty()
        
        if not similarities:
            return None
        
        # 상위 10개
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        valid_cases = similarities[:10]
        
        return {
            "avg_rise": round(np.mean([c["rise_pct"] for c in valid_cases]), 2),
            "avg_days": round(np.mean([c["days_to_max"] for c in valid_cases]), 1),
            "valid_cases": valid_cases,
            "current_prices": current_prices
        }
    except Exception as e:
        st.error(f"크로스 분석 오류: {str(e)}")
        return None


# --- [주식 목록] ---
@st.cache_data(ttl=3600)
def load_krx():
    if FDR_AVAILABLE:
        try:
            df = fdr.StockListing('KRX')
            return df[['Code', 'Name']]
        except:
            pass
    
    return pd.DataFrame({
        'Code': ['005930', '000660', '035420', '051910', '035720'],
        'Name': ['삼성전자', 'SK하이닉스', 'NAVER', 'LG화학', '카카오']
    })


# ==================================================================
# ========================= UI 레이아웃 ============================
# ==================================================================

st.set_page_config(page_title="AI 주식 팩트 스캐너", page_icon="📈", layout="wide")

# 세션 초기화
if 'last_input' not in st.session_state:
    st.session_state.last_input = None

st.title("📈 AI 주식 팩트 스캐너")
st.caption("크로스 종목 유사 패턴 분석 + 2주 전 백테스팅")

# 사이드바
with st.sidebar:
    st.header("📌 사용 방법")
    st.markdown("""
    1️⃣ 종목명/코드 입력
    2️⃣ 뉴스 호재/악재 분석
    3️⃣ 2주 전 백테스팅
    4️⃣ 전체 시장 유사 패턴 검색
    """)
    
    st.divider()
    similarity_threshold = st.slider("유사도 기준 (%)", 90, 99, 95, 1)
    
    st.divider()
    st.subheader("📋 인기 종목")
    for stock in ["삼성전자", "SK하이닉스", "NAVER", "카카오", "LG화학"]:
        st.markdown(f"• {stock}")

# 메인
user_input = st.text_input("🔍 종목명 또는 코드 입력", placeholder="예: 삼성전자, 005930")

if user_input:
    # 입력 변경 감지
    if st.session_state.last_input != user_input:
        st.session_state.last_input = user_input
        st.cache_data.clear()
    
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
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 현재가
    stock_data = get_stock_data(ticker_code, yf_ticker)
    if stock_data["price"] > 0:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("💵 현재가", f"{stock_data['price']:,}원")
            st.caption(f"📅 {stock_data['date']} ({stock_data['source']})")
    
    st.divider()
    
    # === 2주 전 백테스팅 ===
    st.subheader("⏮️ 2주 전(14일 전) 검색했다면?")
    
    with st.spinner("백테스팅 분석 중..."):
        backtest = backtest_2weeks_ago(ticker_code, yf_ticker)
    
    if backtest:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("2주 전 가격", f"{backtest['past_price']:,}원")
        with col2:
            st.metric("AI 예측", f"{backtest['predicted_price']:,}원", 
                     f"{backtest['predicted_rise']:+.2f}%")
        with col3:
            st.metric("실제(오늘)", f"{backtest['actual_price']:,}원",
                     f"{backtest['actual_rise']:+.2f}%")
        with col4:
            accuracy = backtest['accuracy']
            direction = "✅" if backtest['direction_match'] else "❌"
            star = "⭐" * min(5, int(accuracy / 20) + 1)
            st.metric("정확도", f"{accuracy:.1f}% {star}")
            st.caption(f"{direction} 방향 {'적중' if backtest['direction_match'] else '불일치'}")
    else:
        st.info("💡 백테스팅 데이터 부족 (최소 1개월 데이터 필요)")
    
    st.divider()
    
    # 뉴스 분석
    st.subheader("📰 최신 뉴스 호재/악재 분석")
    
    try:
        with st.spinner("뉴스 수집 및 AI 분석 중..."):
            news_data = get_news_data(company_name)
            
            news_with_sentiment = []
            for news in news_data:
                sentiment_result = analyze_single_news(ticker_code, news)
                news_with_sentiment.append({**news, "sentiment_data": sentiment_result})
            
            ai_result = analyze_validity(ticker_code, news_with_sentiment)
        
        score = ai_result["correlation_score"]
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if score >= 70:
                st.markdown("### 🟢 호재")
            elif score >= 40:
                st.markdown("### 🟡 중립")
            else:
                st.markdown("### 🔴 악재")
            st.markdown(f"**점수**: {score}/100")
        
        with col2:
            st.info(ai_result["reason"])
        
        st.progress(score / 100)
        
        if news_with_sentiment:
            st.markdown("**📄 개별 뉴스**")
            for news in news_with_sentiment:
                s = news["sentiment_data"]
                emoji = "🟢" if s["sentiment"]=="호재" else ("🔴" if s["sentiment"]=="악재" else "🟡")
                
                with st.expander(f"{emoji} {s['sentiment']} ({s['score']}점) | {news['title'][:50]}..."):
                    st.markdown(f"**근거**: {s['reason']}")
                    st.markdown(f"**링크**: {news['link']}")
        else:
            st.warning("뉴스를 찾을 수 없습니다.")
            
    except Exception as e:
        st.error(f"뉴스 분석 오류: {str(e)}")
    
    st.divider()
    
    # 크로스 종목 분석
    st.subheader("🌐 전체 시장 유사 패턴 분석")
    st.info("💡 현재 차트와 유사한 패턴을 **다른 종목**에서도 검색합니다.")
    
    try:
        df_current = yf.download(yf_ticker, period="3mo", progress=False)
        if len(df_current) >= 14:
            current_prices = df_current['Close'].values[-14:]
            
            with st.spinner("🔍 KRX 50개 종목 검색 중... (1~2분)"):
                cross_fractal = get_cross_stock_fractal(
                    ticker_code, 
                    current_prices,
                    similarity_threshold=similarity_threshold/100
                )
            
            if cross_fractal:
                avg_rise = cross_fractal["avg_rise"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📈 평균 상승률", f"{avg_rise:+.2f}%")
                with col2:
                    st.metric("📅 최고가 도달", f"{cross_fractal['avg_days']:.1f}일")
                with col3:
                    st.metric("🔍 발견 패턴", f"{len(cross_fractal['valid_cases'])}개")
                
                target_price = int(stock_data['price'] * (1 + avg_rise / 100))
                st.success(f"🎯 예상 목표가: {target_price:,}원 ({avg_rise:+.2f}%)")
                
                # 차트
                fig = go.Figure()
                
                # 현재
                fig.add_trace(go.Scatter(
                    x=list(range(14)),
                    y=current_prices.tolist(),
                    mode='lines+markers',
                    name=f'{company_name} 현재',
                    line=dict(color='red', width=3)
                ))
                
                # 유사 패턴
                colors = ['gray', 'lightgray', 'darkgray', 'silver', 'dimgray']
                for idx, case in enumerate(cross_fractal["valid_cases"][:5]):
                    past_with_future = case["past_with_future"]
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(14)),
                        y=past_with_future[:14],
                        mode='lines',
                        name=f'{case["name"]} ({case["similarity"]:.1%})',
                        line=dict(color=colors[idx%5], width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(13, 28)),
                        y=past_with_future[13:28],
                        mode='lines',
                        line=dict(color=colors[idx%5], width=2, dash='dot'),
                        showlegend=False
                    ))
                
                fig.add_vline(x=13.5, line=dict(color='blue', dash='dash'))
                fig.update_layout(
                    title=f"{company_name} vs 시장 유사 패턴",
                    xaxis_title="거래일",
                    yaxis_title="주가",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📋 상세 통계"):
                    for idx, case in enumerate(cross_fractal["valid_cases"], 1):
                        st.markdown(f"""
                        **패턴 {idx}** - {case['name']} ({case['ticker']})
                        - 날짜: {case['date']}
                        - 유사도: {case['similarity']:.2%}
                        - 14일 후: {case['rise_pct']:+.2f}%
                        """)
                        st.divider()
            else:
                st.warning(f"유사도 {similarity_threshold}% 이상 패턴을 찾지 못했습니다. 기준을 낮춰보세요.")
        else:
            st.error("데이터 부족 (최소 14일 필요)")
    except Exception as e:
        st.error(f"크로스 분석 오류: {str(e)}\n{traceback.format_exc()}")

st.divider()
st.caption("© 2026 AI 주식 팩트 스캐너")
