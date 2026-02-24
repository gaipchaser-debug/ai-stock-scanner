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

# --- [차트 패턴 정의] ---
CHART_PATTERNS = {
    "CONTINUATION": {
        "Pennant": {
            "description": "상승 추세 중 짧은 조정 후 재상승",
            "signal": "매수 (상단 돌파 시)",
            "target": "+5~8% (기존 추세 연장)",
            "stop_loss": "-3% (패턴 하단)"
        },
        "Bullish_Flag": {
            "description": "강한 상승 후 수평 조정, 재상승",
            "signal": "매수 (상단 돌파 시)",
            "target": "+8~12% (깃대 길이만큼)",
            "stop_loss": "-3% (플래그 하단)"
        },
        "Channel_Up": {
            "description": "평행 상승 채널 유지",
            "signal": "매수 (하단 지지 시), 매도 (상단 저항 시)",
            "target": "+5~10% (채널 상단)",
            "stop_loss": "-3% (채널 이탈 시)"
        }
    },
    "NEUTRAL": {
        "Symmetrical_Triangle": {
            "description": "고점 낮아지고 저점 높아지며 수렴",
            "signal": "관망 (돌파 방향 확인 필요)",
            "target": "±5~8% (돌파 방향 따라)",
            "stop_loss": "±3% (패턴 이탈 시)"
        },
        "Ascending_Triangle": {
            "description": "수평 저항 + 상승 지지",
            "signal": "약한 매수 (상단 돌파 기대)",
            "target": "+7~10%",
            "stop_loss": "-4% (하단 이탈)"
        },
        "Descending_Triangle": {
            "description": "수평 지지 + 하락 저항",
            "signal": "약한 매도 (하단 이탈 우려)",
            "target": "-7~10%",
            "stop_loss": "+4% (상단 돌파 시)"
        }
    },
    "REVERSAL": {
        "Double_Top": {
            "description": "두 번 고점 형성 후 하락",
            "signal": "매도 (네크라인 이탈 시)",
            "target": "-8~15% (고점-저점 거리만큼)",
            "stop_loss": "+3% (고점 돌파 시)"
        },
        "Double_Bottom": {
            "description": "두 번 저점 형성 후 상승",
            "signal": "매수 (네크라인 돌파 시)",
            "target": "+8~15% (저점-고점 거리만큼)",
            "stop_loss": "-3% (저점 이탈 시)"
        },
        "Head_and_Shoulders": {
            "description": "고점(머리) 양쪽에 낮은 고점(어깨)",
            "signal": "매도 (네크라인 이탈 시)",
            "target": "-10~20% (머리-네크라인 거리만큼)",
            "stop_loss": "+5% (머리 돌파 시)"
        },
        "Inverse_Head_and_Shoulders": {
            "description": "저점(머리) 양쪽에 높은 저점(어깨)",
            "signal": "매수 (네크라인 돌파 시)",
            "target": "+10~20%",
            "stop_loss": "-5% (머리 이탈 시)"
        },
        "Cup_and_Handle": {
            "description": "U자형 바닥 + 작은 조정",
            "signal": "매수 (핸들 돌파 시)",
            "target": "+15~30% (컵 깊이만큼)",
            "stop_loss": "-5% (핸들 하단)"
        }
    },
    "SPECIAL": {
        "Falling_Wedge": {
            "description": "하락 쐐기 (좁아지는 하락)",
            "signal": "매수 (상단 돌파 시)",
            "target": "+10~15%",
            "stop_loss": "-4% (하단 이탈)"
        },
        "Rising_Wedge": {
            "description": "상승 쐐기 (좁아지는 상승)",
            "signal": "매도 (하단 이탈 시)",
            "target": "-10~15%",
            "stop_loss": "+4% (상단 돌파)"
        }
    }
}


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
        
        if score >= 70:
            sentiment = "호재"
        elif score <= 30:
            sentiment = "악재"
        
        return {
            "sentiment": sentiment,
            "score": max(0, min(100, score)),
            "reason": result.get("reason", "")[:100]
        }
    except:
        return {"sentiment": "중립", "score": 50, "reason": "분석 오류"}


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


# --- [차트 패턴 AI 분석] ---
def analyze_chart_pattern_ai(prices):
    """AI로 차트 패턴 인식"""
    if not GEMINI_API_KEY:
        return None
    
    # 최근 20일 데이터
    recent_20 = prices[-20:] if len(prices) >= 20 else prices
    
    # 정규화
    normalized = ((recent_20 - recent_20.min()) / (recent_20.max() - recent_20.min()) * 100).tolist()
    
    # 가격 변화 설명
    price_desc = f"최근 20일 가격 변화: {normalized}"
    
    # 패턴 목록
    pattern_list = []
    for category, patterns in CHART_PATTERNS.items():
        for name, info in patterns.items():
            pattern_list.append(f"- {name}: {info['description']}")
    
    prompt = f"""당신은 20년 경력의 프로 차트 분석가입니다.

**차트 데이터** (정규화 0~100):
{price_desc}

**알려진 차트 패턴들**:
{chr(10).join(pattern_list)}

**분석 요청**:
1. 위 차트 데이터가 어떤 패턴과 가장 유사한가?
2. 유사도는 몇 %인가? (60% 이상만 판단)
3. 현재 패턴이 완성 단계인가, 진행 중인가?

**출력 형식** (JSON):
{{
  "pattern_name": "패턴 이름 (예: Double_Bottom)",
  "similarity": 유사도 (0~100),
  "category": "CONTINUATION/NEUTRAL/REVERSAL/SPECIAL",
  "stage": "형성 중/완성/돌파 임박",
  "reason": "판단 근거 100자"
}}

유사도 60% 미만이면 {{"pattern_name": "None", "similarity": 0}}
"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt, request_options={"timeout": 20})
        text = response.text.strip()
        
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            text = text[start:end]
        
        result = json.loads(text.strip())
        
        if result.get("pattern_name") == "None" or result.get("similarity", 0) < 60:
            return None
        
        return result
        
    except:
        return None


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

st.set_page_config(page_title="AI 차트 패턴 투자 분석", page_icon="📈", layout="wide")

if 'last_input' not in st.session_state:
    st.session_state.last_input = None

st.title("📈 AI 차트 패턴 투자 분석")
st.caption("프로 트레이더의 차트 패턴 분석 + 투자 전략 제안")

# 사이드바
with st.sidebar:
    st.header("📌 차트 패턴 가이드")
    
    st.markdown("### 🟢 CONTINUATION (지속)")
    st.markdown("- Pennant, Flag, Channel")
    st.markdown("- 기존 추세 유지")
    
    st.markdown("### 🟡 NEUTRAL (중립)")
    st.markdown("- Triangle (대칭/상승/하락)")
    st.markdown("- 돌파 방향 확인 필요")
    
    st.markdown("### 🔴 REVERSAL (반전)")
    st.markdown("- Double Top/Bottom")
    st.markdown("- Head & Shoulders")
    st.markdown("- Cup & Handle")
    
    st.markdown("### 🟣 SPECIAL (특수)")
    st.markdown("- Wedge (쐐기)")
    st.markdown("- 강한 반전 신호")

# 메인
user_input = st.text_input("🔍 종목명 또는 코드 입력", placeholder="예: 삼성전자, 005930")

if user_input:
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
        st.metric("💵 현재가", f"{stock_data['price']:,}원", 
                 help=f"{stock_data['date']} ({stock_data['source']})")
    
    st.divider()
    
    # === 차트 패턴 AI 분석 ===
    st.subheader("📊 프로 트레이더 차트 패턴 분석")
    
    try:
        df = yf.download(yf_ticker, period="3mo", progress=False)
        if len(df) >= 20:
            prices = df['Close'].values
            
            with st.spinner("🔍 AI가 차트 패턴을 분석 중..."):
                pattern_result = analyze_chart_pattern_ai(prices)
            
            if pattern_result:
                pattern_name = pattern_result["pattern_name"]
                similarity = pattern_result["similarity"]
                category = pattern_result["category"]
                stage = pattern_result["stage"]
                reason = pattern_result["reason"]
                
                # 패턴 정보 가져오기
                pattern_info = None
                for cat, patterns in CHART_PATTERNS.items():
                    if pattern_name in patterns:
                        pattern_info = patterns[pattern_name]
                        break
                
                if pattern_info:
                    # 카테고리별 색상
                    if category == "CONTINUATION":
                        color = "🟢"
                        bg_color = "green"
                    elif category == "REVERSAL":
                        color = "🔴"
                        bg_color = "red"
                    elif category == "SPECIAL":
                        color = "🟣"
                        bg_color = "purple"
                    else:
                        color = "🟡"
                        bg_color = "orange"
                    
                    st.success(f"""
                    ### {color} **{pattern_name.replace('_', ' ')}** 패턴 감지!
                    
                    **카테고리**: {category}  
                    **유사도**: {similarity}%  
                    **단계**: {stage}
                    
                    **AI 판단 근거**:  
                    {reason}
                    """)
                    
                    # 투자 전략
                    st.markdown("---")
                    st.markdown("### 💡 **프로 트레이더 투자 전략**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"""
                        **📋 패턴 설명**  
                        {pattern_info['description']}
                        
                        **📈 추천 액션**  
                        {pattern_info['signal']}
                        """)
                    
                    with col2:
                        st.warning(f"""
                        **🎯 목표가**  
                        {pattern_info['target']}
                        
                        **⛔ 손절가**  
                        {pattern_info['stop_loss']}
                        """)
                    
                    # 구체적 가격 계산
                    current_price = stock_data["price"]
                    
                    # 목표가 계산 (간단한 예시)
                    if "+" in pattern_info['target']:
                        pct = float(pattern_info['target'].split('+')[1].split('~')[0])
                        target_price = int(current_price * (1 + pct / 100))
                        st.success(f"🎯 **예상 목표가**: {target_price:,}원 (현재가 대비 +{pct}%)")
                    
                    # 손절가 계산
                    if "-" in pattern_info['stop_loss']:
                        pct = float(pattern_info['stop_loss'].split('-')[1].split('%')[0])
                        stop_price = int(current_price * (1 - pct / 100))
                        st.error(f"⛔ **권장 손절가**: {stop_price:,}원 (현재가 대비 -{pct}%)")
                    
                    # 차트 표시
                    st.markdown("---")
                    st.markdown("### 📉 **최근 20일 차트**")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df.index[-20:],
                        open=df['Open'].values[-20:],
                        high=df['High'].values[-20:],
                        low=df['Low'].values[-20:],
                        close=df['Close'].values[-20:],
                        name='가격'
                    ))
                    
                    fig.update_layout(
                        title=f"{company_name} - {pattern_name.replace('_', ' ')} 패턴 ({similarity}% 유사)",
                        xaxis_title="날짜",
                        yaxis_title="가격 (원)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.info("패턴이 감지되었으나 상세 정보를 찾을 수 없습니다.")
            else:
                st.info("""
                💡 **명확한 차트 패턴이 감지되지 않았습니다.**
                
                - 현재 차트가 특정 패턴 형성 초기 단계일 수 있습니다.
                - 며칠 후 다시 확인하시면 패턴이 명확해질 수 있습니다.
                - 또는 현재 시장이 박스권 또는 불규칙한 흐름일 수 있습니다.
                """)
        else:
            st.error("차트 데이터가 부족합니다 (최소 20일 필요).")
    except Exception as e:
        st.error(f"차트 분석 오류: {str(e)}")
    
    st.divider()
    
    # 뉴스 분석
    st.subheader("📰 최신 뉴스 분석")
    
    try:
        with st.spinner("뉴스 수집 중..."):
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
        
        if news_with_sentiment:
            with st.expander("📄 개별 뉴스 보기"):
                for news in news_with_sentiment:
                    s = news["sentiment_data"]
                    emoji = "🟢" if s["sentiment"]=="호재" else ("🔴" if s["sentiment"]=="악재" else "🟡")
                    st.markdown(f"{emoji} **{s['sentiment']}** ({s['score']}점) | {news['title']}")
                    st.caption(f"근거: {s['reason']}")
                    st.markdown(f"[기사 링크]({news['link']})")
                    st.divider()
            
    except Exception as e:
        st.error(f"뉴스 분석 오류: {str(e)}")

st.divider()
st.caption("© 2026 AI 차트 패턴 투자 분석 | 프로 트레이더의 검증된 패턴 기반")
