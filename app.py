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

try:
    import FinanceDataReader as fdr
except ImportError:
    st.error("FinanceDataReader 설치가 필요합니다.")

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

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- [1단계] 데이터 수집 ---
@st.cache_data(ttl=600)
def get_stock_data(yf_ticker):
    try:
        df = yf.download(yf_ticker, period="5d", progress=False)
        if df.empty:
            return {"price": 0}
        current_price = int(df['Close'].iloc[-1])
        return {"price": current_price}
    except Exception as e:
        return {"price": 0}

@st.cache_data(ttl=600)
def get_news_data(company_name):
    try:
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID, 
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        params = {"query": company_name, "display": 5, "sort": "date"}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            items = response.json().get('items', [])
            return [item['title'].replace("<b>","").replace("</b>","") for item in items]
        else:
            return []
    except Exception as e:
        return []

# --- [2단계] AI 검증 ---
def analyze_validity(ticker, news_list):
    if not news_list:
        return {"correlation_score": 50, "reason": "최신 뉴스가 없어 중립적으로 평가합니다."}

    prompt = f"""종목코드: {ticker}
최신 뉴스: {news_list}

이 뉴스들이 주가에 미치는 영향을 0~100점으로 평가해주세요.
호재(긍정적): 70~100점
중립: 40~69점
악재(부정적): 0~39점

반드시 아래 JSON 형식으로만 답변:
{{"correlation_score": 점수, "reason": "이유"}}
"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # 마크다운 제거
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        
        result = json.loads(text.strip())
        return result
        
    except Exception as e:
        return {"correlation_score": 50, "reason": f"AI 분석 오류"}

# --- [3단계] 프랙탈 통계 ---
@st.cache_data(ttl=86400)
def get_fractal_statistics(yf_ticker):
    try:
        df = yf.download(yf_ticker, period="5y", progress=False)
        if len(df) < 40:
            return None
        
        closes = df['Close'].values.flatten()
        dates = df.index
        current_window = closes[-20:]
        current_z = zscore(current_window)
        
        sims = []
        future_window = 10
        
        for i in range(len(closes) - 20 - future_window):
            past_window = closes[i:i+20]
            past_z = zscore(past_window)
            sim = 1 - cosine(current_z, past_z)
            
            base_price = closes[i+19]
            future_prices = closes[i+20 : i+20+future_window]
            max_price = np.max(future_prices)
            max_idx = np.argmax(future_prices) + 1
            rise_pct = (max_price - base_price) / base_price * 100
            
            sims.append({
                "similarity": sim, 
                "date": dates[i+19].strftime('%Y-%m-%d'),
                "rise_pct": rise_pct,
                "days_to_max": max_idx,
                "past_prices": past_window
            })
            
        sims.sort(key=lambda x: x["similarity"], reverse=True)
        valid = [c for c in sims[:5] if c["similarity"] >= 0.80]
        
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

# --- [웹 UI] ---
st.set_page_config(page_title="AI 주식 스캐너", layout="wide", page_icon="📈")

# 한국 주식 목록 로드
@st.cache_data
def load_krx():
    try:
        return fdr.StockListing('KRX')
    except Exception as e:
        return pd.DataFrame()

krx_list = load_krx()

# 헤더
st.title("📈 나만의 AI 팩트 스캐너")
st.markdown("**AI 기반 실시간 주식 분석 시스템** - 뉴스 감성분석 + 프랙탈 패턴 인식")
st.markdown("---")

# 사이드바
with st.sidebar:
    st.header("📚 사용 가이드")
    st.markdown("""
    1. **종목 검색**: 종목명 또는 6자리 코드 입력
    2. **AI 분석**: 최신 뉴스의 호재/악재 자동 판단
    3. **프랙탈 분석**: 과거 유사 차트 패턴 발견
    4. **예측**: 과거 패턴 이후 평균 수익률 확인
    """)
    st.markdown("---")
    st.info("💡 **Tip**: 코스피는 0으로 시작, 코스닥은 그 외 숫자")

# 종목 검색
user_input = st.text_input(
    "🔍 분석할 종목명 또는 종목코드(6자리)를 입력하세요", 
    placeholder="예: 삼성전자 또는 005930"
)

if user_input:
    target_name, target_ticker = "", ""
    
    if user_input.isdigit():
        target_ticker = user_input.zfill(6)
        matched = krx_list[krx_list['Code'] == target_ticker]
        if not matched.empty:
            target_name = matched.iloc[0]['Name']
    else:
        target_name = user_input
        matched = krx_list[krx_list['Name'].str.contains(target_name, na=False)]
        if not matched.empty:
            target_ticker = matched.iloc[0]['Code']
            target_name = matched.iloc[0]['Name']
    
    if not target_name or krx_list.empty:
        st.error("❌ 종목을 찾을 수 없습니다. 정확한 종목명 또는 6자리 코드를 입력해주세요.")
    else:
        yf_ticker = f"{target_ticker}.KS" if target_ticker.startswith("0") else f"{target_ticker}.KQ"
        
        with st.spinner(f"'{target_name}' 데이터를 수집하고 AI로 분석 중입니다... ⏳"):
            stock_data = get_stock_data(yf_ticker)
            news_data = get_news_data(target_name)
            ai_result = analyze_validity(target_ticker, news_data)
            frac_result = get_fractal_statistics(yf_ticker)

        # 1. 종목명 및 현재가
        st.header(f"🏢 {target_name} ({target_ticker})")
        current_price = stock_data.get('price', 0)
        if current_price > 0:
            st.subheader(f"💵 현재가: **{current_price:,}원**")
        else:
            st.warning("⚠️ 현재가를 불러올 수 없습니다.")
        
        col1, col2 = st.columns([1, 1])
        
        # 2. 왼쪽: 뉴스 및 AI 스코어
        with col1:
            st.markdown("### 📰 최신 뉴스 분석")
            if news_data:
                for idx, news in enumerate(news_data, 1):
                    st.markdown(f"**{idx}.** {news}")
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

        # 3. 오른쪽: 프랙탈 분석
        with col2:
            st.markdown("### 📈 과거 5년 유사 차트 분석")
            if frac_result:
                avg_rise = frac_result['avg_rise']
                avg_days = frac_result['avg_days']
                
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
                    past_norm = (case['past_prices'] - case['past_prices'][0]) / case['past_prices'][0] * 100
                    fig.add_trace(go.Scatter(
                        y=past_norm, 
                        mode='lines', 
                        line=dict(color='rgba(150, 150, 150, 0.4)', width=2), 
                        name=f"과거 {i+1} ({case['date']})"
                    ))
                
                curr_norm = (frac_result['current_prices'] - frac_result['current_prices'][0]) / frac_result['current_prices'][0] * 100
                fig.add_trace(go.Scatter(
                    y=curr_norm, 
                    mode='lines+markers', 
                    line=dict(color='red', width=4), 
                    marker=dict(size=6),
                    name='현재 20일 흐름'
                ))
                
                fig.update_layout(
                    title="현재 vs 과거 유사 패턴 비교", 
                    xaxis_title="거래일", 
                    yaxis_title="수익률 (%)", 
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📊 상세 통계"):
                    for i, case in enumerate(frac_result['valid_cases'], 1):
                        st.write(
                            f"**패턴 {i}** ({case['date']}) - "
                            f"유사도: {case['similarity']:.2%} | "
                            f"상승률: {case['rise_pct']:+.2f}% | "
                            f"최고점: {case['days_to_max']}일"
                        )
            else:
                st.warning("⚠️ 과거 5년 내 유사한 차트 패턴을 찾을 수 없습니다.")

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Made with ❤️ by AI Stock Scanner"
    "</div>", 
    unsafe_allow_html=True
)
