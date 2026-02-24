import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import json
import time
from bs4 import BeautifulSoup
import os

# 페이지 설정
st.set_page_config(page_title="AI 주식 팩트 스캐너", page_icon="📊", layout="wide")

# API 키 로드
def get_api_key(key_name):
    key = os.getenv(key_name)
    if key:
        return key
    try:
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    return None

GEMINI_API_KEY = get_api_key("GEMINI_API_KEY")
NAVER_CLIENT_ID = get_api_key("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = get_api_key("NAVER_CLIENT_SECRET")

# Gemini API 설정
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        GEMINI_MODEL = None
else:
    GEMINI_MODEL = None

# 주요 한국 주식 종목 매핑
STOCK_NAME_TO_CODE = {
    "삼성전자": "005930", "sk하이닉스": "000660", "삼성바이오로직스": "207940",
    "현대차": "005380", "셀트리온": "068270", "카카오": "035720",
    "naver": "035420", "네이버": "035420", "lg화학": "051910",
    "lg전자": "066570", "현대모비스": "012330", "삼성물산": "028260",
    "포스코홀딩스": "005490", "kb금융": "105560", "신한지주": "055550",
    "삼성sdi": "006400", "기아": "000270", "하나금융지주": "086790",
    "sk이노베이션": "096770", "lg생활건강": "051900", "키움증권": "039490",
    "미래에셋증권": "006800", "하이브": "352820"
}

# 세션 스테이트
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None

def reset_session():
    st.cache_data.clear()

def parse_ticker_input(user_input):
    user_input = user_input.strip().lower()
    if user_input.isdigit():
        return user_input, None
    if user_input in STOCK_NAME_TO_CODE:
        code = STOCK_NAME_TO_CODE[user_input]
        name = next((k for k, v in STOCK_NAME_TO_CODE.items() if v == code), None)
        return code, name.upper() if name else None
    return user_input, None

# 헤더
st.title("📊 AI 주식 투자 판단 스캐너")
st.markdown("**💡 핵심 질문: 지금 이 가격에서 사야 할까, 말아야 할까?**")
st.markdown("---")

# 종목 입력
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input(
        "종목 코드 또는 이름 입력", 
        key="ticker_input",
        on_change=reset_session,
        placeholder="예: 삼성전자, 005930, 카카오"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("🔍 분석 시작", type="primary", use_container_width=True)

if not ticker_input or not analyze_button:
    st.info("👆 종목을 입력하고 '분석 시작' 버튼을 클릭하세요.")
    st.stop()

ticker_code, parsed_name = parse_ticker_input(ticker_input)

if st.session_state.current_ticker != ticker_code:
    st.session_state.current_ticker = ticker_code
    reset_session()

yf_ticker = ticker_code + ".KS" if ticker_code.isdigit() else ticker_code

# 종목 정보 가져오기
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_info(yf_ticker, max_retries=3):
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(yf_ticker)
            hist = ticker.history(period="5d")
            
            if hist.empty:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None, None, None
            
            current_price = hist['Close'].iloc[-1]
            
            try:
                info = ticker.info
                company_name = info.get('longName', info.get('shortName', '알 수 없음'))
            except:
                company_name = "알 수 없음"
            
            return company_name, current_price, hist
        except:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, None, None
    return None, None, None

with st.spinner("📊 종목 정보 로딩..."):
    company_name, current_price, hist_data = get_stock_info(yf_ticker)

if not company_name or not current_price:
    st.error(f"❌ 종목을 찾을 수 없습니다: `{ticker_input}`")
    st.stop()

display_name = parsed_name if parsed_name else company_name

st.header(f"🏢 {display_name} ({ticker_code})")
st.subheader(f"💰 현재가: {current_price:,.0f} 원")

if hist_data is not None and len(hist_data) >= 2:
    prev_price = hist_data['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    if price_change > 0:
        st.markdown(f"📈 **전일 대비**: +{price_change:,.0f}원 (+{price_change_pct:.2f}%)")
    elif price_change < 0:
        st.markdown(f"📉 **전일 대비**: {price_change:,.0f}원 ({price_change_pct:.2f}%)")

st.markdown("---")

# === 핵심 1: 뉴스 분석 (강력 판단) ===
st.subheader("📰 뉴스 기반 투자 판단")

@st.cache_data(ttl=1800, show_spinner=False)
def get_news_data(company_name, max_news=5):
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return []
    try:
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        params = {"query": company_name, "display": max_news, "sort": "date"}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        items = response.json().get('items', [])
        news_list = []
        for item in items[:max_news]:
            title = BeautifulSoup(item['title'], 'html.parser').get_text()
            desc = BeautifulSoup(item['description'], 'html.parser').get_text()
            news_list.append({'title': title, 'description': desc, 'link': item['link']})
            time.sleep(0.2)
        return news_list
    except:
        return []

def analyze_news_for_investment(ticker, company_name, news_item):
    """뉴스 기반 투자 판단 (강력 판단 강제)"""
    if not GEMINI_MODEL:
        return "중립", 50, "AI 없음"
    
    try:
        prompt = f"""당신은 주식 투자자입니다. **{company_name} ({ticker})** 주식을 **지금 사야 할지** 판단하세요.

**뉴스**:
- 제목: {news_item['title']}
- 내용: {news_item['description']}

---

### ⚡ 판단 규칙 (엄격 적용)

**🟢 강력 매수 (80~100점)**
- 실적 급증 (영업이익 +20% 이상)
- 대형 수주 (1000억 이상)
- 혁신 신제품
- 경쟁사 몰락 (시장 독점 기회)

**🟢 약한 매수 (60~79점)**
- 실적 개선 (+10~20%)
- 중형 수주 (100~1000억)
- 자회사 호재

**🔴 약한 매도 (40~59점)**
- 실적 둔화 (-5~-10%)
- 소형 악재
- 업계 불확실성

**🔴 강력 매도 (0~39점)**
- 실적 급락 (-10% 이상)
- 대형 소송/리콜
- 경영 위기
- 규제 강타

---

### 📤 출력 (JSON)
{{
  "action": "매수" or "매도" or "관망",
  "score": 0~100,
  "reason": "30자 이내 핵심 근거"
}}

**중요**: 관망은 마지막 선택지. 명확히 판단하세요!"""

        response = GEMINI_MODEL.generate_content(
            prompt,
            generation_config={'temperature': 0.0, 'top_p': 0.7, 'max_output_tokens': 300},
            request_options={'timeout': 20}
        )
        
        text = response.text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        result = json.loads(text)
        action = result.get('action', '관망')
        score = max(0, min(100, int(result.get('score', 50))))
        reason = result.get('reason', '')[:50]
        
        return action, score, reason
    except:
        return "관망", 50, "분석 실패"

# 뉴스 분석
if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET:
    with st.spinner("📰 뉴스 분석 중..."):
        news_list = get_news_data(display_name)
        
        if news_list:
            analyzed_news = []
            for news in news_list:
                action, score, reason = analyze_news_for_investment(ticker_code, display_name, news)
                analyzed_news.append({**news, 'action': action, 'score': score, 'reason': reason})
            
            avg_score = sum(n['score'] for n in analyzed_news) / len(analyzed_news)
            
            buy_count = sum(1 for n in analyzed_news if n['action'] == '매수')
            sell_count = sum(1 for n in analyzed_news if n['action'] == '매도')
            hold_count = sum(1 for n in analyzed_news if n['action'] == '관망')
            
            # 최종 판단
            if avg_score >= 65:
                final_decision = "🟢 **매수 추천**"
                decision_color = "green"
                action_text = "지금 투자하기 좋은 시점입니다."
            elif avg_score <= 45:
                final_decision = "🔴 **매수 비추천**"
                decision_color = "red"
                action_text = "지금은 투자를 보류하는 것이 좋습니다."
            else:
                final_decision = "🟡 **신중 판단 필요**"
                decision_color = "orange"
                action_text = "추가 정보를 확인한 후 결정하세요."
            
            st.markdown(f"### {final_decision} (평균 **{avg_score:.0f}점**)")
            st.markdown(f"**뉴스 판단**: 매수 신호 **{buy_count}건**, 매도 신호 **{sell_count}건**, 관망 **{hold_count}건**")
            st.markdown(f"**투자 의견**: {action_text}")
            
            with st.expander("📋 개별 뉴스 분석", expanded=False):
                for i, news in enumerate(analyzed_news, 1):
                    if news['action'] == "매수":
                        emoji = "🟢"
                    elif news['action'] == "매도":
                        emoji = "🔴"
                    else:
                        emoji = "🟡"
                    
                    st.markdown(f"""
**{i}. {emoji} [{news['action']} {news['score']}점]** {news['title']}
- 📝 {news['reason']}
- 🔗 [원문]({news['link']})
""")
                    st.markdown("---")
        else:
            st.info("ℹ️ 최근 뉴스가 없습니다.")
else:
    st.warning("⚠️ 네이버 API 키 필요")

st.markdown("---")

# === 핵심 2: 차트 기반 투자 판단 ===
st.subheader("📈 차트 기반 투자 판단")

@st.cache_data(ttl=600, show_spinner=False)
def get_chart_data(yf_ticker, days=60):
    try:
        ticker = yf.Ticker(yf_ticker)
        hist = ticker.history(period=f"{days+10}d")
        if len(hist) < days:
            return None
        return hist.tail(days)
    except:
        return None

def analyze_chart_for_investment(ticker, current_price, hist_data):
    """차트 패턴 기반 투자 판단"""
    if not GEMINI_MODEL or hist_data is None or len(hist_data) < 20:
        return None
    
    try:
        # 최근 20일 가격
        recent_prices = hist_data['Close'].tail(20).values
        
        # 기술적 지표 계산
        prices = hist_data['Close'].values
        
        # 이동평균선
        ma5 = prices[-5:].mean()
        ma20 = prices[-20:].mean()
        ma60 = prices[-60:].mean() if len(prices) >= 60 else ma20
        
        # RSI (간단 버전)
        changes = np.diff(prices[-14:])
        gains = changes[changes > 0].sum()
        losses = abs(changes[changes < 0].sum())
        rsi = 100 - (100 / (1 + (gains / (losses + 0.0001))))
        
        # 볼린저 밴드
        bb_std = prices[-20:].std()
        bb_upper = ma20 + 2 * bb_std
        bb_lower = ma20 - 2 * bb_std
        
        # MACD (간단)
        ema12 = prices[-12:].mean()
        ema26 = prices[-26:].mean() if len(prices) >= 26 else ema12
        macd = ema12 - ema26
        
        # 가격 정규화
        min_p = recent_prices.min()
        max_p = recent_prices.max()
        if max_p > min_p:
            normalized = ((recent_prices - min_p) / (max_p - min_p) * 100).tolist()
        else:
            normalized = [50.0] * len(recent_prices)
        
        prompt = f"""당신은 주식 차트 전문가입니다. **{ticker}** 주식을 **지금 {current_price:,.0f}원에 사야 할지** 판단하세요.

**📊 차트 데이터**:
- 현재가: {current_price:,.0f}원
- 5일 이평: {ma5:,.0f}원
- 20일 이평: {ma20:,.0f}원
- 60일 이평: {ma60:,.0f}원
- RSI: {rsi:.1f}
- 볼린저 상단: {bb_upper:,.0f}원
- 볼린저 하단: {bb_lower:,.0f}원
- MACD: {macd:.2f}
- 최근 20일 추세: {normalized}

---

### 📐 분석 기준

**🟢 매수 신호**:
- 골든크로스 (단기>장기 이평)
- RSI < 30 (과매도)
- 볼린저 하단 이탈 후 반등
- 상승 추세선 형성
- 지지선 터치 후 반등

**🔴 매도 신호**:
- 데드크로스 (단기<장기 이평)
- RSI > 70 (과매수)
- 볼린저 상단 돌파 후 하락
- 하락 추세선 형성
- 저항선 실패

---

### 📤 출력 (JSON)
{{
  "decision": "매수" or "매도" or "관망",
  "score": 0~100,
  "theory": "적용한 이론 (예: 골든크로스, RSI 과매도)",
  "target_price": 목표가 (정수),
  "reason": "30자 이내 판단 근거"
}}

명확히 판단하세요!"""

        response = GEMINI_MODEL.generate_content(
            prompt,
            generation_config={'temperature': 0.0, 'top_p': 0.8, 'max_output_tokens': 400},
            request_options={'timeout': 20}
        )
        
        text = response.text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        result = json.loads(text)
        
        return {
            'decision': result.get('decision', '관망'),
            'score': max(0, min(100, int(result.get('score', 50)))),
            'theory': result.get('theory', ''),
            'target_price': int(result.get('target_price', current_price)),
            'reason': result.get('reason', '')[:50],
            'ma5': ma5,
            'ma20': ma20,
            'ma60': ma60,
            'rsi': rsi
        }
    except Exception as e:
        return None

with st.spinner("📊 차트 분석 중..."):
    chart_data = get_chart_data(yf_ticker, days=60)
    
    if chart_data is not None:
        analysis = analyze_chart_for_investment(ticker_code, current_price, chart_data)
        
        if analysis:
            # 결과 표시
            if analysis['decision'] == "매수":
                decision_emoji = "🟢"
                decision_text = "**매수 추천**"
            elif analysis['decision'] == "매도":
                decision_emoji = "🔴"
                decision_text = "**매수 비추천**"
            else:
                decision_emoji = "🟡"
                decision_text = "**신중 판단**"
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### {decision_emoji} {decision_text}")
                st.markdown(f"**신뢰도**: {analysis['score']}점")
                st.markdown(f"**적용 이론**: {analysis['theory']}")
                st.markdown(f"**목표가**: {analysis['target_price']:,.0f}원")
                st.markdown(f"**판단 근거**: {analysis['reason']}")
                
                st.markdown("---")
                
                st.markdown(f"""
**📊 기술적 지표**:
- 5일 이평: {analysis['ma5']:,.0f}원
- 20일 이평: {analysis['ma20']:,.0f}원
- 60일 이평: {analysis['ma60']:,.0f}원
- RSI: {analysis['rsi']:.1f}
""")
            
            with col2:
                # 차트 그리기
                fig = go.Figure()
                
                prices = chart_data['Close'].values
                dates = chart_data.index
                
                # 종가
                fig.add_trace(go.Scatter(
                    x=dates, y=prices,
                    mode='lines',
                    name='종가',
                    line=dict(color='royalblue', width=2)
                ))
                
                # 이동평균선
                fig.add_trace(go.Scatter(
                    x=dates[-20:],
                    y=[analysis['ma20']] * 20,
                    mode='lines',
                    name='20일 이평',
                    line=dict(color='orange', width=1, dash='dash')
                ))
                
                # 현재가 표시
                fig.add_trace(go.Scatter(
                    x=[dates[-1]],
                    y=[current_price],
                    mode='markers',
                    name='현재가',
                    marker=dict(size=12, color='red', symbol='star')
                ))
                
                # 목표가 표시
                if analysis['target_price'] != current_price:
                    fig.add_trace(go.Scatter(
                        x=[dates[-1]],
                        y=[analysis['target_price']],
                        mode='markers',
                        name='목표가',
                        marker=dict(size=10, color='green', symbol='diamond')
                    ))
                
                fig.update_layout(
                    title=f"{analysis['theory']} 분석 (최근 60일)",
                    xaxis_title="날짜",
                    yaxis_title="가격 (원)",
                    height=450,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ 차트 패턴 분석 실패")
    else:
        st.warning("⚠️ 차트 데이터 부족")

st.markdown("---")

# 최종 종합 판단
st.subheader("🎯 최종 투자 판단")

if 'analyzed_news' in locals() and analyzed_news and analysis:
    news_score = avg_score
    chart_score = analysis['score']
    
    # 가중 평균 (뉴스 40%, 차트 60%)
    final_score = news_score * 0.4 + chart_score * 0.6
    
    if final_score >= 65:
        final_emoji = "🟢"
        final_text = "**매수 추천**"
        final_advice = f"뉴스와 차트 모두 긍정적입니다. 목표가 {analysis['target_price']:,.0f}원을 노려보세요."
    elif final_score <= 45:
        final_emoji = "🔴"
        final_text = "**매수 비추천**"
        final_advice = "현재는 투자를 보류하고 추세 전환을 기다리세요."
    else:
        final_emoji = "🟡"
        final_text = "**신중 판단 필요**"
        final_advice = "단기 변동성이 큽니다. 분할 매수를 고려하세요."
    
    st.markdown(f"## {final_emoji} {final_text} (종합 {final_score:.0f}점)")
    st.markdown(f"**뉴스 점수**: {news_score:.0f}점 (40%)")
    st.markdown(f"**차트 점수**: {chart_score:.0f}점 (60%)")
    st.markdown(f"**투자 조언**: {final_advice}")

st.markdown("---")
st.caption(f"💡 분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("⚠️ 본 분석은 투자 참고용이며, 투자 결정은 본인의 책임입니다.")
