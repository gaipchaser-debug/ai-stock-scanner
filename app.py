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
    """안전하게 API 키 가져오기"""
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
        st.error(f"⚠️ Gemini API 초기화 실패: {str(e)}")
        GEMINI_MODEL = None
else:
    GEMINI_MODEL = None

# 주요 한국 주식 종목 매핑 (종목명 → 코드)
STOCK_NAME_TO_CODE = {
    "삼성전자": "005930",
    "sk하이닉스": "000660",
    "삼성바이오로직스": "207940",
    "현대차": "005380",
    "셀트리온": "068270",
    "카카오": "035720",
    "naver": "035420",
    "네이버": "035420",
    "lg화학": "051910",
    "lg전자": "066570",
    "현대모비스": "012330",
    "삼성물산": "028260",
    "포스코홀딩스": "005490",
    "kb금융": "105560",
    "신한지주": "055550",
    "삼성sdi": "006400",
    "기아": "000270",
    "하나금융지주": "086790",
    "sk이노베이션": "096770",
    "lg생활건강": "051900",
    "삼성생명": "032830",
    "현대중공업": "009540",
    "sk텔레콤": "017670",
    "삼성화재": "000810",
    "한국전력": "015760",
    "kt": "030200",
    "포스코인터내셔널": "047050",
    "엔씨소프트": "036570",
    "삼성엔지니어링": "028050",
    "카카오뱅크": "323410",
    "lg": "003550",
    "cj": "001040",
    "현대건설": "000720",
    "sk": "034730",
    "롯데케미칼": "011170",
    "삼성전기": "009150",
    "한화솔루션": "009830",
    "고려아연": "010130",
    "대한항공": "003490",
    "sk바이오팜": "326030",
    "두산에너빌리티": "034020",
    "카카오게임즈": "293490",
    "lg이노텍": "011070",
    "크래프톤": "259960",
    "삼성중공업": "010140",
    "코웨이": "021240",
    "한미반도체": "042700",
    "키움증권": "039490",
    "미래에셋증권": "006800",
    "하이브": "352820",
    "삼성바이오에피스": "207940",
    "sk스퀘어": "402340",
    "제일기획": "030000",
    "한국조선해양": "009540",
    "lg유플러스": "032640",
    "현대글로비스": "086280",
    "두산밥캣": "241560",
    "한화": "000880",
    "아모레퍼시픽": "090430",
    "gs건설": "006360",
    "호텔신라": "008770",
    "하이닉스": "000660",
    "셀트리온헬스케어": "091990",
    "셀트리온제약": "068760",
    "에코프로": "086520",
    "에코프로비엠": "247540",
    "포스코케미칼": "003670",
    "lg디스플레이": "034220",
    "한국타이어": "161390",
    "넷마블": "251270",
    "쿠팡": "CPNG",
    "카카오페이": "377300",
    "카카오엔터": "035760"
}

# 세션 스테이트 초기화
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = None

def reset_session():
    """세션 상태 완전 초기화"""
    st.cache_data.clear()
    st.session_state.last_analysis_time = datetime.now()

def parse_ticker_input(user_input):
    """
    사용자 입력을 종목 코드로 변환
    - 숫자만 있으면 코드로 간주
    - 한글/영문이 있으면 종목명으로 간주하여 코드 검색
    """
    user_input = user_input.strip().lower()
    
    # 숫자만 있으면 종목 코드로 간주
    if user_input.isdigit():
        return user_input, None
    
    # 종목명 매핑에서 검색
    if user_input in STOCK_NAME_TO_CODE:
        code = STOCK_NAME_TO_CODE[user_input]
        # 종목명 반환 (대문자 변환)
        name = next((k for k, v in STOCK_NAME_TO_CODE.items() if v == code), None)
        return code, name.upper() if name else None
    
    # 매핑에 없으면 그대로 반환 (사용자가 정확한 코드 입력했을 수 있음)
    return user_input, None

# 헤더
st.title("📊 AI 주식 팩트 스캐너 (전문 트레이더 에디션)")
st.markdown("---")

# 종목 입력 섹션
col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input(
        "종목 코드 또는 이름 입력 (예: 삼성전자, 005930, 카카오, 035720)", 
        key="ticker_input",
        on_change=reset_session
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("🔍 분석 시작", type="primary", use_container_width=True)

if not ticker_input or not analyze_button:
    st.info("👆 종목 코드 또는 이름을 입력하고 '분석 시작' 버튼을 클릭하세요.")
    st.markdown("""
### 💡 사용 가능한 입력 예시:
- **종목 코드**: `005930` (삼성전자), `035720` (카카오), `051910` (LG화학)
- **종목 이름**: `삼성전자`, `카카오`, `LG화학`, `네이버`, `SK하이닉스`
""")
    st.stop()

# 종목 코드 파싱
ticker_code, parsed_name = parse_ticker_input(ticker_input)

# 종목 코드 변경 감지
if st.session_state.current_ticker != ticker_code:
    st.session_state.current_ticker = ticker_code
    reset_session()

# yfinance 티커 형식 생성 (한국 주식은 .KS, 미국 주식은 그대로)
if ticker_code.isdigit():
    yf_ticker = ticker_code + ".KS"
else:
    yf_ticker = ticker_code  # 미국 주식 (예: AAPL, CPNG)

# 종목 정보 가져오기 (재시도 로직 추가)
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_info(yf_ticker, max_retries=3):
    """주식 기본 정보 가져오기 (재시도 포함)"""
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(yf_ticker)
            
            # 먼저 history로 가격 확인
            hist = ticker.history(period="5d")
            
            if hist.empty:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None, None, None
            
            current_price = hist['Close'].iloc[-1]
            
            # info 가져오기 (실패해도 계속 진행)
            try:
                info = ticker.info
                company_name = info.get('longName', info.get('shortName', '알 수 없음'))
            except:
                company_name = "알 수 없음"
            
            return company_name, current_price, hist
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                st.error(f"❌ 데이터 조회 실패 (시도 {attempt + 1}/{max_retries}): {str(e)}")
                return None, None, None
    
    return None, None, None

with st.spinner("📊 종목 정보를 불러오는 중..."):
    company_name, current_price, hist_data = get_stock_info(yf_ticker)

if not company_name or not current_price:
    st.error(f"""
❌ **종목을 찾을 수 없습니다**: `{ticker_input}`

### 가능한 원인:
1. 종목 코드가 올바르지 않음
2. 해당 종목이 상장폐지되었거나 거래 정지 상태
3. 일시적인 데이터 제공 문제

### 해결 방법:
- **종목 코드 확인**: 6자리 숫자 (예: 005930)
- **종목명 사용**: 정확한 한글 이름 (예: 삼성전자, 카카오)
- 잠시 후 다시 시도해보세요
""")
    st.stop()

# 파싱된 이름이 있으면 우선 사용, 없으면 yfinance에서 가져온 이름 사용
display_name = parsed_name if parsed_name else company_name

# 종목 헤더 표시
st.header(f"🏢 {display_name} ({ticker_code})")
st.subheader(f"💰 현재가: {current_price:,.0f} 원")

# 최근 가격 변동 표시
if hist_data is not None and len(hist_data) >= 2:
    prev_price = hist_data['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    if price_change > 0:
        st.markdown(f"📈 **전일 대비**: +{price_change:,.0f}원 (+{price_change_pct:.2f}%)", unsafe_allow_html=True)
    elif price_change < 0:
        st.markdown(f"📉 **전일 대비**: {price_change:,.0f}원 ({price_change_pct:.2f}%)", unsafe_allow_html=True)
    else:
        st.markdown(f"📊 **전일 대비**: 보합", unsafe_allow_html=True)

st.markdown("---")

# 뉴스 분석 섹션
st.subheader("📰 최신 뉴스 분석")

@st.cache_data(ttl=1800, show_spinner=False)
def get_news_data(company_name, max_news=5):
    """네이버 뉴스 검색"""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return []
    
    try:
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        params = {
            "query": company_name,
            "display": max_news,
            "sort": "date"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        items = response.json().get('items', [])
        news_list = []
        
        for item in items[:max_news]:
            title = BeautifulSoup(item['title'], 'html.parser').get_text()
            desc = BeautifulSoup(item['description'], 'html.parser').get_text()
            link = item['link']
            
            news_list.append({
                'title': title,
                'description': desc,
                'link': link
            })
            time.sleep(0.2)
        
        return news_list
        
    except Exception as e:
        return []

def analyze_news_with_ai(ticker, news_item):
    """개별 뉴스 AI 분석"""
    if not GEMINI_MODEL:
        return "중립", 50, "AI 분석 불가"
    
    try:
        prompt = f"""당신은 20년 경력의 전문 증권 애널리스트입니다.

다음 뉴스가 주식 '{ticker}'에 미치는 영향을 분석하세요.

**뉴스 제목**: {news_item['title']}
**뉴스 내용**: {news_item['description']}

**판단 기준**:
- **호재** (70~100점): 매출/이익 증가, 신제품 출시, 대규모 계약, 긍정적 실적 발표, 투자 유치, 기술 혁신
- **악재** (0~30점): 매출/이익 감소, 소송/규제, 부정적 실적, 경영 위기, 리콜, 사고
- **중립** (31~69점): 단순 공시, 인사 이동, 애매한 내용

**출력 형식** (JSON):
{{
  "sentiment": "호재" or "악재" or "중립",
  "score": 0~100 사이 정수,
  "reason": "50자 이내 판단 근거"
}}

반드시 JSON 형식으로만 답변하세요."""

        response = GEMINI_MODEL.generate_content(
            prompt,
            generation_config={
                'temperature': 0.0,
                'top_p': 0.8,
                'max_output_tokens': 300
            },
            request_options={'timeout': 15}
        )
        
        text = response.text.strip()
        
        # JSON 추출
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        result = json.loads(text)
        
        sentiment = result.get('sentiment', '중립')
        score = max(0, min(100, int(result.get('score', 50))))
        reason = result.get('reason', '')[:100]
        
        return sentiment, score, reason
        
    except Exception as e:
        return "중립", 50, f"분석 오류"

# 뉴스 수집 및 분석
if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET:
    with st.spinner("📰 뉴스를 수집하고 분석 중..."):
        news_list = get_news_data(display_name)
        
        if news_list:
            analyzed_news = []
            for news in news_list:
                sentiment, score, reason = analyze_news_with_ai(ticker_code, news)
                analyzed_news.append({
                    **news,
                    'sentiment': sentiment,
                    'score': score,
                    'reason': reason
                })
            
            # 평균 점수 계산
            avg_score = sum(n['score'] for n in analyzed_news) / len(analyzed_news)
            
            # 감정별 카운트
            sentiment_counts = {
                '호재': sum(1 for n in analyzed_news if n['sentiment'] == '호재'),
                '악재': sum(1 for n in analyzed_news if n['sentiment'] == '악재'),
                '중립': sum(1 for n in analyzed_news if n['sentiment'] == '중립')
            }
            
            # 종합 평가
            if avg_score >= 65:
                overall = "🟢 전반적으로 호재"
            elif avg_score <= 35:
                overall = "🔴 전반적으로 악재"
            else:
                overall = "🟡 중립적"
            
            st.markdown(f"### {overall} (평균 {avg_score:.1f}점)")
            st.markdown(f"**분석 뉴스**: 총 {len(analyzed_news)}건 (호재 {sentiment_counts['호재']}건, 악재 {sentiment_counts['악재']}건, 중립 {sentiment_counts['중립']}건)")
            
            # 개별 뉴스 표시
            with st.expander("📋 개별 뉴스 분석 결과 보기", expanded=True):
                for i, news in enumerate(analyzed_news, 1):
                    emoji = "🟢" if news['sentiment'] == "호재" else "🔴" if news['sentiment'] == "악재" else "🟡"
                    
                    st.markdown(f"""
**{i}. {emoji} [{news['sentiment']}] {news['title']}**
- 점수: {news['score']}점
- 이유: {news['reason']}
- [원문 보기]({news['link']})
""")
                    st.markdown("---")
        else:
            st.info("ℹ️ 최근 뉴스를 찾을 수 없습니다.")
else:
    st.warning("⚠️ 네이버 API 키가 설정되지 않아 뉴스 분석을 건너뜁니다.")

st.markdown("---")

# 차트 패턴 분석 섹션
st.subheader("📈 전문 트레이더 차트 패턴 분석")

CHART_PATTERNS = {
    "지속형": {
        "Pennant (페넌트)": {"signal": "매수", "target": "+5~8%", "stop": "-3%"},
        "Bullish Flag (강세 깃발)": {"signal": "매수", "target": "+8~12%", "stop": "-3%"},
        "Channel Up (상승 채널)": {"signal": "매수", "target": "+5~10%", "stop": "-3%"}
    },
    "중립형": {
        "Symmetrical Triangle (대칭 삼각형)": {"signal": "관망", "target": "±5~8%", "stop": "±3%"},
        "Ascending Triangle (상승 삼각형)": {"signal": "매수", "target": "+7~10%", "stop": "-4%"},
        "Descending Triangle (하락 삼각형)": {"signal": "매도", "target": "-7~10%", "stop": "+4%"}
    },
    "반전형": {
        "Double Bottom (쌍바닥)": {"signal": "매수", "target": "+8~15%", "stop": "-3%"},
        "Head and Shoulders (머리어깨)": {"signal": "매도", "target": "-10~20%", "stop": "+5%"},
        "Cup and Handle (컵 손잡이)": {"signal": "매수", "target": "+15~30%", "stop": "-5%"}
    },
    "특수형": {
        "Falling Wedge (하락 쐐기)": {"signal": "매수", "target": "+10~15%", "stop": "-4%"},
        "Rising Wedge (상승 쐐기)": {"signal": "매도", "target": "-10~15%", "stop": "+4%"}
    }
}

@st.cache_data(ttl=600, show_spinner=False)
def get_recent_prices(yf_ticker, days=20):
    """최근 N일 가격 데이터"""
    try:
        ticker = yf.Ticker(yf_ticker)
        hist = ticker.history(period=f"{days+10}d")
        
        if len(hist) < days:
            return None
        
        prices = hist['Close'].tail(days).values
        return prices
    except:
        return None

def normalize_prices(prices):
    """가격을 0~100으로 정규화"""
    min_p = prices.min()
    max_p = prices.max()
    
    if max_p == min_p:
        return np.full_like(prices, 50.0)
    
    return (prices - min_p) / (max_p - min_p) * 100

def detect_chart_pattern_with_ai(ticker, normalized_prices):
    """AI로 차트 패턴 감지"""
    if not GEMINI_MODEL:
        return None, 0, None
    
    try:
        pattern_desc = "\n".join([
            f"**{category}**:\n" + "\n".join([f"  - {name}" for name in patterns.keys()])
            for category, patterns in CHART_PATTERNS.items()
        ])
        
        prompt = f"""당신은 전문 차트 분석가입니다.

다음은 주식 '{ticker}'의 최근 20일 가격 데이터를 0~100으로 정규화한 값입니다:
{normalized_prices.tolist()}

아래 차트 패턴 중 가장 유사한 패턴을 찾으세요:

{pattern_desc}

**출력 형식** (JSON):
{{
  "pattern": "패턴 이름 (정확히 위 목록에서 선택)",
  "similarity": 0~100 사이 유사도,
  "category": "지속형/중립형/반전형/특수형"
}}

유사도가 60 미만이면 "패턴 없음"을 반환하세요.
반드시 JSON 형식으로만 답변하세요."""

        response = GEMINI_MODEL.generate_content(
            prompt,
            generation_config={
                'temperature': 0.1,
                'top_p': 0.9,
                'max_output_tokens': 300
            },
            request_options={'timeout': 20}
        )
        
        text = response.text.strip()
        
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        
        result = json.loads(text)
        
        pattern = result.get('pattern', '패턴 없음')
        similarity = max(0, min(100, int(result.get('similarity', 0))))
        category = result.get('category', None)
        
        if similarity < 60 or pattern == "패턴 없음":
            return None, 0, None
        
        return pattern, similarity, category
        
    except:
        return None, 0, None

# 가격 데이터 가져오기
with st.spinner("📊 차트 패턴 분석 중..."):
    prices = get_recent_prices(yf_ticker, days=20)
    
    if prices is not None:
        normalized = normalize_prices(prices)
        pattern_name, similarity, category = detect_chart_pattern_with_ai(ticker_code, normalized)
        
        if pattern_name and category:
            pattern_info = CHART_PATTERNS[category][pattern_name]
            
            # 목표가 및 손절가 계산
            target_str = pattern_info['target']
            stop_str = pattern_info['stop']
            
            if '~' in target_str:
                target_pct = float(target_str.replace('%', '').replace('+', '').replace('-', '').split('~')[1])
            else:
                target_pct = float(target_str.replace('%', '').replace('+', '').replace('-', ''))
            
            stop_pct = float(stop_str.replace('%', '').replace('-', '').replace('+', ''))
            
            if '매수' in pattern_info['signal']:
                target_price = current_price * (1 + target_pct / 100)
                stop_price = current_price * (1 - stop_pct / 100)
            else:
                target_price = current_price * (1 - target_pct / 100)
                stop_price = current_price * (1 + stop_pct / 100)
            
            # 결과 표시
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
### 🎯 감지된 패턴
                
**패턴 이름**: {pattern_name}  
**카테고리**: {category}  
**유사도**: {similarity}%  

---

### 📊 투자 전략

**신호**: {pattern_info['signal']}  
**목표가**: {target_price:,.0f}원 ({target_str})  
**손절가**: {stop_price:,.0f}원 ({stop_str})  
""")
            
            with col2:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    y=prices,
                    mode='lines+markers',
                    name='종가',
                    line=dict(color='royalblue', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f"{pattern_name} 패턴 (최근 20일)",
                    xaxis_title="일자",
                    yaxis_title="가격 (원)",
                    height=400,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ 명확한 차트 패턴이 감지되지 않았습니다. (유사도 < 60%)")
    else:
        st.warning("⚠️ 차트 데이터를 가져올 수 없습니다.")

st.markdown("---")

# 백테스팅 섹션
st.subheader("⏮️ 백테스팅 (2주 전 분석)")

@st.cache_data(ttl=600, show_spinner=False)
def backtest_2weeks_ago(yf_ticker):
    """2주 전 시점 백테스팅"""
    try:
        ticker = yf.Ticker(yf_ticker)
        hist = ticker.history(period="1mo")
        
        if len(hist) < 14:
            return None
        
        price_2w_ago = hist['Close'].iloc[-14]
        current_price_bt = hist['Close'].iloc[-1]
        
        recent_trend = hist['Close'].tail(10).pct_change().mean()
        predicted_price = price_2w_ago * (1 + recent_trend * 14)
        
        actual_change = (current_price_bt - price_2w_ago) / price_2w_ago * 100
        predicted_change = (predicted_price - price_2w_ago) / price_2w_ago * 100
        accuracy = 100 - abs(actual_change - predicted_change)
        accuracy = max(0, min(100, accuracy))
        
        direction_match = (actual_change > 0) == (predicted_change > 0)
        
        return {
            'past_price': price_2w_ago,
            'predicted_price': predicted_price,
            'current_price': current_price_bt,
            'accuracy': accuracy,
            'direction_match': direction_match
        }
    except:
        return None

bt_result = backtest_2weeks_ago(yf_ticker)

if bt_result:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("2주 전 가격", f"{bt_result['past_price']:,.0f}원")
    
    with col2:
        st.metric("AI 예측 가격", f"{bt_result['predicted_price']:,.0f}원")
    
    with col3:
        st.metric("실제 현재가", f"{bt_result['current_price']:,.0f}원")
    
    with col4:
        stars = "⭐" * int(bt_result['accuracy'] / 20)
        direction = "✅" if bt_result['direction_match'] else "❌"
        st.metric("정확도", f"{bt_result['accuracy']:.1f}% {stars}")
        st.markdown(f"**방향성**: {direction}")
else:
    st.info("ℹ️ 백테스팅 데이터가 부족합니다.")

st.markdown("---")
st.caption(f"💡 분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("⚠️ 본 분석은 투자 참고용이며, 투자 결정은 본인의 책임입니다.")
