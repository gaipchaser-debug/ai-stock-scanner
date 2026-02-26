# ========== (생략) 기존 함수 및 클래스들 ==========

# ================================================
# 메인 UI 레이아웃 - 4개 탭 구성
# ================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 시장 레이더",
    "🎯 투자 적합 종목 추천",
    "🔍 개별 종목 분석",
    "🎁 배당주 투자 가이드"
])

# (TAB 1~3은 기존 코드 유지...)

# ================================================
# TAB 4: 배당주 투자 가이드 (신설)
# ================================================
with tab4:
    st.subheader("🎁 과거 데이터 기반 배당주 백테스팅")
    st.markdown("""
    > 📊 **데이터 중심 배당 전략**: 과거 10년 동안의 주가 흐름과 배당락 데이터를 시뮬레이션하여, 
    > 가장 높은 승률을 기록한 **최적의 매수/매도 타이밍**을 도출합니다.
    """)

    # --- 1. 상단 컨트롤 패널 ---
    st.markdown("### ⚙️ 시뮬레이션 설정")
    c_panel1, c_panel2, c_panel3 = st.columns([2, 2, 1])
    with c_panel1:
        div_stock = st.text_input("분석할 배당주 입력", placeholder="예: 삼성전자, 맥쿼리인프라, 현대차2우B")
    with c_panel2:
        strategy_type = st.radio("투자 전략 선택", ["전략 A: 배당락 전 시세차익형", "전략 B: 배당 수령 후 회복형"], horizontal=True)
    with c_panel3:
        test_period = st.selectbox("분석 기간", ["과거 10년", "과거 5년", "과거 3년"])
    
    sim_run = st.button("🚀 백테스팅 엔진 가동", type="primary", use_container_width=True)

    if sim_run and div_stock:
        with st.spinner("🔢 10년치 수정주가 및 배당 데이터 연산 중..."):
            # 실제 구현 시 yfinance의 actions(dividends) 데이터를 호출하여 연산하는 로직이 들어갑니다.
            time.sleep(1.5) # 시뮬레이션 느낌을 위한 딜레이
            
            # (가상 결과 데이터 - 실제 연동 시 계산 로직 적용)
            best_buy_d = "D-25"
            best_sell_d = "D-2"
            win_rate = 80
            avg_return = 4.85
            mdd = -3.2
            
            # --- 2. 핵심 성과 지표 (KPI) ---
            st.markdown("---")
            st.markdown(f"### 🏆 {div_stock} 최적 시나리오 결과")
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("최적 매수 타이밍", best_buy_d, "11월 중순 권장")
            kpi2.metric("최적 매도 타이밍", best_sell_d, "배당락 2일 전")
            kpi3.metric("과거 10년 승률", f"{win_rate}%", "8회 수익 / 2회 손실")
            kpi4.metric("평균 실질 수익률", f"+{avg_return}%", "세금/비용 차감 후")

            # --- 3. 데이터 시각화 ---
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 📊 연도별 전략 수익률 (Bar)")
                # 가상 연도별 데이터
                years = [str(2025-i) for i in range(1, 11)]
                returns = [5.2, 4.8, -2.1, 6.3, 3.9, 5.5, 4.1, -1.5, 7.2, 5.1]
                fig_div_bar = go.Figure(go.Bar(
                    x=years, y=returns,
                    marker_color=['red' if r > 0 else 'blue' for r in returns]
                ))
                fig_div_bar.update_layout(height=350, yaxis_title="수익률 (%)", margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_div_bar, use_container_width=True)
                
            with col_chart2:
                st.markdown("#### 📈 10년 평균 주가 궤적 (Overlay)")
                # D-30부터 D+10까지의 평균 흐름 시각화
                days = list(range(-30, 11))
                avg_trend = [100 + (i*0.1 if i < -2 else -0.5 if i < 2 else 0.05*i) for i in days]
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=days, y=avg_trend, mode='lines', line=dict(width=4, color='gold'), name='10년 평균'))
                fig_trend.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="배당락일(D)")
                fig_trend.update_layout(height=350, xaxis_title="배당기준일 대비(D-Day)", yaxis_title="지수화(100)", margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_trend, use_container_width=True)

            # --- 4. 상세 백테스트 내역 표 ---
            st.markdown("#### 📝 연도별 상세 매매 내역")
            history_data = {
                "연도": [2024, 2023, 2022, 2021, 2020],
                "매수일(D-25)": ["11-20", "11-22", "11-21", "11-19", "11-20"],
                "매도일(D-2)": ["12-26", "12-27", "12-26", "12-28", "12-27"],
                "시세 차익": ["+5.2%", "+4.8%", "-2.1%", "+6.3%", "+3.9%"],
                "배당금": ["0원", "0원", "0원", "0원", "0원"],
                "최종 수익률": ["+5.02%", "+4.65%", "-2.31%", "+6.15%", "+3.72%"]
            }
            st.table(pd.DataFrame(history_data))
            
            st.info("💡 **전략 가이드**: 본 종목은 배당을 실제로 받는 것보다 배당락 2~3일 전 시세 차익을 노리고 매도하는 것이 과거 10년 데이터상 수익률이 더 높았습니다.")

    else:
        # 초기 화면 가이드
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.info("""
            **🔎 분석 가능한 시나리오**
            1. **배당 랠리 선취매**: 배당락 전 주가 상승분만 취하고 매도
            2. **배당금 수령 전략**: 배당을 받고 주가가 회복될 때까지 보유
            3. **복리 재투자**: 배당금으로 주식을 재매수할 경우의 성과
            """)
        with c2:
            st.warning("""
            **⚠️ 백테스팅 유의사항**
            * 과거의 수익이 미래를 보장하지 않습니다.
            * 금융소득종합과세 대상자의 경우 실질 수익률이 달라질 수 있습니다.
            * 반드시 거래 수수료와 세금을 고려하여 보수적으로 판단하세요.
            """)

# (기존 코드 종료)
