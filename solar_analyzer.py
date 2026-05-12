import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="태양광 발전 분석기",
    page_icon="☀️",
    layout="wide",
)

# ── 커스텀 CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

  /* 배경 */
  .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%); }

  /* 사이드바 */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1e35 0%, #0a1525 100%);
    border-right: 1px solid #1e3a5f;
  }
  [data-testid="stSidebar"] * { color: #c8deff !important; }
  [data-testid="stSidebar"] .stSlider > div > div > div { background: #1e4080 !important; }

  /* 메트릭 카드 */
  .metric-card {
    background: linear-gradient(135deg, #0f2040 0%, #0a1830 100%);
    border: 1px solid #1e4a8a;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,120,255,0.15);
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0078ff, #00d4ff);
  }
  .metric-label {
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a9fd4;
    margin-bottom: 10px;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #00d4ff;
    line-height: 1;
  }
  .metric-unit {
    font-size: 1rem;
    color: #5a9fd4;
    margin-top: 6px;
  }

  /* 정보 배지 */
  .info-badge {
    background: rgba(30,64,128,0.4);
    border: 1px solid #1e4a8a;
    border-radius: 10px;
    padding: 14px 20px;
    margin: 6px 0;
    color: #a8c8f0;
    font-size: 0.88rem;
  }
  .info-badge strong { color: #00d4ff; }

  /* 섹션 헤더 */
  .section-header {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a8abf;
    margin: 28px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e3a5f;
  }

  /* 경고/상태 배너 */
  .status-banner {
    border-radius: 10px;
    padding: 12px 20px;
    margin: 16px 0;
    font-size: 0.9rem;
    font-weight: 600;
  }
  .status-good { background: rgba(0,180,80,0.15); border: 1px solid #00b450; color: #00e060; }
  .status-warn { background: rgba(255,160,0,0.15); border: 1px solid #ffa000; color: #ffb830; }
  .status-bad  { background: rgba(255,60,60,0.15);  border: 1px solid #ff3c3c; color: #ff6060; }

  /* 제목 */
  h1 { color: #ffffff !important; font-weight: 900 !important; letter-spacing: -0.02em; }
  h3 { color: #5a9fd4 !important; }
  p, li { color: #a8c8f0; }

  /* Plotly 배경 투명화 */
  .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── 물리 상수 & 모델 함수 ─────────────────────────────────────────────────────
T_STC = 25.0          # 표준 시험 조건(STC) 온도 [°C]
TEMP_COEFF = -0.004   # 온도 계수: -0.4 %/°C
NOCT = 45.0           # 공칭 동작 온도 [°C]  (Nominal Operating Cell Temperature)
G_STC = 1000.0        # STC 일사량 [W/m²]

def cell_temperature(T_amb: float, G: float) -> float:
    """
    NOCT 모델: 셀 온도 = 주변온도 + (NOCT-20)/800 × G
    IEC 61215 표준 기반
    """
    return T_amb + (NOCT - 20.0) / 800.0 * G

def effective_efficiency(eta_stc: float, T_cell: float) -> float:
    """
    온도 보정 효율: η_eff = η_STC × [1 + γ × (T_cell − T_STC)]
    γ = -0.004 /°C (단결정 실리콘 대표값)
    """
    return eta_stc * (1 + TEMP_COEFF * (T_cell - T_STC))

def power_output(G: float, A: float, eta_eff: float) -> float:
    """P = G × A × η_eff  [W]"""
    return G * A * eta_eff

def monthly_energy(P_kw: float, peak_hours: float = 3.5) -> float:
    """E_month = P_kw × peak_sun_hours × 30  [kWh]"""
    return P_kw * peak_hours * 30


# ── 사이드바 입력 ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ 입력 파라미터")
    st.markdown("---")

    st.markdown('<div class="section-header">☀ 태양 환경</div>', unsafe_allow_html=True)
    irradiance = st.slider("일사량 (W/m²)", min_value=100, max_value=1200,
                           value=800, step=50,
                           help="수평면 전일사량 (GHI). 맑은 날 정오 ≈ 1000 W/m²")
    peak_sun_hours = st.slider("일 평균 피크 일조 시간 (h/day)", 1.0, 8.0, 3.5, 0.5,
                               help="한국 평균 약 3~4 h/day")

    st.markdown('<div class="section-header">🔲 패널 사양</div>', unsafe_allow_html=True)
    panel_area = st.number_input("패널 총 면적 (m²)", min_value=1.0, max_value=500.0,
                                 value=20.0, step=1.0)
    eta_stc_pct = st.slider("STC 기준 패널 효율 (%)", min_value=10.0, max_value=25.0,
                             value=18.0, step=0.5,
                             help="제조사 데이터시트의 모듈 효율 (STC: 1000 W/m², 25°C)")
    eta_stc = eta_stc_pct / 100.0

    st.markdown('<div class="section-header">🌡 온도 환경</div>', unsafe_allow_html=True)
    T_amb = st.slider("외부 기온 (°C)", min_value=-10, max_value=50,
                      value=25, step=1)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#3a6a9f; line-height:1.7;">
    <b>적용 물리 모델</b><br>
    • 셀 온도: NOCT 모델 (IEC 61215)<br>
    • 효율 저하: γ = −0.4 %/°C<br>
    • 월 발전량: P × 피크시간 × 30일
    </div>
    """, unsafe_allow_html=True)


# ── 핵심 계산 ─────────────────────────────────────────────────────────────────
T_cell = cell_temperature(T_amb, irradiance)
eta_eff = effective_efficiency(eta_stc, T_cell)
eta_eff = max(eta_eff, 0.0)   # 음수 방지
P_w     = power_output(irradiance, panel_area, eta_eff)
P_kw    = P_w / 1000.0
E_month = monthly_energy(P_kw, peak_sun_hours)
eta_loss_pct = (eta_eff - eta_stc) / eta_stc * 100  # 음수 = 손실


# ── 헤더 ─────────────────────────────────────────────────────────────────────
st.markdown("# ☀️ 태양광 발전 시스템 효율 및 경제성 분석기")
st.markdown(
    "<p style='color:#4a8abf; font-size:0.9rem; margin-top:-12px;'>"
    "온도 계수 보정 모델(NOCT + γ = −0.4 %/°C) 기반 실시간 시뮬레이션</p>",
    unsafe_allow_html=True
)
st.markdown("---")


# ── 주요 메트릭 ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">⚡ 실시간 출력</div>
      <div class="metric-value">{P_kw:.2f}</div>
      <div class="metric-unit">kW</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">🗓 월 예상 발전량</div>
      <div class="metric-value">{E_month:.1f}</div>
      <div class="metric-unit">kWh / 월</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">🌡 셀 온도 (NOCT)</div>
      <div class="metric-value">{T_cell:.1f}</div>
      <div class="metric-unit">°C</div>
    </div>""", unsafe_allow_html=True)

with col4:
    loss_color = "#00e060" if eta_loss_pct >= 0 else "#ff6060"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">📉 온도 효율 변화</div>
      <div class="metric-value" style="color:{loss_color};">{eta_loss_pct:+.1f}</div>
      <div class="metric-unit">% (STC 대비)</div>
    </div>""", unsafe_allow_html=True)


# ── 상태 배너 ─────────────────────────────────────────────────────────────────
if T_cell < 35:
    banner_cls, icon, msg = "status-good", "✅", f"셀 온도 {T_cell:.1f}°C — 최적 동작 범위입니다."
elif T_cell < 55:
    banner_cls, icon, msg = "status-warn", "⚠️", f"셀 온도 {T_cell:.1f}°C — 출력 저하 주의 구간입니다."
else:
    banner_cls, icon, msg = "status-bad",  "🔴", f"셀 온도 {T_cell:.1f}°C — 고온으로 인한 심각한 출력 저하 중!"

st.markdown(f'<div class="status-banner {banner_cls}">{icon} {msg}</div>', unsafe_allow_html=True)


# ── 세부 정보 ─────────────────────────────────────────────────────────────────
with st.expander("📋 상세 계산 내역 보기"):
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"""
        <div class="info-badge">🌞 입력 일사량: <strong>{irradiance} W/m²</strong></div>
        <div class="info-badge">📐 패널 면적: <strong>{panel_area} m²</strong></div>
        <div class="info-badge">⚙️ STC 효율: <strong>{eta_stc_pct:.1f}%</strong></div>
        <div class="info-badge">🌡 외부 기온: <strong>{T_amb}°C</strong></div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div class="info-badge">🔥 셀 온도 (NOCT): <strong>{T_cell:.2f}°C</strong></div>
        <div class="info-badge">📉 보정 효율: <strong>{eta_eff*100:.2f}%</strong></div>
        <div class="info-badge">⚡ 출력 (W): <strong>{P_w:.1f} W</strong></div>
        <div class="info-badge">☀️ 피크 일조: <strong>{peak_sun_hours} h/day</strong></div>
        """, unsafe_allow_html=True)

    st.markdown("""
    **셀 온도 공식 (NOCT 모델):**  
    `T_cell = T_amb + (NOCT − 20) / 800 × G`  
    **효율 보정 공식:**  
    `η_eff = η_STC × [1 + γ × (T_cell − 25)]`,  γ = −0.004 /°C
    """)


# ── 그래프 1: 온도 vs 효율 ────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 온도 변화에 따른 발전 효율 곡선</div>',
            unsafe_allow_html=True)

T_range = np.linspace(-10, 70, 300)

# 현재 일사량 기준 셀 온도 범위
T_cell_range = T_range + (NOCT - 20) / 800 * irradiance
eta_range = np.array([max(effective_efficiency(eta_stc, tc), 0) for tc in T_cell_range])
eta_range_pct = eta_range * 100

fig1 = go.Figure()

# 효율 영역 채우기
fig1.add_trace(go.Scatter(
    x=T_range, y=eta_range_pct,
    fill='tozeroy',
    fillcolor='rgba(0,120,255,0.08)',
    line=dict(color='rgba(0,180,255,0.9)', width=2.5),
    name='보정 효율 (%)',
    hovertemplate='외기 온도: %{x:.1f}°C<br>보정 효율: %{y:.2f}%<extra></extra>'
))

# STC 기준선
fig1.add_hline(
    y=eta_stc_pct, line_dash="dash", line_color="#ffa500",
    annotation_text=f"STC 효율 {eta_stc_pct:.1f}%",
    annotation_position="top right",
    annotation_font_color="#ffa500"
)

# 현재 동작점
fig1.add_trace(go.Scatter(
    x=[T_amb], y=[eta_eff * 100],
    mode='markers',
    marker=dict(size=14, color='#00ffaa', symbol='circle',
                line=dict(color='white', width=2)),
    name='현재 동작점',
    hovertemplate=f'현재: {T_amb}°C → {eta_eff*100:.2f}%<extra></extra>'
))

# 위험 온도 영역
fig1.add_vrect(x0=50, x1=70, fillcolor="rgba(255,60,60,0.07)",
               line_width=0, annotation_text="고온 위험", annotation_position="top left",
               annotation_font_color="#ff6060")

fig1.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,20,40,0.6)',
    font=dict(family="Noto Sans KR", color="#a8c8f0"),
    xaxis=dict(
        title="외부 기온 (°C)", gridcolor='rgba(30,74,138,0.4)',
        color='#5a9fd4', zerolinecolor='#1e3a5f'
    ),
    yaxis=dict(
        title="보정 효율 (%)", gridcolor='rgba(30,74,138,0.4)',
        color='#5a9fd4', tickformat='.1f'
    ),
    legend=dict(bgcolor='rgba(10,20,40,0.8)', bordercolor='#1e4a8a', borderwidth=1),
    hovermode='x unified',
    margin=dict(l=60, r=40, t=20, b=60),
    height=380,
)

st.plotly_chart(fig1, use_container_width=True)


# ── 그래프 2: 일사량 × 온도 → 출력 히트맵 ────────────────────────────────────
st.markdown('<div class="section-header">🗺 일사량 × 외기온도 → 출력 히트맵 (kW)</div>',
            unsafe_allow_html=True)

G_axis = np.linspace(100, 1200, 60)
T_axis = np.linspace(-10, 50, 50)
Z = np.zeros((len(T_axis), len(G_axis)))

for i, t in enumerate(T_axis):
    for j, g in enumerate(G_axis):
        tc = cell_temperature(t, g)
        eff = max(effective_efficiency(eta_stc, tc), 0)
        Z[i, j] = power_output(g, panel_area, eff) / 1000  # kW

fig2 = go.Figure(data=go.Heatmap(
    z=Z, x=G_axis, y=T_axis,
    colorscale=[
        [0.0, '#0a1830'], [0.3, '#0044aa'],
        [0.6, '#0099ff'], [0.85, '#00ddff'], [1.0, '#ffffff']
    ],
  colorbar=dict(title=dict(text='출력 (kW)', font=dict(color='#a8c8f0')),
              tickfont=dict(color='#a8c8f0')),
    hovertemplate='일사량: %{x:.0f} W/m²<br>기온: %{y:.0f}°C<br>출력: %{z:.2f} kW<extra></extra>'
))

# 현재 동작점
fig2.add_trace(go.Scatter(
    x=[irradiance], y=[T_amb],
    mode='markers',
    marker=dict(size=14, color='#ff4444', symbol='x', line=dict(width=3, color='white')),
    name='현재 조건',
    hoverinfo='skip'
))

fig2.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,20,40,0.6)',
    font=dict(family="Noto Sans KR", color="#a8c8f0"),
    xaxis=dict(title="일사량 (W/m²)", color='#5a9fd4'),
    yaxis=dict(title="외기 온도 (°C)", color='#5a9fd4'),
    legend=dict(bgcolor='rgba(10,20,40,0.8)', bordercolor='#1e4a8a', borderwidth=1),
    margin=dict(l=60, r=20, t=20, b=60),
    height=380,
)

st.plotly_chart(fig2, use_container_width=True)


# ── 경제성 분석 ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💰 경제성 분석</div>', unsafe_allow_html=True)

electricity_price = 120  # 원/kWh (한국 평균 산업용)
annual_energy = E_month * 12
annual_revenue = annual_energy * electricity_price
payback_cost   = panel_area * 800_000  # 약 80만원/m² 설치비 가정
payback_years  = payback_cost / annual_revenue if annual_revenue > 0 else float('inf')

ec1, ec2, ec3 = st.columns(3)
with ec1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">📅 연간 발전량</div>
      <div class="metric-value" style="font-size:2rem;">{annual_energy:.0f}</div>
      <div class="metric-unit">kWh / 년</div>
    </div>""", unsafe_allow_html=True)
with ec2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">💵 연간 절감액</div>
      <div class="metric-value" style="font-size:2rem;">{annual_revenue/10000:.0f}</div>
      <div class="metric-unit">만원 / 년</div>
    </div>""", unsafe_allow_html=True)
with ec3:
    pb_str = f"{payback_years:.1f}" if payback_years < 100 else "∞"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">⏳ 단순 회수 기간</div>
      <div class="metric-value" style="font-size:2rem;">{pb_str}</div>
      <div class="metric-unit">년 (전기료 120원/kWh 기준)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:0.75rem; color:#3a6a9f; margin-top:16px; text-align:right;">
* 설치비 80만원/m², 전기요금 120원/kWh 기준 단순 추정값입니다. 실제 경제성은 지역, 설비, 보조금에 따라 다릅니다.
</div>
""", unsafe_allow_html=True)
