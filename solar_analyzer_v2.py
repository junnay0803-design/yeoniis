import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="태양광 발전 분석기 v2",
    page_icon="☀️",
    layout="wide",
)

# ── 커스텀 CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
  .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%); }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1e35 0%, #0a1525 100%);
    border-right: 1px solid #1e3a5f;
  }
  [data-testid="stSidebar"] * { color: #c8deff !important; }

  .metric-card {
    background: linear-gradient(135deg, #0f2040 0%, #0a1830 100%);
    border: 1px solid #1e4a8a;
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,120,255,0.15);
    margin-bottom: 12px;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0078ff, #00d4ff);
  }
  .metric-label { font-size: 0.78rem; letter-spacing: 0.15em; text-transform: uppercase; color: #5a9fd4; margin-bottom: 8px; }
  .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 2.4rem; font-weight: 700; color: #00d4ff; line-height: 1; }
  .metric-unit  { font-size: 0.9rem; color: #5a9fd4; margin-top: 6px; }

  .metric-card.green::before { background: linear-gradient(90deg, #00b450, #00ff88); }
  .metric-card.green .metric-value { color: #00ff88; }
  .metric-card.orange::before { background: linear-gradient(90deg, #ff8800, #ffcc00); }
  .metric-card.orange .metric-value { color: #ffcc00; }

  .info-badge {
    background: rgba(30,64,128,0.4); border: 1px solid #1e4a8a;
    border-radius: 10px; padding: 12px 18px; margin: 5px 0;
    color: #a8c8f0; font-size: 0.86rem;
  }
  .info-badge strong { color: #00d4ff; }

  .section-header {
    font-size: 0.73rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: #4a8abf; margin: 24px 0 10px;
    padding-bottom: 8px; border-bottom: 1px solid #1e3a5f;
  }
  .status-banner { border-radius: 10px; padding: 12px 20px; margin: 14px 0; font-size: 0.9rem; font-weight: 600; }
  .status-good { background: rgba(0,180,80,0.15);  border: 1px solid #00b450; color: #00e060; }
  .status-warn { background: rgba(255,160,0,0.15); border: 1px solid #ffa000; color: #ffb830; }
  .status-bad  { background: rgba(255,60,60,0.15); border: 1px solid #ff3c3c; color: #ff6060; }

  .weather-card {
    background: linear-gradient(135deg, #0a1e3a 0%, #071528 100%);
    border: 1px solid #1e4a8a; border-radius: 16px;
    padding: 20px 24px; margin: 12px 0;
  }
  .weather-title { font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 12px; }

  .mppt-box {
    background: rgba(0,180,80,0.1); border: 1px solid #00b450;
    border-radius: 12px; padding: 16px 20px; margin: 10px 0;
  }
  .mppt-title { color: #00ff88; font-weight: 700; font-size: 0.95rem; margin-bottom: 6px; }
  .mppt-body  { color: #a8c8f0; font-size: 0.85rem; line-height: 1.7; }

  h1 { color: #ffffff !important; font-weight: 900 !important; letter-spacing: -0.02em; }
  h3 { color: #5a9fd4 !important; }
  p, li { color: #a8c8f0; }
  .stTabs [data-baseweb="tab"] { color: #5a9fd4 !important; }
  .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom-color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)


# ── 물리 상수 & 모델 함수 ─────────────────────────────────────────────────────
T_STC     = 25.0
TEMP_COEFF = -0.004
NOCT      = 45.0
G_STC     = 1000.0

def cell_temperature(T_amb, G):
    return T_amb + (NOCT - 20.0) / 800.0 * G

def effective_efficiency(eta_stc, T_cell):
    return eta_stc * (1 + TEMP_COEFF * (T_cell - T_STC))

def power_output(G, A, eta_eff):
    return G * A * eta_eff

def monthly_energy(P_kw, peak_hours=3.5):
    return P_kw * peak_hours * 30

# ── I-V 커브 모델 (단일 다이오드 모델 근사) ──────────────────────────────────
def iv_curve(Isc, Voc, n=1.3, T_cell=25.0):
    """
    단일 다이오드 모델 근사 (안정화 버전):
    I = Isc * (1 - exp((V - Voc) / (n*Vt))) 수식을 사용하여
    지수 함수 오버플로우 및 I0 언더플로우 문제를 해결함.
    """
    k = 1.381e-23   # 볼츠만 상수
    q = 1.602e-19   # 전자 전하
    T_k = max(T_cell + 273.15, 250.0)
    Vt = n * k * T_k / q               

    V = np.linspace(0, Voc, 500)
    
    # 상대 전압 (V - Voc)를 사용하여 exp 인자를 0 이하로 고정 (안전한 계산)
    I = Isc * (1 - np.exp(np.clip((V - Voc) / Vt, -700, 0)))
    I = np.clip(I, 0, None)
    P = V * I
    return V, I, P

def find_mpp(V, I, P):
    idx = np.argmax(P)
    return V[idx], I[idx], P[idx]

# ── OpenWeatherMap API ────────────────────────────────────────────────────────
def fetch_weather(city: str, api_key: str):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            temp       = data["main"]["temp"]
            clouds_pct = data["clouds"]["all"]       # 0~100 %
            weather    = data["weather"][0]["description"]
            city_name  = data["name"]
            country    = data["sys"]["country"]
            humidity   = data["main"]["humidity"]
            wind       = data["wind"]["speed"]
            return {
                "success": True,
                "temp": temp,
                "clouds": clouds_pct,
                "weather": weather,
                "city": f"{city_name}, {country}",
                "humidity": humidity,
                "wind": wind,
            }
        else:
            return {"success": False, "error": f"API 오류 ({resp.status_code}): 도시명 또는 API 키를 확인하세요."}
    except Exception as e:
        return {"success": False, "error": str(e)}

def clouds_to_irradiance(clouds_pct: float) -> float:
    """구름량(%) → 추정 일사량(W/m²) 선형 근사"""
    return 1000.0 * (1 - clouds_pct / 100.0 * 0.75)


# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ 입력 파라미터")
    st.markdown("---")

    st.markdown('<div class="section-header">☀ 태양 환경</div>', unsafe_allow_html=True)
    irradiance = st.slider("일사량 (W/m²)", 100, 1200, 800, 50)
    peak_sun_hours = st.slider("피크 일조 시간 (h/day)", 1.0, 8.0, 3.5, 0.5)

    st.markdown('<div class="section-header">🔲 패널 사양</div>', unsafe_allow_html=True)
    panel_area   = st.number_input("패널 총 면적 (m²)", 1.0, 500.0, 20.0, 1.0)
    eta_stc_pct  = st.slider("STC 효율 (%)", 10.0, 25.0, 18.0, 0.5)
    eta_stc      = eta_stc_pct / 100.0

    st.markdown('<div class="section-header">🌡 온도 환경</div>', unsafe_allow_html=True)
    T_amb = st.slider("외부 기온 (°C)", -10, 50, 25, 1)

    st.markdown('<div class="section-header">📡 I-V 커브 파라미터</div>', unsafe_allow_html=True)
    Isc = st.number_input("단락전류 Isc (A)", 1.0, 20.0, 9.5, 0.1,
                          help="STC 기준 단락전류")
    Voc = st.number_input("개방전압 Voc (V)", 10.0, 100.0, 44.0, 0.5,
                          help="STC 기준 개방전압")
    n_factor = st.slider("이상 인자 n", 1.0, 2.0, 1.3, 0.05,
                         help="단결정 ≈ 1.2~1.4, 다결정 ≈ 1.3~1.5")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.73rem; color:#3a6a9f; line-height:1.8;">
    <b>적용 물리 모델</b><br>
    • 셀 온도: NOCT (IEC 61215)<br>
    • 효율 저하: γ = −0.4 %/°C<br>
    • I-V 커브: 단일 다이오드 근사<br>
    • 날씨: OpenWeatherMap API
    </div>
    """, unsafe_allow_html=True)


# ── 핵심 계산 ─────────────────────────────────────────────────────────────────
T_cell       = cell_temperature(T_amb, irradiance)
eta_eff      = max(effective_efficiency(eta_stc, T_cell), 0.0)
P_w          = power_output(irradiance, panel_area, eta_eff)
P_kw         = P_w / 1000.0
E_month      = monthly_energy(P_kw, peak_sun_hours)
eta_loss_pct = (eta_eff - eta_stc) / eta_stc * 100

# ── 헤더 ─────────────────────────────────────────────────────────────────────
st.markdown("# ☀️ 태양광 발전 시스템 분석기 v2")
st.markdown(
    "<p style='color:#4a8abf;font-size:0.9rem;margin-top:-12px;'>"
    "NOCT 온도 모델 · 단일 다이오드 I-V 커브 · 실시간 날씨 연동</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ── 상단 메트릭 ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">⚡ 실시간 출력</div>'
                f'<div class="metric-value">{P_kw:.2f}</div><div class="metric-unit">kW</div></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card green"><div class="metric-label">🗓 월 발전량</div>'
                f'<div class="metric-value">{E_month:.1f}</div><div class="metric-unit">kWh / 월</div></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card orange"><div class="metric-label">🌡 셀 온도</div>'
                f'<div class="metric-value">{T_cell:.1f}</div><div class="metric-unit">°C (NOCT)</div></div>',
                unsafe_allow_html=True)
with c4:
    lc = "#00e060" if eta_loss_pct >= 0 else "#ff6060"
    st.markdown(f'<div class="metric-card"><div class="metric-label">📉 온도 효율 변화</div>'
                f'<div class="metric-value" style="color:{lc};">{eta_loss_pct:+.1f}</div>'
                f'<div class="metric-unit">% (STC 대비)</div></div>',
                unsafe_allow_html=True)

# 상태 배너
if T_cell < 35:
    bcls, bico, bmsg = "status-good", "✅", f"셀 온도 {T_cell:.1f}°C — 최적 동작 범위"
elif T_cell < 55:
    bcls, bico, bmsg = "status-warn", "⚠️", f"셀 온도 {T_cell:.1f}°C — 출력 저하 주의"
else:
    bcls, bico, bmsg = "status-bad",  "🔴", f"셀 온도 {T_cell:.1f}°C — 고온 심각 저하!"
st.markdown(f'<div class="status-banner {bcls}">{bico} {bmsg}</div>', unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  탭 구성
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📈 효율 분석",
    "⚡ I-V / P-V 커브 & MPPT",
    "🌤 실시간 날씨 연동",
])


# ── TAB 1: 효율 분석 (기존 그래프) ──────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📈 온도 변화에 따른 발전 효율 곡선</div>',
                unsafe_allow_html=True)

    T_range      = np.linspace(-10, 70, 300)
    T_cell_range = T_range + (NOCT - 20) / 800 * irradiance
    eta_range    = np.array([max(effective_efficiency(eta_stc, tc), 0) for tc in T_cell_range])

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=T_range, y=eta_range * 100,
        fill='tozeroy', fillcolor='rgba(0,120,255,0.08)',
        line=dict(color='rgba(0,180,255,0.9)', width=2.5),
        name='보정 효율 (%)',
        hovertemplate='외기: %{x:.1f}°C → 효율: %{y:.2f}%<extra></extra>'
    ))
    fig1.add_hline(y=eta_stc_pct, line_dash="dash", line_color="#ffa500",
                   annotation_text=f"STC {eta_stc_pct:.1f}%",
                   annotation_position="top right",
                   annotation_font_color="#ffa500")
    fig1.add_trace(go.Scatter(
        x=[T_amb], y=[eta_eff * 100], mode='markers',
        marker=dict(size=14, color='#00ffaa', line=dict(color='white', width=2)),
        name='현재 동작점'
    ))
    fig1.add_vrect(x0=50, x1=70, fillcolor="rgba(255,60,60,0.07)", line_width=0,
                   annotation_text="고온 위험", annotation_position="top left",
                   annotation_font_color="#ff6060")
    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,20,40,0.6)',
        font=dict(family="Noto Sans KR", color="#a8c8f0"),
        xaxis=dict(title="외부 기온 (°C)", gridcolor='rgba(30,74,138,0.4)', color='#5a9fd4'),
        yaxis=dict(title="보정 효율 (%)", gridcolor='rgba(30,74,138,0.4)', color='#5a9fd4'),
        legend=dict(bgcolor='rgba(10,20,40,0.8)', bordercolor='#1e4a8a', borderwidth=1),
        hovermode='x unified', margin=dict(l=60, r=40, t=20, b=60), height=360,
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<div class="section-header">🗺 일사량 × 외기온도 → 출력 히트맵 (kW)</div>',
                unsafe_allow_html=True)
    G_ax = np.linspace(100, 1200, 60)
    T_ax = np.linspace(-10, 50, 50)
    Z    = np.zeros((len(T_ax), len(G_ax)))
    for i, t in enumerate(T_ax):
        for j, g in enumerate(G_ax):
            eff  = max(effective_efficiency(eta_stc, cell_temperature(t, g)), 0)
            Z[i,j] = power_output(g, panel_area, eff) / 1000

    fig2 = go.Figure(data=go.Heatmap(
        z=Z, x=G_ax, y=T_ax,
        colorscale=[[0,'#0a1830'],[0.3,'#0044aa'],[0.6,'#0099ff'],[0.85,'#00ddff'],[1,'#ffffff']],
        colorbar=dict(
            title=dict(text='출력 (kW)', font=dict(color='#a8c8f0')),
            tickfont=dict(color='#a8c8f0')
        ),
        hovertemplate='일사량: %{x:.0f} W/m²<br>기온: %{y:.0f}°C<br>출력: %{z:.2f} kW<extra></extra>'
    ))
    fig2.add_trace(go.Scatter(
        x=[irradiance], y=[T_amb], mode='markers',
        marker=dict(size=14, color='#ff4444', symbol='x', line=dict(width=3, color='white')),
        name='현재 조건'
    ))
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,20,40,0.6)',
        font=dict(family="Noto Sans KR", color="#a8c8f0"),
        xaxis=dict(title="일사량 (W/m²)", color='#5a9fd4'),
        yaxis=dict(title="외기 온도 (°C)", color='#5a9fd4'),
        legend=dict(bgcolor='rgba(10,20,40,0.8)', bordercolor='#1e4a8a', borderwidth=1),
        margin=dict(l=60, r=20, t=20, b=60), height=360,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 경제성
    st.markdown('<div class="section-header">💰 경제성 분석</div>', unsafe_allow_html=True)
    elec_price   = 120
    annual_e     = E_month * 12
    annual_rev   = annual_e * elec_price
    install_cost = panel_area * 800_000
    payback      = install_cost / annual_rev if annual_rev > 0 else float('inf')
    e1,e2,e3 = st.columns(3)
    with e1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">📅 연간 발전량</div>'
                    f'<div class="metric-value" style="font-size:2rem;">{annual_e:.0f}</div>'
                    f'<div class="metric-unit">kWh / 년</div></div>', unsafe_allow_html=True)
    with e2:
        st.markdown(f'<div class="metric-card green"><div class="metric-label">💵 연간 절감액</div>'
                    f'<div class="metric-value" style="font-size:2rem;">{annual_rev/10000:.0f}</div>'
                    f'<div class="metric-unit">만원 / 년</div></div>', unsafe_allow_html=True)
    with e3:
        pb_s = f"{payback:.1f}" if payback < 100 else "∞"
        st.markdown(f'<div class="metric-card orange"><div class="metric-label">⏳ 회수 기간</div>'
                    f'<div class="metric-value" style="font-size:2rem;">{pb_s}</div>'
                    f'<div class="metric-unit">년 (120원/kWh 기준)</div></div>', unsafe_allow_html=True)


# ── TAB 2: I-V / P-V 커브 & MPPT ─────────────────────────────────────────────
with tab2:
    st.markdown("### ⚡ I-V 커브 & P-V 커브 시뮬레이션")
    st.markdown("""
    <p style='color:#a8c8f0; font-size:0.88rem;'>
    단일 다이오드 모델 기반. 셀 온도에 따른 Voc 저하와 Isc 미세 증가를 반영합니다.
    </p>
    """, unsafe_allow_html=True)

    # 온도 보정 파라미터
    dT      = T_cell - T_STC
    Isc_t   = Isc * (1 + 0.0005 * dT)
    Voc_t   = Voc * (1 - 0.003  * dT)
    Voc_t   = max(Voc_t, 1.0)

    V, I, P = iv_curve(Isc_t, Voc_t, n=n_factor, T_cell=T_cell)
    Vmpp, Impp, Pmpp = find_mpp(V, I, P)
    FF = Pmpp / (Isc_t * Voc_t)  # 충진율

    # ── 메트릭 ──────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">🔵 Isc (온도 보정)</div>'
                    f'<div class="metric-value" style="font-size:2rem;">{Isc_t:.2f}</div>'
                    f'<div class="metric-unit">A</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card orange"><div class="metric-label">🟠 Voc (온도 보정)</div>'
                    f'<div class="metric-value" style="font-size:2rem;">{Voc_t:.2f}</div>'
                    f'<div class="metric-unit">V</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card green"><div class="metric-label">⭐ MPP 출력</div>'
                    f'<div class="metric-value" style="font-size:2rem;">{Pmpp:.1f}</div>'
                    f'<div class="metric-unit">W  (Vmpp={Vmpp:.1f}V, Impp={Impp:.2f}A)</div></div>',
                    unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">📐 충진율 (FF)</div>'
                    f'<div class="metric-value" style="font-size:2rem;">{FF:.3f}</div>'
                    f'<div class="metric-unit">이상적 FF ≈ 0.80~0.85</div></div>',
                    unsafe_allow_html=True)

    # ── I-V / P-V 이중 그래프 ──────────────────────────────────────────────
    fig_iv = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("I-V 커브 (전류-전압)", "P-V 커브 (전력-전압)"),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )

    fig_iv.add_trace(go.Scatter(
        x=V, y=I, name='I-V 커브',
        line=dict(color='#00aaff', width=2.5),
        hovertemplate='V: %{x:.2f} V<br>I: %{y:.3f} A<extra></extra>'
    ), row=1, col=1)

    fig_iv.add_trace(go.Scatter(
        x=[Vmpp], y=[Impp], name='MPP (최대전력점)',
        mode='markers',
        marker=dict(size=14, color='#ff4444', symbol='star',
                    line=dict(color='white', width=2)),
        hovertemplate=f'MPP: {Vmpp:.2f}V, {Impp:.2f}A<extra></extra>'
    ), row=1, col=1)

    fig_iv.add_trace(go.Scatter(
        x=[0], y=[Isc_t], mode='markers+text',
        marker=dict(size=10, color='#00ffaa'),
        text=[f'Isc={Isc_t:.2f}A'], textposition='middle right',
        textfont=dict(color='#00ffaa', size=11),
        name='Isc', showlegend=True
    ), row=1, col=1)
    fig_iv.add_trace(go.Scatter(
        x=[Voc_t], y=[0], mode='markers+text',
        marker=dict(size=10, color='#ffcc00'),
        text=[f'Voc={Voc_t:.2f}V'], textposition='top left',
        textfont=dict(color='#ffcc00', size=11),
        name='Voc', showlegend=True
    ), row=1, col=1)

    fig_iv.add_shape(
        type="rect", x0=0, y0=0, x1=Vmpp, y1=Impp,
        fillcolor="rgba(255,68,68,0.10)", line=dict(color='rgba(255,68,68,0.4)', width=1, dash='dot'),
        row=1, col=1
    )
    fig_iv.add_shape(
        type="rect", x0=0, y0=0, x1=Voc_t, y1=Isc_t,
        fillcolor="rgba(0,180,255,0.05)", line=dict(color='rgba(0,180,255,0.3)', width=1, dash='dash'),
        row=1, col=1
    )

    fig_iv.add_trace(go.Scatter(
        x=V, y=P, name='P-V 커브',
        line=dict(color='#00ff88', width=2.5),
        fill='tozeroy', fillcolor='rgba(0,255,136,0.06)',
        hovertemplate='V: %{x:.2f} V<br>P: %{y:.2f} W<extra></extra>'
    ), row=2, col=1)

    fig_iv.add_trace(go.Scatter(
        x=[Vmpp], y=[Pmpp], name='Pmax',
        mode='markers',
        marker=dict(size=14, color='#ff4444', symbol='star',
                    line=dict(color='white', width=2)),
        hovertemplate=f'Pmax: {Pmpp:.1f}W @ {Vmpp:.2f}V<extra></extra>'
    ), row=2, col=1)

    fig_iv.add_vline(x=Vmpp, line_dash="dash", line_color="#ff4444",
                     annotation_text=f"Vmpp={Vmpp:.1f}V",
                     annotation_font_color="#ff4444",
                     annotation_position="top right")

    _bg = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,20,40,0.6)')
    _ax = dict(gridcolor='rgba(30,74,138,0.4)', color='#5a9fd4', zerolinecolor='#1e3a5f')

    fig_iv.update_layout(
        **_bg,
        font=dict(family="Noto Sans KR", color="#a8c8f0"),
        legend=dict(bgcolor='rgba(10,20,40,0.8)', bordercolor='#1e4a8a', borderwidth=1,
                    orientation='h', y=-0.08),
        margin=dict(l=60, r=40, t=40, b=60),
        height=620,
    )
    fig_iv.update_xaxes(**_ax, title_text="전압 (V)", row=2, col=1)
    fig_iv.update_xaxes(**_ax, row=1, col=1)
    fig_iv.update_yaxes(**_ax, title_text="전류 (A)", row=1, col=1)
    fig_iv.update_yaxes(**_ax, title_text="전력 (W)", row=2, col=1)

    st.plotly_chart(fig_iv, use_container_width=True)

    # ── MPPT 설명 ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 MPPT가 왜 필요한가?</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="mppt-box">
          <div class="mppt-title">⭐ MPPT란? (Maximum Power Point Tracking)</div>
          <div class="mppt-body">
          태양광 패널은 항상 동일한 전력을 내지 않습니다.<br>
          P-V 커브에서 볼 수 있듯이, <b style="color:#00ff88">전압에 따라 출력 전력이 크게 달라집니다.</b><br><br>
          MPPT 인버터는 실시간으로 Vmpp를 추적하여<br>
          항상 최대 전력점(<b style="color:#ff4444">★ 빨간 별</b>)에서 동작하도록 제어합니다.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        V_fixed = Voc_t * 0.70
        idx_fixed = np.argmin(np.abs(V - V_fixed))
        P_fixed = P[idx_fixed]
        gain_pct = (Pmpp - P_fixed) / P_fixed * 100 if P_fixed > 0 else 0

        st.markdown(f"""
        <div class="mppt-box" style="border-color:#0078ff; background:rgba(0,120,255,0.08);">
          <div class="mppt-title" style="color:#00aaff;">📊 MPPT 효과 비교 (현재 조건)</div>
          <div class="mppt-body">
          고정 전압 동작 (Voc×70% = {V_fixed:.1f}V)<br>
          &nbsp;&nbsp;→ 출력: <b style="color:#ffcc00">{P_fixed:.1f} W</b><br><br>
          MPPT 추적 동작 (Vmpp = {Vmpp:.1f}V)<br>
          &nbsp;&nbsp;→ 출력: <b style="color:#00ff88">{Pmpp:.1f} W</b><br><br>
          <b style="color:#ff4444">MPPT로 약 {gain_pct:.1f}% 추가 발전 가능!</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # 온도별 커브 비교 (원복된 부분)
    st.markdown('<div class="section-header">🌡 온도별 I-V 커브 비교</div>', unsafe_allow_html=True)
    fig_temp = go.Figure()
    temps_compare = [-10, 0, 15, 25, 40, 55, 70]
    colors_t = ['#00ffff','#00aaff','#0066ff','#ffffff','#ffaa00','#ff6600','#ff2200']

    for tc, col in zip(temps_compare, colors_t):
        _dT   = tc - T_STC
        _Isc  = Isc * (1 + 0.0005 * _dT)
        _Voc  = max(Voc * (1 - 0.003 * _dT), 1.0)
        _V, _I, _ = iv_curve(_Isc, _Voc, n=n_factor, T_cell=tc)
        bold = (tc == 25)
        fig_temp.add_trace(go.Scatter(
            x=_V, y=_I, name=f'{tc}°C',
            line=dict(color=col, width=3 if bold else 1.5,
                      dash='solid' if bold else 'solid'),
            opacity=1.0 if bold else 0.7,
            hovertemplate=f'{tc}°C: V=%{{x:.2f}}V I=%{{y:.3f}}A<extra></extra>'
        ))

    fig_temp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,20,40,0.6)',
        font=dict(family="Noto Sans KR", color="#a8c8f0"),
        xaxis=dict(title="전압 (V)", gridcolor='rgba(30,74,138,0.4)', color='#5a9fd4'),
        yaxis=dict(title="전류 (A)", gridcolor='rgba(30,74,138,0.4)', color='#5a9fd4'),
        legend=dict(bgcolor='rgba(10,20,40,0.8)', bordercolor='#1e4a8a', borderwidth=1,
                    orientation='h'),
        annotations=[dict(
            x=0.5, y=1.05, xref='paper', yref='paper',
            text="온도 ↑ → Voc 감소, Isc 미세 증가 → 전체 출력 감소",
            showarrow=False, font=dict(color='#a8c8f0', size=11)
        )],
        margin=dict(l=60, r=40, t=50, b=60), height=380,
    )
    st.plotly_chart(fig_temp, use_container_width=True)


# ── TAB 3: 실시간 날씨 연동 ──────────────────────────────────────────────────
with tab3:
    st.markdown("### 🌤 실시간 날씨 기반 발전량 계산")
    with st.expander("🔑 OpenWeatherMap API 키 설정", expanded=True):
        st.markdown("""
        <div class="info-badge">
        무료 API 키 발급: <strong>https://openweathermap.org/api</strong> → 회원가입 → API Keys 탭<br>
        무료 플랜으로 충분합니다 (1,000 calls/day).
        </div>
        """, unsafe_allow_html=True)
        api_key = st.text_input("API Key", type="password", placeholder="예: a1b2c3d4e5f6...")

    city_input = st.text_input("🏙 도시 이름 (영문)", value="Seoul", placeholder="예: Seoul, Tokyo")

    if st.button("🔍 날씨 조회 및 발전량 계산", type="primary"):
        if not api_key:
            st.error("⚠️ API 키를 먼저 입력해 주세요!")
        else:
            with st.spinner(f"'{city_input}' 날씨 조회 중..."):
                result = fetch_weather(city_input, api_key)

            if not result["success"]:
                st.error(f"❌ {result['error']}")
            else:
                w = result
                st.markdown(f"""
                <div class="weather-card">
                  <div class="weather-title">📍 {w['city']} — 현재 날씨</div>
                  <div style="display:flex; gap:24px; flex-wrap:wrap;">
                    <div class="info-badge" style="flex:1;">🌡 기온: <strong>{w['temp']:.1f}°C</strong></div>
                    <div class="info-badge" style="flex:1;">☁️ 구름량: <strong>{w['clouds']}%</strong></div>
                    <div class="info-badge" style="flex:1;">💧 습도: <strong>{w['humidity']}%</strong></div>
                    <div class="info-badge" style="flex:1;">💨 풍속: <strong>{w['wind']} m/s</strong></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                G_weather   = clouds_to_irradiance(w['clouds'])
                T_cell_w    = cell_temperature(w['temp'], G_weather)
                eta_eff_w   = max(effective_efficiency(eta_stc, T_cell_w), 0.0)
                P_w_weather = power_output(G_weather, panel_area, eta_eff_w)
                P_kw_w      = P_w_weather / 1000.0
                E_month_w   = monthly_energy(P_kw_w, peak_sun_hours)

                st.markdown('<div class="section-header">⚡ 실시간 날씨 기반 발전량</div>', unsafe_allow_html=True)
                wc1, wc2, wc3, wc4 = st.columns(4)
                with wc1:
                    st.markdown(f'<div class="metric-card orange"><div class="metric-label">☀️ 추정 일사량</div>'
                                f'<div class="metric-value" style="font-size:2rem;">{G_weather:.0f}</div>'
                                f'<div class="metric-unit">W/m²</div></div>', unsafe_allow_html=True)
                with wc2:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">🔥 셀 온도</div>'
                                f'<div class="metric-value" style="font-size:2rem;">{T_cell_w:.1f}</div>'
                                f'<div class="metric-unit">°C</div></div>', unsafe_allow_html=True)
                with wc3:
                    st.markdown(f'<div class="metric-card green"><div class="metric-label">⚡ 실시간 출력</div>'
                                f'<div class="metric-value" style="font-size:2rem;">{P_kw_w:.2f}</div>'
                                f'<div class="metric-unit">kW</div></div>', unsafe_allow_html=True)
                with wc4:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">🗓 월 발전량</div>'
                                f'<div class="metric-value" style="font-size:2rem;">{E_month_w:.1f}</div>'
                                f'<div class="metric-unit">kWh / 월</div></div>', unsafe_allow_html=True)

# ── 푸터 ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size:0.73rem; color:#2a5a8f; text-align:center; line-height:1.9;">
태양광 발전 분석기 v2 &nbsp;|&nbsp; NOCT 셀 온도 모델 (IEC 61215) &nbsp;|&nbsp;
단일 다이오드 I-V 모델 &nbsp;|&nbsp; OpenWeatherMap API<br>
설치비·전기요금 기준값은 추정치이며 실제와 다를 수 있습니다.
</div>
""", unsafe_allow_html=True)