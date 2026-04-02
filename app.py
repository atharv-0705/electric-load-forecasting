"""
app.py - Electric Load Forecasting Dashboard
Run:  streamlit run app.py
"""

import streamlit as st
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="Load Forecaster",
    page_icon="zap",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@700;800&display=swap');
:root{
  --bg:#0D1117; --surface:#161B22; --b1:rgba(0,212,255,.18);
  --cyan:#00D4FF; --green:#7CFC00; --orange:#FF6B35;
  --text:#E6EDF3; --muted:#8B949E;
}
html,body,[class*="css"],.stApp{ font-family:'JetBrains Mono',monospace !important; }
.stApp { background:var(--bg)!important; color:var(--text); }
section[data-testid="stSidebar"]{ background:#0A0F14!important; border-right:1px solid var(--b1); }
.stButton>button{ background:linear-gradient(135deg,#00D4FF18,#7CFC0018);
  border:1px solid var(--cyan); color:var(--cyan); border-radius:8px; transition:all .2s; }
.stButton>button:hover{ background:linear-gradient(135deg,#00D4FF35,#7CFC0035); }
.stButton>button[kind="primary"]{ background:linear-gradient(135deg,#00D4FF,#7CFC00)!important;
  color:#0D1117!important; border:none!important; font-weight:700; }
[data-testid="stMetricValue"]{ color:var(--cyan)!important; }
.stProgress>div>div{ background:linear-gradient(90deg,var(--cyan),var(--green))!important; }
.stTabs [data-baseweb="tab"]{ color:var(--muted)!important; }
.stTabs [data-baseweb="tab"][aria-selected="true"]{ color:var(--cyan)!important; border-bottom:2px solid var(--cyan)!important; }
hr{ border-color:var(--b1)!important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0D1117 0%,#161B22 60%,#0D1117 100%);
  border:1px solid rgba(0,212,255,.2);border-radius:16px;padding:2.8rem 3rem;
  margin-bottom:2rem;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-50px;right:-50px;width:220px;height:220px;
    background:radial-gradient(circle,rgba(0,212,255,.1),transparent 70%);
    border-radius:50%;pointer-events:none"></div>
  <div style="font-size:.72rem;color:#8B949E;letter-spacing:3px;text-transform:uppercase;margin-bottom:.6rem">
    SHORT-TERM ELECTRIC POWER LOAD FORECASTING SYSTEM
  </div>
  <h1 style="font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;
    background:linear-gradient(90deg,#00D4FF,#7CFC00);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .6rem">
    ARIMA + ANN Hybrid Model
  </h1>
  <p style="color:#8B949E;font-size:.88rem;max-width:640px;margin:0;line-height:1.7">
    Production-grade forecasting combining classical time-series analysis (ARIMA)
    with deep learning (ANN) to deliver superior short-term load predictions,
    reduce MAPE, and support real-time grid scheduling decisions.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Navigation cards ──────────────────────────────────────────────────────
st.markdown("### Workflow")

cards = [
    ("#00D4FF","rgba(0,212,255,.15)","01","Data Upload",
     "Upload CSV/Excel or use the built-in 1-year sample. Explore load patterns with interactive charts."),
    ("#7CFC00","rgba(124,252,0,.12)","02","Model Training",
     "Configure ARIMA order (or auto-select via AIC), set ANN layers, and train the hybrid model."),
    ("#FF6B35","rgba(255,107,53,.12)","03","Forecasting",
     "Generate short-term forecasts (24-168 h), evaluate on test data, download results."),
    ("#FF69B4","rgba(255,105,180,.12)","04","Model Comparison",
     "Compare ARIMA vs Hybrid on MAPE, RMSE, R2, and view error distributions."),
    ("#C0A0FF","rgba(192,160,255,.12)","05","ARIMA Analysis",
     "Stationarity tests, ACF/PACF, seasonal decomposition, model summary."),
    ("#FFD700","rgba(255,215,0,.10)","06","Run History",
     "Every experiment is logged automatically for reproducibility."),
]

cols = st.columns(3)
for i, (color, border, num, title, desc) in enumerate(cards):
    with cols[i % 3]:
        st.markdown(f"""
<div style="background:#161B22;border:1px solid {border};border-radius:10px;
            padding:1.3rem;margin-bottom:.8rem;min-height:140px">
  <div style="font-size:.65rem;color:{color};letter-spacing:2px;text-transform:uppercase;margin-bottom:.3rem">STEP {num}</div>
  <div style="font-weight:600;margin-bottom:.4rem;font-size:.95rem">{title}</div>
  <div style="color:#8B949E;font-size:.78rem;line-height:1.5">{desc}</div>
</div>""", unsafe_allow_html=True)

st.markdown("---")

with st.expander("System Architecture"):
    st.code("""
  Data Input (CSV / Excel)
       |
       v
  +----------------------------------------------------------+
  |  Preprocessor  (clean -> outlier removal -> features)    |
  +----------------------------------------------------------+
       |
       +-------------------+-------------------+
       v                                       v
  +-----------------+             +----------------------+
  |   ARIMA Model   | -residuals->|      ANN Model       |
  |  (linear part)  |             |   (nonlinear part)   |
  +-----------------+             +----------------------+
       |  ARIMA prediction              | ANN correction
       +------------------+-------------+
                          v
            Hybrid Forecast = ARIMA + ANN
                          |
                          v
          +----------------------------------+
          |       Streamlit Dashboard        |
          |  (charts, metrics, CSV export)   |
          +----------------------------------+
    """, language="text")

with st.expander("Technology Stack"):
    st.markdown("""
| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit + Plotly | Dashboard UI |
| **Time Series** | Statsmodels ARIMA | Linear component |
| **Deep Learning** | scikit-learn MLP | Residual correction |
| **Data** | Pandas + NumPy | Preprocessing and features |
| **Scaling** | scikit-learn MinMaxScaler | Feature normalisation |
| **Persistence** | JSON + joblib | Run logging and model saving |
| **Language** | Python 3.11+ | Core runtime |
    """)

st.markdown("---")
st.markdown("### Session Status")
c1, c2, c3, c4 = st.columns(4)
flags = [
    ("Data Loaded",    st.session_state.get("data_ready",  False)),
    ("Model Trained",  st.session_state.get("model_ready", False)),
    ("Forecast Ready", "test_results" in st.session_state),
    ("Dataset Rows",   len(st.session_state.get("df_feat", [])) if "df_feat" in st.session_state else 0),
]
for col, (lbl, val) in zip([c1, c2, c3, c4], flags):
    with col:
        if isinstance(val, bool):
            st.markdown(f"**{lbl}:** {'Yes' if val else 'No'}")
        else:
            st.markdown(f"**{lbl}:** {val:,}")
