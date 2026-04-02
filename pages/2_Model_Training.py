"""pages/2_Model_Training.py"""

import streamlit as st
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.hybrid_model import HybridModel
from utils.visualizer    import chart_residuals, chart_training_loss
from utils.database      import DB

st.set_page_config(page_title="Model Training", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"],.stApp{font-family:'JetBrains Mono',monospace!important}
.stApp{background:#0D1117!important;color:#E6EDF3}
section[data-testid="stSidebar"]{background:#0A0F14!important}
.info-box{background:#161B22;border-left:3px solid #00D4FF;
          padding:.6rem 1rem;border-radius:0 6px 6px 0;margin:.4rem 0}
.res-card{background:#161B22;border-radius:8px;padding:1rem;text-align:center}
.rv{font-size:1.5rem;font-weight:700}
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,#00D4FF,#7CFC00)!important;
  color:#0D1117!important;border:none!important;font-weight:700;}
.speed-box{background:rgba(124,252,0,0.08);border:1px solid rgba(124,252,0,0.3);
           border-radius:8px;padding:.8rem 1rem;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)

st.markdown("## Model Training")

if not st.session_state.get("data_ready"):
    st.warning("Please upload data on the Data Upload page first.")
    st.stop()

df_feat = st.session_state["df_feat"]

# ── Speed info banner ─────────────────────────────────────────────────────
st.markdown("""
<div class="speed-box">
  <b style="color:#7CFC00">Fast Training Mode</b> — 
  ARIMA uses 8 curated candidates (not full grid) &nbsp;|&nbsp;
  ANN uses early stopping &nbsp;|&nbsp;
  Expected time: <b style="color:#00D4FF">30 – 90 seconds</b>
</div>
""", unsafe_allow_html=True)

with st.expander("Hybrid Architecture", expanded=False):
    st.code("""
 Raw Load Data
      |
      v
 Preprocessor  (clean + feature engineering)
      |
      +--------------------+--------------------+
      v                                         v
 ARIMA Model                            Feature Matrix
 (fits linear autocorrelation)               |
      |                                        v
      | residuals                        ANN Model
      |                            (learns nonlinear residuals)
      |                                        |
      +--------------------+-------------------+
                           v
         Hybrid Forecast = ARIMA + ANN correction
    """, language="text")

# ── Sidebar config ────────────────────────────────────────────────────────
st.sidebar.markdown("### Configuration")
st.sidebar.markdown("**ARIMA**")

auto_arima = st.sidebar.checkbox("Auto-select order (fast AIC)", value=True)
p = st.sidebar.slider("p - AR order",      0, 5, 2, disabled=auto_arima)
d = st.sidebar.slider("d - Differencing",  0, 2, 1, disabled=auto_arima)
q = st.sidebar.slider("q - MA order",      0, 5, 2, disabled=auto_arima)

st.sidebar.markdown("**ANN**")
epochs = st.sidebar.slider("Max Epochs", 20, 200, 100,
                            help="Early stopping will stop sooner if converged")
h1 = st.sidebar.slider("Layer 1", 16, 128, 64)
h2 = st.sidebar.slider("Layer 2",  8,  64, 32)
h3 = st.sidebar.slider("Layer 3",  4,  32, 16)
layers = (h1, h2, h3)

test_pct = st.sidebar.slider("Test split %", 10, 40, 20)

# ── Dataset info ──────────────────────────────────────────────────────────
n_train = int(len(df_feat) * (1 - test_pct / 100))
n_test  = len(df_feat) - n_train
c1, c2, c3 = st.columns(3)
c1.metric("Total records",    f"{len(df_feat):,}")
c2.metric("Training records", f"{n_train:,}")
c3.metric("Test records",     f"{n_test:,}")

# ── Train button ──────────────────────────────────────────────────────────
if st.button("Start Training", type="primary", use_container_width=True):
    df_train = df_feat.iloc[:n_train].copy()
    df_test  = df_feat.iloc[n_train:].copy()
    st.session_state["df_train"] = df_train
    st.session_state["df_test"]  = df_test

    bar    = st.progress(0, "Initialising...")
    status = st.empty()
    timer  = st.empty()

    import time
    t_start = time.time()

    def cb(frac, msg):
        elapsed = time.time() - t_start
        bar.progress(frac, msg)
        status.markdown(f'<div class="info-box">{msg}</div>',
                        unsafe_allow_html=True)
        timer.caption(f"Elapsed: {elapsed:.1f}s")

    try:
        model = HybridModel(
            arima_order=(p, d, q),
            ann_layers=layers,
            epochs=epochs,
            auto_arima=auto_arima,
        )
        info = model.fit(df_train, progress=cb)

        elapsed = time.time() - t_start
        st.session_state["model"]       = model
        st.session_state["train_info"]  = info
        st.session_state["model_ready"] = True

        bar.progress(1.0, "Training complete!")
        timer.empty()
        st.success(f"Model trained in {elapsed:.1f} seconds!")

        # Result cards
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f'<div class="res-card">'
                f'<div class="rv" style="color:#00D4FF">{model.arima_order}</div>'
                f'<div style="color:#8B949E;font-size:.8rem">ARIMA Order (p,d,q)</div>'
                f'</div>', unsafe_allow_html=True)
        with r2:
            aic = info.get("arima", {}).get("aic", "N/A")
            st.markdown(
                f'<div class="res-card">'
                f'<div class="rv" style="color:#7CFC00">{aic}</div>'
                f'<div style="color:#8B949E;font-size:.8rem">AIC Score</div>'
                f'</div>', unsafe_allow_html=True)
        with r3:
            bk = info.get("ann", {}).get("backend", "N/A")
            ep = info.get("ann", {}).get("epochs_run", "N/A")
            st.markdown(
                f'<div class="res-card">'
                f'<div class="rv" style="color:#FF6B35">{ep} epochs</div>'
                f'<div style="color:#8B949E;font-size:.8rem">ANN ({bk})</div>'
                f'</div>', unsafe_allow_html=True)

        # Charts
        cl, cr = st.columns(2)
        with cl:
            res = model.arima.residuals_
            if res is not None and len(res) > 0:
                st.plotly_chart(chart_residuals(res), use_container_width=True)
        with cr:
            hist = model.ann.history_
            if hist.get("loss"):
                st.plotly_chart(chart_training_loss(hist), use_container_width=True)

        DB.log_run({
            "model":       "Hybrid",
            "arima_order": str(model.arima_order),
            "ann_layers":  str(layers),
            "epochs":      epochs,
            "aic":         info.get("arima", {}).get("aic"),
            "train_n":     n_train,
            "test_n":      n_test,
            "time_sec":    round(elapsed, 1),
        })

    except Exception as e:
        bar.progress(0, "Failed")
        st.error(f"Training error: {e}")
        import traceback
        with st.expander("Stack trace"):
            st.code(traceback.format_exc())

elif st.session_state.get("model_ready"):
    st.info("Model already trained. Go to Forecasting to generate predictions.")
    st.json(st.session_state.get("train_info", {}))
