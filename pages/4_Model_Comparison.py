"""pages/4_Model_Comparison.py"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.metrics    import metrics_table, pct_improvement
from utils.visualizer import chart_metrics_bar, chart_error_dist, C, _layout

st.set_page_config(page_title="Model Comparison", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap'); # monospace font for code-like look
html,body,[class*="css"],.stApp{font-family:'JetBrains Mono',monospace!important} # apply to all Streamlit elements for consistent look
.stApp{background:#0D1117!important;color:#E6EDF3}                                # dark background with light text for contrast
section[data-testid="stSidebar"]{background:#0A0F14!important}                    # darker sidebar for separation
.imp{background:rgba(124,252,0,.08);border:1px solid #7CFC0044;border-radius:10px; # light green highlight for improvement metrics
     padding:1.4rem;text-align:center}                 
.iv{font-size:2.2rem;font-weight:700;color:#7CFC00}                                 # large bright green for improvement values
.il{font-size:.72rem;color:#8B949E;text-transform:uppercase;letter-spacing:1px}    # smaller, muted text for labels
</style>
""", unsafe_allow_html=True)                                                      # custom CSS for styling the page

st.markdown("## Model Comparison & Accuracy Metrics")

if "test_results" not in st.session_state:
    st.warning("Complete Forecasting > Test Evaluation first.")
    st.stop()

r  = st.session_state["test_results"]
hm = r["hm"]
am = r["am"]
actual      = r["actual"]
hybrid_pred = r["hybrid_pred"]
arima_pred  = r["arima_pred"]

mape_imp = pct_improvement(am["MAPE"], hm["MAPE"])
rmse_imp = pct_improvement(am["RMSE"], hm["RMSE"])

c1, c2, c3 = st.columns(3)
for col, (val, lbl) in zip([c1, c2, c3], [
    (f"{mape_imp:+.1f}%", "MAPE Improvement (Hybrid vs ARIMA)"),
    (f"{rmse_imp:+.1f}%", "RMSE Improvement"),
    (f"{hm['R2']:.4f}",   "Hybrid R2 Score"),
]):
    with col:
        st.markdown(f'<div class="imp"><div class="iv">{val}</div>'
                    f'<div class="il">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Detailed Metrics")
mdict = {"ARIMA": am, "Hybrid (ARIMA+ANN)": hm}
df_m  = metrics_table(mdict)
st.dataframe(df_m.style.format({
    "MAPE": "{:.4f}%", "RMSE": "{:.4f}", "MAE": "{:.4f}",
    "R2": "{:.4f}", "SMAPE": "{:.4f}%", "NRMSE": "{:.4f}%",
}), use_container_width=True)

st.plotly_chart(chart_metrics_bar(mdict), use_container_width=True)
st.plotly_chart(chart_error_dist(actual, hybrid_pred, arima_pred), use_container_width=True)

st.markdown("### Rolling 24-Hour MAPE")
W  = 24
nz = actual != 0
h_mape = np.abs((actual - hybrid_pred) / np.where(nz, actual, 1)) * 100
a_mape = np.abs((actual - arima_pred[:len(actual)]) / np.where(nz, actual, 1)) * 100

fig = go.Figure()
fig.add_trace(go.Scatter(y=pd.Series(a_mape).rolling(W).mean(), name="ARIMA",
                         line=dict(color=C["arima"], width=1.8, dash="dot")))
fig.add_trace(go.Scatter(y=pd.Series(h_mape).rolling(W).mean(), name="Hybrid",
                         line=dict(color=C["hybrid"], width=2)))
fig.update_layout(**_layout(title="Rolling 24h MAPE", xaxis_title="Time Step",
                             yaxis_title="MAPE (%)", height=300))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Metric Definitions"):
    st.markdown("""
| Metric | What it measures | Target |
|--------|-----------------|--------|
| **MAPE** | Mean Absolute Percentage Error | < 5% |
| **RMSE** | Root Mean Squared Error | Minimise |
| **MAE**  | Mean Absolute Error | Minimise |
| **R2**   | Variance explained by the model | > 0.95 |
| **SMAPE**| Symmetric MAPE | < 5% |
| **NRMSE**| RMSE as % of data range | < 10% |
    """)
