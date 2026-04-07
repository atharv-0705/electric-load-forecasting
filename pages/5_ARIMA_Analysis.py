"""pages/5_ARIMA_Analysis.py"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.visualizer import chart_acf_pacf, chart_decomposition, C, _layout

st.set_page_config(page_title="ARIMA Analysis", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"],.stApp{font-family:'JetBrains Mono',monospace!important}
.stApp{background:#0D1117!important;color:#E6EDF3}
section[data-testid="stSidebar"]{background:#0A0F14!important}
</style>
""", unsafe_allow_html=True)

st.markdown("## ARIMA Diagnostic Analysis")

if not st.session_state.get("model_ready"):
    st.warning("Train a model first on the Model Training page.")
    st.stop()

model   = st.session_state["model"]
df_feat = st.session_state["df_feat"]
arima   = model.arima
ts      = model.prep.prepare_arima_series(df_feat)

t1, t2, t3, t4 = st.tabs(["Stationarity", "ACF / PACF", "Decomposition", "Model Summary"])

with t1:
    st.markdown("### Augmented Dickey-Fuller Test")
    r = arima.check_stationarity(ts)
    c1, c2, c3 = st.columns(3)
    c1.metric("ADF Statistic", r["adf"])
    c2.metric("p-value",       r["p_value"])
    c3.metric("Result", "Stationary" if r["is_stationary"] else "Non-stationary")

    st.markdown("**Critical Values:**")
    st.dataframe(pd.DataFrame.from_dict(r["critical_values"], orient="index",
                                        columns=["Value"]), use_container_width=False)
    if not r["is_stationary"]:
        st.info("Series is non-stationary. ARIMA uses differencing (d >= 1).")

    fig = make_subplots(2, 1, subplot_titles=["Original Series", "1st Difference"])
    fig.add_trace(go.Scatter(y=ts.values[:720], mode="lines",
                             line=dict(color=C["actual"], width=1), name="Original"), 1, 1)
    fig.add_trace(go.Scatter(y=ts.diff().dropna().values[:720], mode="lines",
                             line=dict(color=C["hybrid"], width=1), name="Differenced"), 2, 1)
    fig.update_layout(**_layout(height=380, title="Stationarity Check (first 30 days)"))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    nlags = st.slider("Lags", 20, 72, 48)
    try:
        acf_v, pacf_v = arima.get_acf_pacf(ts, nlags=nlags)
        conf = 1.96 / np.sqrt(len(ts))
        st.plotly_chart(chart_acf_pacf(acf_v, pacf_v, conf), use_container_width=True)
        st.info(f"Selected ARIMA order: {model.arima_order}")
    except Exception as e:
        st.error(f"Could not compute ACF/PACF: {e}")

with t3:
    weeks = st.slider("Analysis window (weeks)", 1, 8, 2)
    try:
        decomp = arima.decompose(ts.iloc[:24*7*weeks], period=24)
        if decomp:
            st.plotly_chart(chart_decomposition(decomp), use_container_width=True)
        else:
            st.warning("Decomposition failed — need more data or a longer period.")
    except Exception as e:
        st.error(str(e))

with t4:
    st.code(arima.summary_text(), language="text")
