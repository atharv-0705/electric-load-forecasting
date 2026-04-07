"""pages/3_Forecasting.py"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta # datetime manipulation for future forecasting
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.visualizer import chart_forecast, chart_future_forecast, chart_scatter
from utils.metrics    import compute_metrics
from utils.database   import DB

st.set_page_config(page_title="Forecasting", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"],.stApp{font-family:'JetBrains Mono',monospace!important}
.stApp{background:#0D1117!important;color:#E6EDF3}
section[data-testid="stSidebar"]{background:#0A0F14!important}
.kpi{background:#161B22;border:1px solid rgba(124,252,0,.25);border-radius:10px;padding:1.1rem;text-align:center}
.kv{font-size:1.8rem;font-weight:700;color:#7CFC00}
.kl{font-size:.7rem;color:#8B949E;text-transform:uppercase;letter-spacing:1px}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#00D4FF,#7CFC00)!important;
  color:#0D1117!important;border:none!important;font-weight:700;}
</style>
""", unsafe_allow_html=True)

st.markdown("## Forecasting")

if not st.session_state.get("model_ready"):
    st.warning("No trained model. Go to Model Training first.")
    st.stop()

model    = st.session_state["model"]
df_train = st.session_state["df_train"]
df_test  = st.session_state["df_test"]

st.sidebar.markdown("### Forecast Settings")
horizon    = st.sidebar.selectbox("Horizon", [24, 48, 72, 168],
                                   format_func=lambda h: f"{h}h ({h//24}d)")
show_ci    = st.sidebar.checkbox("Confidence interval", True)
show_arima = st.sidebar.checkbox("ARIMA baseline",      True)

tab1, tab2, tab3 = st.tabs(["Test Evaluation", "Future Forecast", "Export"])

# ── Test Evaluation ───────────────────────────────────────────────────────
with tab1:
    st.markdown("### Held-out test set evaluation")
    if len(df_test) == 0:
        st.info("No test data. Retrain with test split > 0%.")
        st.stop()

    with st.spinner("Running predictions..."):
        try:
            hybrid_pred = model.predict(df_test)
            actual      = df_test["load_mw"].values
            n           = min(len(actual), len(hybrid_pred))
            actual, hybrid_pred = actual[:n], hybrid_pred[:n]

            arima_fc, lower, upper = model.arima.forecast(steps=n)
            arima_fc = arima_fc[:n]

            hm = compute_metrics(actual, hybrid_pred)
            am = compute_metrics(actual, arima_fc)

            kpis = [("Hybrid MAPE",  f"{hm['MAPE']:.2f}%"),
                    ("ARIMA MAPE",   f"{am['MAPE']:.2f}%"),
                    ("RMSE",         f"{hm['RMSE']:.2f} MW"),
                    ("R2",           f"{hm['R2']:.4f}"),
                    ("MAE",          f"{hm['MAE']:.2f} MW")]
            cols = st.columns(5)
            for col, (lbl, val) in zip(cols, kpis):
                with col:
                    st.markdown(f'<div class="kpi"><div class="kv">{val}</div>'
                                f'<div class="kl">{lbl}</div></div>',
                                unsafe_allow_html=True)

            dates = df_test["datetime"].values[:n] if "datetime" in df_test.columns \
                    else np.arange(n)

            st.plotly_chart(chart_forecast(
                dates, actual,
                arima_fc if show_arima else None,
                hybrid_pred,
                lower[:n] if show_ci else None,
                upper[:n] if show_ci else None,
            ), use_container_width=True)

            cL, cR = st.columns(2)
            with cL:
                st.plotly_chart(chart_scatter(actual, hybrid_pred, "Hybrid"),
                                use_container_width=True)
            with cR:
                st.plotly_chart(chart_scatter(actual, arima_fc, "ARIMA"),
                                use_container_width=True)

            st.session_state["test_results"] = dict(
                dates=dates, actual=actual,
                hybrid_pred=hybrid_pred, arima_pred=arima_fc,
                hm=hm, am=am)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ── Future Forecast ───────────────────────────────────────────────────────
with tab2:
    st.markdown(f"### Future {horizon}-hour forecast")
    if st.button("Generate Future Forecast", type="primary"):
        with st.spinner(f"Forecasting {horizon} hours ahead..."):
            try:
                res = model.forecast(steps=horizon, last_df=df_train)
                last_dt = pd.to_datetime(df_train["datetime"].max()) \
                          if "datetime" in df_train.columns else pd.Timestamp.now()
                future_dates = [last_dt + timedelta(hours=i+1) for i in range(horizon)]

                st.plotly_chart(chart_future_forecast(
                    future_dates, res["arima_fc"], res["hybrid_fc"],
                    res["lower"], res["upper"]), use_container_width=True)

                fc_df = pd.DataFrame({
                    "DateTime":           future_dates,
                    "ARIMA_Forecast_MW":  np.round(res["arima_fc"],  2),
                    "ANN_Correction_MW":  np.round(res["ann_corr"],  2),
                    "Hybrid_Forecast_MW": np.round(res["hybrid_fc"], 2),
                    "Lower_CI_MW":        np.round(res["lower"],     2),
                    "Upper_CI_MW":        np.round(res["upper"],     2),
                })
                st.dataframe(fc_df, use_container_width=True)

                cA, cB, cC = st.columns(3)
                cA.metric("Peak",  f"{res['hybrid_fc'].max():.1f} MW")
                cB.metric("Min",   f"{res['hybrid_fc'].min():.1f} MW")
                cC.metric("Mean",  f"{res['hybrid_fc'].mean():.1f} MW")

                st.session_state["future_df"] = fc_df

            except Exception as e:
                st.error(f"Forecast error: {e}")

# ── Export ────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Export Results")
    if "future_df" in st.session_state:
        fdf = st.session_state["future_df"]
        st.download_button("Download Future Forecast CSV",
                           fdf.to_csv(index=False).encode(),
                           "hybrid_forecast.csv", "text/csv")
        st.dataframe(fdf.head(24), use_container_width=True)

    if "test_results" in st.session_state:
        r = st.session_state["test_results"]
        rdf = pd.DataFrame({"datetime": r["dates"], "actual_mw": r["actual"],
                             "hybrid_mw": r["hybrid_pred"], "arima_mw": r["arima_pred"]})
        st.download_button("Download Test Evaluation CSV",
                           rdf.to_csv(index=False).encode(),
                           "test_evaluation.csv", "text/csv")
    else:
        st.info("Run the Test Evaluation tab first to enable download.")
