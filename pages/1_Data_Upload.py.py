"""pages/1_Data_Upload.py"""

import streamlit as st
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.preprocessor import DataPreprocessor
from utils.visualizer   import chart_overview, chart_daily, chart_heatmap, chart_monthly_box
from data.generate_sample_data import generate_power_load_data

st.set_page_config(page_title="Data Upload", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"],.stApp{font-family:'JetBrains Mono',monospace!important}
.stApp{background:#0D1117!important;color:#E6EDF3}
section[data-testid="stSidebar"]{background:#0A0F14!important}
.card{background:#161B22;border:1px solid rgba(0,212,255,0.2);border-radius:10px;padding:1.1rem;text-align:center}
.cv{font-size:1.75rem;font-weight:700;color:#00D4FF}
.cl{font-size:.7rem;color:#8B949E;text-transform:uppercase;letter-spacing:1px;margin-top:.2rem}
</style>
""", unsafe_allow_html=True)

st.markdown("## Data Upload & Exploration")
st.caption("Upload your power load CSV/Excel, or use the built-in 1-year sample dataset.")

prep = DataPreprocessor()

c1, c2 = st.columns([2.5, 1])
with c1:
    uploaded = st.file_uploader("Upload dataset (CSV / Excel)", type=["csv","xlsx"],
                                help="Required columns: datetime, load_mw")
with c2:
    st.markdown("""
**Required columns:**
- `datetime` — hourly timestamps
- `load_mw` — power load in MW

Optional: `temperature_c`, `humidity_pct`
    """)

use_sample = st.checkbox("Use built-in sample dataset (365 days, 8760 hourly records)", value=False)

df_raw = None
if uploaded:
    try:
        import pandas as pd
        df_raw = pd.read_excel(uploaded) if uploaded.name.endswith(("xlsx","xls")) \
                 else pd.read_csv(uploaded)
        st.success(f"Loaded {uploaded.name} — {len(df_raw):,} rows")
    except Exception as e:
        st.error(f"Read error: {e}")
elif use_sample:
    with st.spinner("Generating sample dataset..."):
        df_raw = generate_power_load_data(n_days=365)
    st.success("Sample dataset ready — 8760 hourly records (2023-01-01 to 2023-12-31)")

if df_raw is None:
    st.info("Upload a file or enable the sample dataset to begin.")
    st.stop()

ok, msg = prep.validate_data(df_raw)
if not ok:
    st.error(f"Validation failed: {msg}"); st.stop()

df_clean = prep.clean_data(df_raw)
df_feat  = prep.engineer_features(df_clean)
stats    = prep.summary_stats(df_feat)

st.session_state.update(df_raw=df_raw, df_clean=df_clean, df_feat=df_feat, data_ready=True)

st.markdown("---")
kpis = [("Records",   f"{stats['records']:,}"),
        ("Mean Load",  f"{stats['mean_load']} MW"),
        ("Peak Load",  f"{stats['max_load']} MW"),
        ("Min Load",   f"{stats['min_load']} MW"),
        ("Peak Hour",  f"{stats['peak_hour']}:00")]

cols = st.columns(5)
for col, (lbl, val) in zip(cols, kpis):
    with col:
        st.markdown(f'<div class="card"><div class="cv">{val}</div>'
                    f'<div class="cl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown(f"**Date range:** {stats['date_range']}")
st.markdown("---")

st.plotly_chart(chart_overview(df_feat), use_container_width=True)

r1c1, r1c2 = st.columns(2)
with r1c1: st.plotly_chart(chart_daily(df_feat),       use_container_width=True)
with r1c2: st.plotly_chart(chart_monthly_box(df_feat), use_container_width=True)

st.plotly_chart(chart_heatmap(df_feat), use_container_width=True)

with st.expander("Preview processed data"):
    st.dataframe(df_feat.head(200), use_container_width=True)
    st.caption(f"Shape: {df_feat.shape[0]:,} x {df_feat.shape[1]}")

st.success("Data ready — go to Model Training in the sidebar.")
