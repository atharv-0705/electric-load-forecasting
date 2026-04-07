"""pages/6_Run_History.py"""

import streamlit as st
import pandas as pd
import sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.database import DB

st.set_page_config(page_title="Run History", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"],.stApp{font-family:'JetBrains Mono',monospace!important}
.stApp{background:#0D1117!important;color:#E6EDF3}
section[data-testid="stSidebar"]{background:#0A0F14!important}
</style>
""", unsafe_allow_html=True)

st.markdown("## Run History")
st.caption("Every training run is auto-logged here for reproducibility.")

runs = DB.get_runs()
if runs:
    df = pd.DataFrame(runs)
    order = ["id", "ts", "model", "arima_order", "ann_layers", "epochs", "aic", "train_n", "test_n"]
    show  = [c for c in order if c in df.columns]
    st.dataframe(df[show].sort_values("id", ascending=False), use_container_width=True)
    st.download_button("Download Run History CSV",
                       df.to_csv(index=False).encode(),
                       "run_history.csv", "text/csv")
else:
    st.info("No runs yet. Train a model to populate this log.")
