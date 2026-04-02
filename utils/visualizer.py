"""
utils/visualizer.py
Plotly chart factory — dark industrial theme throughout.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Theme constants ────────────────────────────────────────────────────────
C = dict(
    actual="#00D4FF", arima="#FF6B35", hybrid="#7CFC00",
    ann="#FF69B4", band="rgba(124,252,0,0.12)",
    bg="#0D1117", surface="#161B22",
    grid="rgba(255,255,255,0.06)", text="#E6EDF3", muted="#8B949E",
)

_BASE = dict(
    plot_bgcolor=C["bg"], paper_bgcolor=C["surface"],
    font=dict(family="JetBrains Mono, monospace", color=C["text"], size=11),
    xaxis=dict(gridcolor=C["grid"], linecolor=C["grid"]),
    yaxis=dict(gridcolor=C["grid"], linecolor=C["grid"]),
    legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    margin=dict(l=50, r=20, t=45, b=45),
    hovermode="x unified",
)

def _layout(**kw):
    d = dict(_BASE); d.update(kw); return d

# ── Charts ────────────────────────────────────────────────────────────────

def chart_overview(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["load_mw"],
        name="Load (MW)", mode="lines",
        line=dict(color=C["actual"], width=1.1),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.04)"))
    fig.update_layout(**_layout(title="⚡ Power Load Time Series",
        xaxis_title="DateTime", yaxis_title="Load (MW)", height=340))
    return fig

def chart_daily(df: pd.DataFrame) -> go.Figure:
    h = df.groupby("hour")["load_mw"].agg(["mean","std"]).reset_index()
    fig = go.Figure([
        go.Scatter(x=h["hour"], y=h["mean"]+h["std"],
                   fill=None, mode="lines", line_color="rgba(0,212,255,0)", showlegend=False),
        go.Scatter(x=h["hour"], y=h["mean"]-h["std"],
                   fill="tonexty", mode="lines", fillcolor="rgba(0,212,255,0.1)",
                   line_color="rgba(0,212,255,0)", name="±1σ"),
        go.Scatter(x=h["hour"], y=h["mean"], mode="lines+markers",
                   name="Mean Load", line=dict(color=C["actual"], width=2.5),
                   marker=dict(size=5)),
    ])
    fig.update_layout(**_layout(title="📊 Average Daily Load Profile",
        xaxis_title="Hour of Day", yaxis_title="Load (MW)", height=310))
    return fig

def chart_heatmap(df: pd.DataFrame) -> go.Figure:
    pv = df.pivot_table(values="load_mw", index="hour", columns="day_of_week", aggfunc="mean")
    pv.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns.tolist(), y=pv.index.tolist(),
        colorscale="Viridis",
        hovertemplate="Day: %{x}<br>Hour: %{y}<br>Load: %{z:.1f} MW<extra></extra>"))
    fig.update_layout(**_layout(title="🗓️ Load Heatmap (Hour × Day)",
        xaxis_title="Day", yaxis_title="Hour", height=320))
    return fig

def chart_monthly_box(df: pd.DataFrame) -> go.Figure:
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure([go.Box(y=df[df["month"]==m]["load_mw"], name=months[m-1],
                            marker_color=C["actual"], showlegend=False) for m in range(1,13)])
    fig.update_layout(**_layout(title="📅 Monthly Load Distribution",
        xaxis_title="Month", yaxis_title="Load (MW)", height=310))
    return fig

def chart_forecast(dates, actual, arima_pred, hybrid_pred,
                   lower=None, upper=None) -> go.Figure:
    fig = go.Figure()
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=list(dates)+list(dates)[::-1],
            y=list(upper)+list(lower)[::-1],
            fill="toself", fillcolor=C["band"],
            line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual",
                             line=dict(color=C["actual"], width=2)))
    if arima_pred is not None:
        fig.add_trace(go.Scatter(x=dates, y=arima_pred, name="ARIMA",
                                 line=dict(color=C["arima"], width=1.8, dash="dot")))
    fig.add_trace(go.Scatter(x=dates, y=hybrid_pred, name="Hybrid (ARIMA+ANN)",
                             line=dict(color=C["hybrid"], width=2.2)))
    fig.update_layout(**_layout(title="🔮 Forecast vs Actual",
        xaxis_title="DateTime", yaxis_title="Load (MW)", height=390))
    return fig

def chart_future_forecast(future_dates, arima_fc, hybrid_fc, lower, upper) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(future_dates)+list(future_dates)[::-1],
        y=list(upper)+list(lower)[::-1],
        fill="toself", fillcolor=C["band"],
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Band"))
    fig.add_trace(go.Scatter(x=future_dates, y=arima_fc, name="ARIMA Forecast",
                             line=dict(color=C["arima"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=future_dates, y=hybrid_fc, name="Hybrid Forecast",
                             line=dict(color=C["hybrid"], width=2.5)))
    fig.update_layout(**_layout(title="⏭️ Short-Term Future Forecast",
        xaxis_title="DateTime", yaxis_title="Forecasted Load (MW)", height=370))
    return fig

def chart_residuals(residuals: pd.Series) -> go.Figure:
    fig = make_subplots(1, 2, subplot_titles=["Residuals Over Time", "Distribution"])
    fig.add_trace(go.Scatter(y=residuals.values, mode="lines",
                             line=dict(color=C["arima"], width=1), name="Residuals"), 1, 1)
    fig.add_trace(go.Histogram(x=residuals.values, nbinsx=50,
                               marker_color=C["actual"], opacity=0.8, name="Dist"), 1, 2)
    fig.update_layout(**_layout(title="📉 ARIMA Residual Analysis", height=300))
    return fig

def chart_training_loss(history: dict) -> go.Figure:
    fig = go.Figure()
    if history.get("loss"):
        fig.add_trace(go.Scatter(y=history["loss"], name="Train Loss",
                                 line=dict(color=C["hybrid"], width=2)))
    if history.get("val_loss"):
        fig.add_trace(go.Scatter(y=history["val_loss"], name="Val Loss",
                                 line=dict(color=C["arima"], width=2, dash="dot")))
    fig.update_layout(**_layout(title="🧠 ANN Learning Curve",
        xaxis_title="Epoch", yaxis_title="Loss (MSE)", height=280))
    return fig

def chart_scatter(actual: np.ndarray, pred: np.ndarray, model="Hybrid") -> go.Figure:
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=pred, mode="markers",
                             marker=dict(color=C["hybrid"], size=3, opacity=0.5),
                             name="Predicted"))
    fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
                             name="Perfect Fit",
                             line=dict(color=C["actual"], dash="dash")))
    fig.update_layout(**_layout(title=f"🎯 {model}: Actual vs Predicted",
        xaxis_title="Actual (MW)", yaxis_title="Predicted (MW)", height=330))
    return fig

def chart_metrics_bar(metrics_dict: dict) -> go.Figure:
    names  = list(metrics_dict.keys())
    colors = [C["arima"], C["hybrid"]]
    metric_keys = ["MAPE","RMSE","MAE"]
    fig = make_subplots(1, 3, subplot_titles=metric_keys)
    for i, mk in enumerate(metric_keys, 1):
        vals = [metrics_dict[n].get(mk, 0) for n in names]
        for j, (name, val) in enumerate(zip(names, vals)):
            fig.add_trace(go.Bar(x=[name], y=[val], name=name,
                                 marker_color=colors[j%len(colors)],
                                 showlegend=(i==1)), 1, i)
    fig.update_layout(**_layout(title="📊 Model Metric Comparison",
        height=300, barmode="group"))
    return fig

def chart_acf_pacf(acf_v, pacf_v, conf_band: float) -> go.Figure:
    lags = np.arange(len(acf_v))
    fig  = make_subplots(1, 2, subplot_titles=["ACF (→ MA order q)", "PACF (→ AR order p)"])
    for col_i, (vals, clr) in enumerate([(acf_v, C["actual"]), (pacf_v, C["hybrid"])], 1):
        fig.add_trace(go.Bar(x=lags, y=vals, marker_color=clr, name=["ACF","PACF"][col_i-1]), 1, col_i)
        for sign in [1, -1]:
            fig.add_hline(y=sign*conf_band, line_dash="dash",
                          line_color="rgba(255,255,255,0.25)", row=1, col=col_i)
    fig.update_layout(**_layout(title="ACF & PACF Analysis", height=320))
    return fig

def chart_decomposition(decomp) -> go.Figure:
    parts  = [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid]
    titles = ["Observed","Trend","Seasonal","Residual"]
    colors = [C["actual"], C["hybrid"], C["arima"], C["ann"]]
    fig    = make_subplots(4, 1, subplot_titles=titles, shared_xaxes=True)
    for i, (comp, clr) in enumerate(zip(parts, colors), 1):
        fig.add_trace(go.Scatter(y=comp.values, mode="lines",
                                 line=dict(color=clr, width=1.1),
                                 showlegend=False), i, 1)
    fig.update_layout(**_layout(title="🌊 Seasonal Decomposition", height=560))
    return fig

def chart_error_dist(actual, hybrid_pred, arima_pred) -> go.Figure:
    h_err = actual - hybrid_pred
    a_err = actual - arima_pred[:len(actual)]
    fig   = go.Figure()
    fig.add_trace(go.Histogram(x=a_err, name="ARIMA Error",
                               marker_color=C["arima"], opacity=0.6, nbinsx=60))
    fig.add_trace(go.Histogram(x=h_err, name="Hybrid Error",
                               marker_color=C["hybrid"], opacity=0.6, nbinsx=60))
    fig.update_layout(**_layout(title="Prediction Error Distribution",
        xaxis_title="Error (MW)", barmode="overlay", height=300))
    return fig
