"""
utils/metrics.py
Standard forecasting accuracy metrics.
"""

import numpy as np
import pandas as pd


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    a, p = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if len(a) == 0:
        return {k: 0.0 for k in ["MAPE", "RMSE", "MAE", "R2", "SMAPE", "NRMSE"]}

    err  = a - p
    nz   = a != 0
    mape = float(np.mean(np.abs(err[nz] / a[nz])) * 100) if nz.any() else 0.0
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    denom = (np.abs(a) + np.abs(p)) / 2
    nd    = denom != 0
    smape = float(np.mean(np.abs(err[nd]) / denom[nd]) * 100) if nd.any() else 0.0
    ss_res, ss_tot = np.sum(err**2), np.sum((a - a.mean())**2)
    r2    = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
    rng   = a.max() - a.min()
    nrmse = float(rmse / rng * 100) if rng != 0 else 0.0

    return {"MAPE": round(mape,4), "RMSE": round(rmse,4), "MAE": round(mae,4),
            "R2": round(r2,4), "SMAPE": round(smape,4), "NRMSE": round(nrmse,4)}


def metrics_table(metrics_dict: dict) -> pd.DataFrame:
    rows = [{"Model": m, **v} for m, v in metrics_dict.items()]
    return pd.DataFrame(rows)


def pct_improvement(base: float, improved: float) -> float:
    return round((base - improved) / base * 100, 2) if base != 0 else 0.0
