"""
utils/preprocessor.py
Handles cleaning, feature engineering, scaling, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


class DataPreprocessor:
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.is_fitted = False

    # ── Validation ────────────────────────────────────────────────────────
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        required = ["datetime", "load_mw"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            date_c = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            load_c = [c for c in df.columns if any(k in c.lower() for k in ["load","power","mw","kw"])]
            if date_c and load_c:
                return True, f"Auto-detected: date='{date_c[0]}', load='{load_c[0]}'"
            return False, f"Missing columns: {missing}"
        return True, "OK"

    # ── Cleaning ──────────────────────────────────────────────────────────
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        # Interpolate missing
        df["load_mw"] = df["load_mw"].interpolate(method="linear")
        # IQR clip
        Q1, Q3 = df["load_mw"].quantile(0.25), df["load_mw"].quantile(0.75)
        IQR = Q3 - Q1
        df["load_mw"] = df["load_mw"].clip(Q1 - 2.5*IQR, Q3 + 2.5*IQR)
        return df

    # ── Feature engineering ───────────────────────────────────────────────
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour"]        = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["month"]       = df["datetime"].dt.month
        df["day_of_year"] = df["datetime"].dt.dayofyear
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
        df["quarter"]     = df["datetime"].dt.quarter
        # Cyclical encoding
        df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
        df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
        df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
        df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
        df["dow_sin"]   = np.sin(2*np.pi*df["day_of_week"]/7)
        df["dow_cos"]   = np.cos(2*np.pi*df["day_of_week"]/7)
        # Lag features
        for lag in [1, 2, 3, 24, 48, 168]:
            df[f"lag_{lag}"] = df["load_mw"].shift(lag)
        # Rolling stats
        for w in [6, 12, 24]:
            df[f"roll_mean_{w}"] = df["load_mw"].shift(1).rolling(w).mean()
            df[f"roll_std_{w}"]  = df["load_mw"].shift(1).rolling(w).std()
        return df.dropna().reset_index(drop=True)

    # ── ARIMA input ───────────────────────────────────────────────────────
    def prepare_arima_series(self, df: pd.DataFrame) -> pd.Series:
        s = df.set_index("datetime")["load_mw"] if "datetime" in df.columns else df["load_mw"]
        return s

    # ── ANN input ─────────────────────────────────────────────────────────
    def prepare_ann_features(self, df: pd.DataFrame):
        exclude = {"datetime", "load_mw"}
        feat_cols = [c for c in df.columns if c not in exclude][:20]
        X = self.scaler_X.fit_transform(df[feat_cols].values)
        y = self.scaler_y.fit_transform(df["load_mw"].values.reshape(-1, 1)).flatten()
        self.is_fitted = True
        self._feat_cols = feat_cols
        return X, y, feat_cols

    def transform_features(self, df: pd.DataFrame, feat_cols: list) -> np.ndarray:
        available = [c for c in feat_cols if c in df.columns]
        return self.scaler_X.transform(df[available].values)

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return self.scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()

    # ── Split ─────────────────────────────────────────────────────────────
    def train_test_split(self, df: pd.DataFrame, test_ratio: float = 0.2):
        n = int(len(df) * (1 - test_ratio))
        return df.iloc[:n].copy(), df.iloc[n:].copy()

    # ── Stats ─────────────────────────────────────────────────────────────
    def summary_stats(self, df: pd.DataFrame) -> dict:
        return {
            "records":    len(df),
            "date_range": f"{df['datetime'].min().date()} → {df['datetime'].max().date()}" if "datetime" in df.columns else "N/A",
            "missing":    int(df.isnull().sum().sum()),
            "mean_load":  round(df["load_mw"].mean(), 2),
            "max_load":   round(df["load_mw"].max(), 2),
            "min_load":   round(df["load_mw"].min(), 2),
            "std_load":   round(df["load_mw"].std(), 2),
            "peak_hour":  int(df.groupby("hour")["load_mw"].mean().idxmax()) if "hour" in df.columns else "N/A",
        }
