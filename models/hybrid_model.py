"""
models/hybrid_model.py
Hybrid ARIMA + ANN — optimized for speed.

Speed improvements:
  - ARIMA fits on max 2000 rows (representative sample)
  - ANN uses max 15 features
  - Progress callbacks for responsive UI
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from typing import Optional, Callable

from models.arima_model import ARIMAModel
from models.ann_model   import ANNModel
from utils.preprocessor import DataPreprocessor
from utils.metrics      import compute_metrics


# Max rows fed to ARIMA — keeps it fast on large datasets
ARIMA_MAX_ROWS = 2000
# Max feature columns for ANN
ANN_MAX_FEATS  = 15


class HybridModel:
    def __init__(
        self,
        arima_order=(2, 1, 2),
        ann_layers=(64, 32, 16),
        epochs=100,
        auto_arima=True,
    ):
        self.arima_order = arima_order
        self.ann_layers  = ann_layers
        self.epochs      = epochs
        self.auto_arima  = auto_arima

        self.arima = ARIMAModel(order=arima_order)
        self.ann   = ANNModel(layers=ann_layers, epochs=epochs)
        self.prep  = DataPreprocessor()

        self.is_fitted   = False
        self._feat_cols  = []
        self.arima_info_ = {}
        self.ann_info_   = {}

    # ── Training ──────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame,
            progress: Optional[Callable] = None) -> dict:

        def _p(f, m):
            if progress:
                progress(f, m)

        # 1. Prepare ARIMA series (subsample for speed)
        _p(0.05, "Preparing time series...")
        ts_full = self.prep.prepare_arima_series(df)

        # Subsample: use last N rows so ARIMA stays fast
        if len(ts_full) > ARIMA_MAX_ROWS:
            ts = ts_full.iloc[-ARIMA_MAX_ROWS:]
            _p(0.08, f"Subsampled to {ARIMA_MAX_ROWS} rows for ARIMA speed...")
        else:
            ts = ts_full

        # 2. Auto-select ARIMA order (fast — 8 candidates)
        if self.auto_arima:
            _p(0.15, "Selecting ARIMA order (fast mode — 8 candidates)...")
            try:
                self.arima_order = self.arima.auto_order(ts)
            except Exception:
                self.arima_order = (2, 1, 2)

        # 3. Fit ARIMA
        _p(0.35, f"Fitting ARIMA{self.arima_order}...")
        self.arima_info_ = self.arima.fit(ts, order=self.arima_order)

        # 4. Extract residuals
        _p(0.50, "Extracting ARIMA residuals...")
        residuals = self.arima.residuals_

        # 5. Engineer features (on full training data)
        _p(0.60, "Engineering features for ANN...")
        df_f = self.prep.engineer_features(df)

        # Align residuals to df_f
        if "datetime" in df_f.columns and hasattr(residuals, "index"):
            res_aligned = residuals.reindex(df_f["datetime"]).fillna(0).values
        else:
            n = min(len(residuals), len(df_f))
            res_aligned = np.concatenate(
                [residuals.values[:n], np.zeros(max(0, len(df_f) - n))]
            )

        # 6. Prepare features (limit columns for speed)
        exclude   = {"datetime", "load_mw"}
        feat_cols = [c for c in df_f.columns
                     if c not in exclude][:ANN_MAX_FEATS]
        self._feat_cols = feat_cols

        X     = self.prep.scaler_X.fit_transform(df_f[feat_cols].values)
        y_res = res_aligned[:len(X)]
        self.prep.scaler_y.fit(df_f["load_mw"].values.reshape(-1, 1))

        # 7. Train ANN
        _p(0.72, f"Training ANN ({self.epochs} epochs max, early stop)...")
        self.ann_info_ = self.ann.fit(X, y_res)

        self.is_fitted = True
        _p(1.00, "Training complete!")

        return {"arima": self.arima_info_, "ann": self.ann_info_}

    # ── Predict ───────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df_f     = self.prep.engineer_features(df)
        available = [c for c in self._feat_cols if c in df_f.columns]
        X        = self.prep.scaler_X.transform(df_f[available].values)
        ann_corr = self.ann.predict(X)

        arima_v = (
            self.arima.fitted_vals_.values
            if hasattr(self.arima.fitted_vals_, "values")
            else np.array(self.arima.fitted_vals_)
        )

        n = min(len(arima_v), len(ann_corr))
        return arima_v[:n] + ann_corr[:n]

    # ── Forecast ──────────────────────────────────────────────────────────
    def forecast(self, steps: int = 24,
                 last_df: Optional[pd.DataFrame] = None) -> dict:
        arima_fc, lower, upper = self.arima.forecast(steps)

        ann_corr = np.zeros(steps)
        if last_df is not None and len(last_df) > 0:
            try:
                df_f      = self.prep.engineer_features(last_df.tail(steps * 3))
                available = [c for c in self._feat_cols if c in df_f.columns]
                X         = self.prep.scaler_X.transform(df_f[available].values)
                if len(X) > 0:
                    corr = (
                        self.ann.predict(X[-steps:])
                        if len(X) >= steps
                        else self.ann.predict(X)
                    )
                    ann_corr[:len(corr)] = corr
            except Exception:
                pass

        return {
            "arima_fc":  arima_fc,
            "ann_corr":  ann_corr,
            "hybrid_fc": arima_fc + ann_corr,
            "lower":     lower,
            "upper":     upper,
        }

    # ── Evaluate ──────────────────────────────────────────────────────────
    def evaluate(self, df_test: pd.DataFrame) -> dict:
        actual      = df_test["load_mw"].values
        hybrid_pred = self.predict(df_test)
        n           = min(len(actual), len(hybrid_pred))
        actual      = actual[:n]
        hybrid_pred = hybrid_pred[:n]

        arima_fc, _, _ = self.arima.forecast(steps=n)

        return {
            "ARIMA":              compute_metrics(actual, arima_fc[:n]),
            "Hybrid (ARIMA+ANN)": compute_metrics(actual, hybrid_pred),
        }
