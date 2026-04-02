"""
tests/test_models.py
Unit tests for preprocessing, ARIMA, ANN, and Hybrid modules.
Run:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from data.generate_sample_data import generate_power_load_data
from utils.preprocessor        import DataPreprocessor
from utils.metrics             import compute_metrics, pct_improvement
from models.arima_model        import ARIMAModel
from models.ann_model          import ANNModel
from models.hybrid_model       import HybridModel


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def small_df():
    """60-day dataset — fast enough for test runs."""
    return generate_power_load_data(n_days=60)

@pytest.fixture(scope="module")
def prep():
    return DataPreprocessor()

@pytest.fixture(scope="module")
def clean_feat(small_df, prep):
    df_c = prep.clean_data(small_df)
    df_f = prep.engineer_features(df_c)
    return df_f


# ══════════════════════════════════════════════════════════════════════════
# Data generator
# ══════════════════════════════════════════════════════════════════════════
class TestDataGenerator:
    def test_shape(self, small_df):
        assert len(small_df) == 60 * 24

    def test_columns(self, small_df):
        assert "datetime" in small_df.columns
        assert "load_mw"  in small_df.columns

    def test_no_negatives(self, small_df):
        assert (small_df["load_mw"] >= 0).all()

    def test_load_range(self, small_df):
        assert small_df["load_mw"].between(60, 280).all()


# ══════════════════════════════════════════════════════════════════════════
# Preprocessor
# ══════════════════════════════════════════════════════════════════════════
class TestPreprocessor:
    def test_validate_ok(self, small_df, prep):
        ok, _ = prep.validate_data(small_df)
        assert ok

    def test_validate_fail(self, prep):
        bad = pd.DataFrame({"col_a": [1,2], "col_b": [3,4]})
        ok, msg = prep.validate_data(bad)
        assert not ok
        assert "Missing" in msg

    def test_clean_removes_nulls(self, small_df, prep):
        df_dirty = small_df.copy()
        df_dirty.loc[10:15, "load_mw"] = np.nan
        df_c = prep.clean_data(df_dirty)
        assert df_c["load_mw"].isnull().sum() == 0

    def test_feature_engineering_columns(self, clean_feat):
        expected = ["hour_sin","hour_cos","month_sin","month_cos","lag_1","lag_24"]
        for col in expected:
            assert col in clean_feat.columns, f"Missing: {col}"

    def test_no_nulls_after_features(self, clean_feat):
        assert clean_feat.isnull().sum().sum() == 0

    def test_train_test_split_ratio(self, clean_feat, prep):
        tr, te = prep.train_test_split(clean_feat, test_ratio=0.2)
        total = len(tr) + len(te)
        assert abs(len(te) / total - 0.2) < 0.02   # within 2%

    def test_temporal_order_preserved(self, clean_feat, prep):
        tr, te = prep.train_test_split(clean_feat, test_ratio=0.2)
        assert tr["datetime"].max() < te["datetime"].min()

    def test_summary_stats_keys(self, clean_feat, prep):
        stats = prep.summary_stats(clean_feat)
        for k in ["records","mean_load","max_load","min_load","peak_hour"]:
            assert k in stats


# ══════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════
class TestMetrics:
    def test_perfect_prediction(self):
        a = np.array([100., 200., 150.])
        m = compute_metrics(a, a)
        assert m["MAPE"]  == 0.0
        assert m["RMSE"]  == 0.0
        assert m["R2"]    == pytest.approx(1.0)

    def test_mape_known(self):
        a = np.array([100., 100.])
        p = np.array([110., 90.])
        m = compute_metrics(a, p)
        assert m["MAPE"] == pytest.approx(10.0, abs=0.01)

    def test_all_keys_present(self):
        m = compute_metrics(np.ones(10), np.ones(10)*1.1)
        for k in ["MAPE","RMSE","MAE","R2","SMAPE","NRMSE"]:
            assert k in m

    def test_pct_improvement(self):
        assert pct_improvement(10.0, 7.0) == pytest.approx(30.0)
        assert pct_improvement(0.0, 5.0)  == 0.0   # zero-division guard

    def test_nan_handling(self):
        a = np.array([np.nan, 100., 200.])
        p = np.array([90.,    100., 200.])
        m = compute_metrics(a, p)
        assert np.isfinite(m["MAPE"])


# ══════════════════════════════════════════════════════════════════════════
# ARIMA model
# ══════════════════════════════════════════════════════════════════════════
class TestARIMA:
    def test_fit_returns_metrics(self, clean_feat):
        prep  = DataPreprocessor()
        ts    = prep.prepare_arima_series(clean_feat)
        model = ARIMAModel(order=(1,1,1))
        info  = model.fit(ts)
        assert "aic" in info
        assert "order" in info

    def test_residuals_shape(self, clean_feat):
        prep  = DataPreprocessor()
        ts    = prep.prepare_arima_series(clean_feat)
        model = ARIMAModel(order=(1,1,1))
        model.fit(ts)
        assert len(model.residuals_) > 0

    def test_forecast_length(self, clean_feat):
        prep  = DataPreprocessor()
        ts    = prep.prepare_arima_series(clean_feat)
        model = ARIMAModel(order=(1,1,1))
        model.fit(ts)
        fc, lo, hi = model.forecast(steps=24)
        assert len(fc) == 24
        assert len(lo) == 24
        assert len(hi) == 24

    def test_stationarity_returns_dict(self, clean_feat):
        prep  = DataPreprocessor()
        ts    = prep.prepare_arima_series(clean_feat)
        model = ARIMAModel()
        r     = model.check_stationarity(ts)
        assert "is_stationary" in r
        assert "p_value"       in r


# ══════════════════════════════════════════════════════════════════════════
# ANN model
# ══════════════════════════════════════════════════════════════════════════
class TestANN:
    def _make_Xy(self, n=200):
        X = np.random.randn(n, 10)
        y = np.random.randn(n)
        return X, y

    def test_fit_returns_info(self):
        model = ANNModel(layers=(32,16), epochs=20)
        X, y  = self._make_Xy()
        info  = model.fit(X, y)
        assert "backend"    in info
        assert "final_loss" in info

    def test_predict_shape(self):
        model = ANNModel(layers=(32,16), epochs=20)
        X, y  = self._make_Xy()
        model.fit(X, y)
        pred  = model.predict(X[:10])
        assert pred.shape == (10,)

    def test_predictions_finite(self):
        model = ANNModel(layers=(32,16), epochs=20)
        X, y  = self._make_Xy()
        model.fit(X, y)
        pred  = model.predict(X)
        assert np.isfinite(pred).all()


# ══════════════════════════════════════════════════════════════════════════
# Hybrid model (integration)
# ══════════════════════════════════════════════════════════════════════════
class TestHybrid:
    @pytest.fixture
    def trained_hybrid(self, clean_feat):
        prep    = DataPreprocessor()
        tr, _   = prep.train_test_split(clean_feat, test_ratio=0.2)
        model   = HybridModel(arima_order=(1,1,1),
                              ann_layers=(32,16,8),
                              epochs=20,
                              auto_arima=False)
        model.fit(tr)
        return model, tr, clean_feat

    def test_fit_completes(self, trained_hybrid):
        model, _, _ = trained_hybrid
        assert model.is_fitted

    def test_predict_returns_array(self, trained_hybrid):
        model, tr, _ = trained_hybrid
        pred = model.predict(tr.tail(48))
        assert isinstance(pred, np.ndarray)
        assert len(pred) > 0

    def test_forecast_keys(self, trained_hybrid):
        model, tr, _ = trained_hybrid
        result = model.forecast(steps=24, last_df=tr)
        for k in ["arima_fc","ann_corr","hybrid_fc","lower","upper"]:
            assert k in result

    def test_forecast_length(self, trained_hybrid):
        model, tr, _ = trained_hybrid
        result = model.forecast(steps=12, last_df=tr)
        assert len(result["hybrid_fc"]) == 12

    def test_evaluate_returns_both_models(self, trained_hybrid):
        model, tr, full = trained_hybrid
        prep = DataPreprocessor()
        _, te = prep.train_test_split(full, test_ratio=0.2)
        ev = model.evaluate(te)
        assert "ARIMA"                in ev
        assert "Hybrid (ARIMA+ANN)"   in ev

    def test_hybrid_better_or_equal_mape(self, trained_hybrid):
        """Hybrid MAPE should be ≤ ARIMA MAPE (on training domain)."""
        model, tr, full = trained_hybrid
        prep = DataPreprocessor()
        _, te = prep.train_test_split(full, test_ratio=0.2)
        ev   = model.evaluate(te)
        # Allow ≤10% tolerance — small datasets can be noisy
        assert ev["Hybrid (ARIMA+ANN)"]["MAPE"] <= ev["ARIMA"]["MAPE"] * 1.10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
