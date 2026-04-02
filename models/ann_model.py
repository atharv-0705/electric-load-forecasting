"""
models/ann_model.py
ANN for residual learning — optimized for speed.
Default: 100 epochs, small network, early stopping after 10 rounds.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing  import MinMaxScaler
import joblib
import os


class ANNModel:
    def __init__(self, layers=(64, 32, 16), epochs: int = 100, lr: float = 0.001):
        self.layers   = layers
        self.epochs   = epochs
        self.lr       = lr
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model    = None
        self.history_ = {"loss": [], "val_loss": []}
        self._backend = "sklearn"

    # ── Build sklearn MLP ─────────────────────────────────────────────────
    def _build_sklearn(self):
        return MLPRegressor(
            hidden_layer_sizes=self.layers,
            activation="relu",
            solver="adam",
            max_iter=self.epochs,
            random_state=42,
            early_stopping=True,       # stops when val loss plateaus
            validation_fraction=0.1,
            n_iter_no_change=10,       # stop after 10 rounds no improvement
            learning_rate_init=self.lr,
            tol=1e-4,
            verbose=False,
        )

    # ── Try Keras upgrade ─────────────────────────────────────────────────
    def _try_keras(self, input_dim: int):
        try:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            from tensorflow.keras.models     import Sequential
            from tensorflow.keras.layers     import Dense, Dropout
            from tensorflow.keras.optimizers import Adam

            m = Sequential([
                Dense(self.layers[0], activation="relu", input_shape=(input_dim,)),
                Dropout(0.1),
                Dense(self.layers[1], activation="relu"),
                Dense(self.layers[2] if len(self.layers) > 2 else 16, activation="relu"),
                Dense(1),
            ])
            m.compile(optimizer=Adam(self.lr), loss="mse")
            return m, "keras"
        except ImportError:
            return None, "sklearn"

    # ── Fit ───────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        Xs = self.scaler_X.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        keras_model, backend = self._try_keras(X.shape[1])

        if backend == "keras":
            from tensorflow.keras.callbacks import EarlyStopping
            self.model    = keras_model
            self._backend = "keras"
            hist = self.model.fit(
                Xs, ys,
                epochs=self.epochs,
                batch_size=64,          # larger batch = faster
                validation_split=0.1,
                verbose=0,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                ],
            )
            self.history_ = {
                "loss":     hist.history["loss"],
                "val_loss": hist.history.get("val_loss", []),
            }
            final_loss = float(hist.history["loss"][-1])
        else:
            self.model    = self._build_sklearn()
            self._backend = "sklearn"
            self.model.fit(Xs, ys)
            lc = getattr(self.model, "loss_curve_", [0.1])
            self.history_ = {"loss": lc, "val_loss": []}
            final_loss    = float(lc[-1]) if lc else 0.0

        return {
            "backend":    self._backend,
            "final_loss": round(final_loss, 6),
            "epochs_run": len(self.history_["loss"]),
        }

    # ── Predict ───────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_X.transform(X)
        if self._backend == "keras":
            p = self.model.predict(Xs, verbose=0).flatten()
        else:
            p = self.model.predict(Xs)
        return self.scaler_y.inverse_transform(p.reshape(-1, 1)).flatten()

    # ── Save ──────────────────────────────────────────────────────────────
    def save(self, path: str = "db/ann_model"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if self._backend == "keras":
            try:
                self.model.save(path + ".h5")
            except Exception:
                pass
        else:
            joblib.dump(self.model, path + "_sklearn.pkl")
        joblib.dump(self.scaler_X, path + "_sx.pkl")
        joblib.dump(self.scaler_y, path + "_sy.pkl")
