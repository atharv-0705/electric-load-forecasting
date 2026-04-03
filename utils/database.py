"""
utils/database.py
Lightweight file-based persistence (JSON + joblib).
Stores run history, model configs, and forecast exports.
"""

import json, os, joblib # for model serialization
import pandas as pd
from datetime import datetime

_DB = "db"
os.makedirs(_DB, exist_ok=True)

_RUNS = os.path.join(_DB, "runs.json")
_REG  = os.path.join(_DB, "registry.json")

def _load(f: str) -> list:
    try:
        with open(f) as fp: return json.load(fp)
    except Exception: return []

def _save(f: str, data):
    with open(f, "w") as fp: json.dump(data, fp, indent=2)


class DB:
    # ── Run log ───────────────────────────────────────────────────────────
    @staticmethod
    def log_run(info: dict):
        runs = _load(_RUNS)
        info["ts"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info["id"]  = len(runs) + 1
        runs.append(info)
        _save(_RUNS, runs[-100:])

    @staticmethod
    def get_runs() -> list:
        return _load(_RUNS)

    # ── Model registry ────────────────────────────────────────────────────
    @staticmethod
    def save_model(model, name: str, metrics: dict):
        path = os.path.join(_DB, f"{name}.pkl")
        joblib.dump(model, path)
        reg = _load(_REG)
        reg.append({"name": name, "path": path, "metrics": metrics,
                    "saved_at": datetime.now().isoformat()})
        _save(_REG, reg)

    @staticmethod
    def load_model(name: str):
        path = os.path.join(_DB, f"{name}.pkl")
        return joblib.load(path) if os.path.exists(path) else None

    @staticmethod
    def get_registry() -> list:
        return _load(_REG)

    # ── CSV exports ───────────────────────────────────────────────────────
    @staticmethod
    def save_forecast(df: pd.DataFrame, name: str) -> str:
        path = os.path.join(_DB, f"{name}.csv")
        df.to_csv(path, index=False)
        return path
