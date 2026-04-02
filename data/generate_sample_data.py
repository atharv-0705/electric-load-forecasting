"""
data/generate_sample_data.py
Generates realistic synthetic hourly power load data with:
  - Daily seasonality (peak hours 9-21)
  - Weekly seasonality (lower on weekends)
  - Annual seasonality (summer/winter peaks)
  - Temperature correlation
  - Random noise
"""

import numpy as np
import pandas as pd
from datetime import datetime


def generate_power_load_data(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    periods    = n_days * 24
    date_range = pd.date_range(start=datetime(2023, 1, 1), periods=periods, freq="1h")
    t          = np.arange(periods)

    trend   = 100 + 0.005 * t
    annual  = 30 * np.cos(2 * np.pi * t / (365 * 24) - np.pi)
    dow     = np.array([d.weekday() for d in date_range])
    weekly  = np.where(dow >= 5, -15, 5)
    hour    = np.array([d.hour for d in date_range])
    daily   = (
        20 * np.sin(np.pi * (hour - 6) / 12) * (hour >= 6) * (hour <= 22)
        + 5 * np.sin(2 * np.pi * hour / 24)
    )
    temp_fx = 10 * np.sin(2 * np.pi * t / (365 * 24) + np.pi / 4)
    noise   = np.random.normal(0, 5, periods)

    load        = np.clip(trend + annual + weekly + daily + temp_fx + noise, 80, 250)
    temperature = (25 + 10 * np.sin(2 * np.pi * t / (365 * 24) - np.pi / 2)
                   + np.random.normal(0, 2, periods))
    humidity    = np.clip(
        60 + 15 * np.sin(2 * np.pi * t / (365 * 24)) + np.random.normal(0, 5, periods),
        20, 100
    )

    return pd.DataFrame({
        "datetime":      date_range,
        "load_mw":       np.round(load, 2),
        "temperature_c": np.round(temperature, 1),
        "humidity_pct":  np.round(humidity, 1),
        "hour":          hour,
        "day_of_week":   dow,
        "month":         [d.month for d in date_range],
        "is_weekend":    (dow >= 5).astype(int),
    })


if __name__ == "__main__":
    df = generate_power_load_data()
    df.to_csv("data/sample_power_load.csv", index=False)
    print(f"Generated {len(df)} records")
    print(df.head())
