import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.config import DATA_DIR, EVENTS, SENSORS, RANDOM_STATE


def load_and_normalize(
    max_files_per_event: int = None,
    only_real: bool = True,
    min_sensors: int = 2,          # понизили с 3 до 2 — поможет событию 9
    fit_scaler: bool = True,
    scaler_path: str = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    from src.loader import load_files

    df = load_files(
        max_files_per_event=max_files_per_event,
        only_real=only_real,
        min_sensors=min_sensors,
    )

    scaler = StandardScaler()

    sensor_data = df[SENSORS].copy()

    if fit_scaler:
        normal_mask = df["label"] == 0
        scaler.fit(sensor_data[normal_mask].fillna(0))
        print("Scaler is trained on real data")

        if scaler_path:
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved in: {scaler_path}")
    else:
        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            print(f"Scaler loaded from: {scaler_path}")
        else:
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    df[SENSORS] = scaler.transform(sensor_data.fillna(0))

    return df, scaler


def print_normalization_stats(df: pd.DataFrame):
    print("=== Normalization stats ===")
    print("(mean ~0, std ~1 — if everything's good)\n")
    stats = df[SENSORS].agg(["mean", "std", "min", "max"]).round(3)
    print(stats.to_string())