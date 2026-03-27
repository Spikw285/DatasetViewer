import pandas as pd
import numpy as np
from src.config import SENSORS, WINDOW_SIZE, WINDOW_STEP


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for source_file, group in df.groupby("source"):
        group = group.reset_index(drop=True)
        n = len(group)

        for start in range(0, n - WINDOW_SIZE, WINDOW_STEP):
            window = group.iloc[start: start + WINDOW_SIZE]

            anomaly_ratio = (window["label"] == 1).mean()
            label = 1 if anomaly_ratio > 0.1 else 0
            event_type = window["event_type"].iloc[0]

            row = {"label": label, "event_type": event_type, "source": source_file}

            for col in SENSORS:
                vals = window[col].dropna()

                if len(vals) < WINDOW_SIZE * 0.5:
                    for stat in ("mean", "std", "min", "max", "trend", "range"):
                        row[f"{col}_{stat}"] = np.nan
                    continue

                row[f"{col}_mean"]  = vals.mean()
                row[f"{col}_std"]   = vals.std()
                row[f"{col}_min"]   = vals.min()
                row[f"{col}_max"]   = vals.max()
                row[f"{col}_range"] = vals.max() - vals.min()

                x = np.arange(len(vals))
                row[f"{col}_trend"] = np.polyfit(x, vals, 1)[0]

            records.append(row)

    result = pd.DataFrame(records)
    print(f"Dataframes extracted: {len(result):,}")
    print(f"Signs: {len([c for c in result.columns if c not in ('label', 'event_type')])}")
    return result