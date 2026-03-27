import pandas as pd
import numpy as np
from src.config import DATA_DIR, EVENTS, SENSORS


def load_files(
    events: dict = None,
    max_files_per_event: int = None,
    only_real: bool = True,
    min_sensors: int = 3,
) -> pd.DataFrame:
    if events is None:
        events = EVENTS

    all_dfs = []

    for event_id, name in events.items():
        folder = DATA_DIR / str(event_id)
        files = sorted(folder.glob("*.parquet"))

        if only_real:
            files = [f for f in files
                     if not f.name.startswith(("DRAWN", "SIMULATED"))]

        if max_files_per_event:
            files = files[:max_files_per_event]

        loaded = 0
        for f in files:
            df = pd.read_parquet(f)

            available = [c for c in SENSORS
                         if c in df.columns and df[c].isnull().mean() < 0.5]
            if len(available) < min_sensors:
                continue

            df = df[available + ["class"]].copy()
            df = df.ffill().bfill()

            for col in SENSORS:
                if col not in df.columns:
                    df[col] = np.nan

            df["label"] = (df["class"].fillna(0).astype(int) > 0).astype(int)
            df["event_type"] = event_id
            df["source"] = f.name
            df = df.drop(columns=["class"])

            all_dfs.append(df)
            loaded += 1

        print(f"  [{event_id}] {name}: Loaded {loaded} files")

    return pd.concat(all_dfs, ignore_index=True)