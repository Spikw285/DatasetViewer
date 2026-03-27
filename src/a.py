import pandas as pd
try:
    df = pd.read_parquet("../outputs/features.parquet")
    print("Файл живой! Можно не переделывать!")
    print(df.head())
except Exception as e:
    print(f"Ошибка: {e}")