import pandas as pd

def read_parquet_info(path: str) -> None:
    """Quick inspection of any parquet file."""
    df = pd.read_parquet(path)

    print("=== Structure ===")
    print(f"Shape: {df.shape}")
    print(df.dtypes)

    print("\n=== First 5 rows ===")
    print(df.head())

    print("\n=== Descriptive statistics ===")
    print(df.describe().round(3))