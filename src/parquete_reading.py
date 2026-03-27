import pandas as pd
df_check = pd.read_parquet("../outputs/features.parquet")

print("=== Structure ===")
print(df_check.shape)
print(df_check.dtypes)

print("\n=== First 5 rows ===")
df_check.head()
print("=== Descriptive characteristic ===")
df_check.describe().round(3)